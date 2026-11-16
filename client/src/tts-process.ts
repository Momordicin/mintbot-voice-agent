// Child processes of the GPT-SoVITS speech-synthesis component (tts.ts): 7za (7zip-bin)
// lists the unpacked size of an archive and extracts it; the service starts the package's
// runtime/python.exe api_v2.py on a loopback port and waits until it answers HTTP, and a
// probe does the same, then ends its process tree. api_v2.py is started with -c and the
// config file the caller passes, so it never reads or rewrites the package's own
// GPT_SoVITS/configs/tts_infer.yaml. Failures are ProcessFailure, with the last lines the
// process wrote to stderr.
import { spawn, type ChildProcess } from "node:child_process";
import * as path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { path7za } from "7zip-bin";

const HOST = "127.0.0.1";
const PROBE_TIMEOUT_MS = 300_000;
const PROBE_POLL_MS = 1000;
const PROBE_REQUEST_TIMEOUT_MS = 2000;
const KILL_WAIT_MS = 5000;
const STDERR_TAIL_LINES = 20;

export class ProcessFailure extends Error {
  readonly stderr: string[];
  constructor(message: string, stderr: string[]) {
    super(message);
    this.stderr = stderr;
  }
}

function collectTail(stream: NodeJS.ReadableStream | null, tail: string[]): void {
  let rest = "";
  stream?.setEncoding("utf-8");
  stream?.on("data", (chunk: string) => {
    const lines = (rest + chunk).split(/\r?\n/);
    rest = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim() === "") continue;
      tail.push(line);
      if (tail.length > STDERR_TAIL_LINES) tail.shift();
    }
  });
}

function waitForExit(child: ChildProcess): Promise<number | null> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(child.exitCode);
  return new Promise((resolve) => {
    child.once("exit", (code) => resolve(code));
    child.once("error", () => resolve(null));
  });
}

// taskkill /T: api_v2.py and 7za may have started children of their own
async function killTree(child: ChildProcess): Promise<void> {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return;
  const exited = waitForExit(child);
  const killer = spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
  await waitForExit(killer);
  await Promise.race([exited, sleep(KILL_WAIT_MS)]);
}

async function runSevenZip(args: string[], signal: AbortSignal, onStdout?: (chunk: string) => void): Promise<void> {
  signal.throwIfAborted();
  const child = spawn(path7za, args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  const stderr: string[] = [];
  collectTail(child.stderr, stderr);
  child.stdout?.setEncoding("utf-8");
  child.stdout?.on("data", (chunk: string) => onStdout?.(chunk));
  const onAbort = () => {
    killTree(child).catch(() => {});
  };
  signal.addEventListener("abort", onAbort, { once: true });
  try {
    const code = await waitForExit(child);
    signal.throwIfAborted();
    if (code !== 0) throw new ProcessFailure(`7za exited with code ${code}`, stderr);
  } finally {
    signal.removeEventListener("abort", onAbort);
    await killTree(child);
  }
}

// Total size of the files in the archive, from the last line of `7za l`.
export async function sevenZipUnpackedSize(archive: string, signal: AbortSignal): Promise<number> {
  let output = "";
  await runSevenZip(["l", archive], signal, (chunk) => {
    output += chunk;
  });
  const totals = output
    .split(/\r?\n/)
    .map((line) => /(\d+)\s+(\d+)\s+\d+ files?\b/.exec(line))
    .filter((match): match is RegExpExecArray => match !== null);
  const last = totals.at(-1);
  if (!last) throw new ProcessFailure("7za listing has no total line", []);
  return Number(last[1]);
}

// onPercent: rising whole percentages from 7za's progress output.
export async function sevenZipExtract(archive: string, dest: string, signal: AbortSignal,
                                      onPercent: (percent: number) => void): Promise<void> {
  let best = 0;
  await runSevenZip(["x", archive, `-o${dest}`, "-y", "-bsp1", "-bso0"], signal, (chunk) => {
    for (const match of chunk.matchAll(/(\d+)%/g)) {
      const percent = Number(match[1]);
      if (percent > best && percent <= 100) {
        best = percent;
        onPercent(percent);
      }
    }
  });
}

async function answersHttp(url: string, signal?: AbortSignal): Promise<boolean> {
  const timeout = AbortSignal.timeout(PROBE_REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
    await response.body?.cancel();
    return true;
  } catch {
    return false;
  }
}

// A running api_v2.py. stderr: its last error output lines, kept up to date. exited:
// resolves with a message when the process has ended and its stderr is closed (the tail is
// complete). stop(): ends its process tree.
export interface TtsServiceProcess {
  url: string;
  stderr: string[];
  exited: Promise<string>;
  stop(): Promise<void>;
}

// Starts <dir>/runtime/python.exe api_v2.py -c <config> on 127.0.0.1:port and resolves once
// it answers HTTP (at most 300 s); on failure or abort the process tree is ended. Fails at
// once when something already answers on the port. config: absolute path of the YAML file
// api_v2.py reads and writes instead of the package's own.
export async function startTtsService(dir: string, config: string, port: number, signal: AbortSignal): Promise<TtsServiceProcess> {
  const url = `http://${HOST}:${port}/`;
  if (await answersHttp(url, signal)) throw new ProcessFailure(`port ${port} is already in use`, []);
  signal.throwIfAborted();
  const child = spawn(path.join(dir, "runtime", "python.exe"), ["api_v2.py", "-a", HOST, "-p", String(port), "-c", config],
                      { cwd: dir, windowsHide: true, stdio: ["ignore", "ignore", "pipe"], env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
  const stderr: string[] = [];
  collectTail(child.stderr, stderr);
  let exitMessage: string | null = null;
  let onExit: (message: string) => void = () => {};
  const exited = new Promise<string>((resolve) => {
    onExit = resolve;
  });
  child.once("exit", (code) => {
    exitMessage = `api_v2.py exited with code ${code}`;
  });
  child.once("close", (code) => {
    onExit(exitMessage ?? `api_v2.py exited with code ${code}`);
  });
  child.once("error", (err) => {
    exitMessage = `api_v2.py failed to start: ${err.message}`;
    onExit(exitMessage);
  });
  const deadline = Date.now() + PROBE_TIMEOUT_MS;
  try {
    for (;;) {
      signal.throwIfAborted();
      if (exitMessage !== null) throw new ProcessFailure(exitMessage, stderr);
      if (Date.now() > deadline) throw new ProcessFailure(`api_v2.py did not answer within ${PROBE_TIMEOUT_MS / 1000} s`, stderr);
      if (await answersHttp(url, signal)) {
        if (exitMessage !== null) throw new ProcessFailure(exitMessage, stderr);
        return { url, stderr, exited, stop: () => killTree(child) };
      }
      await sleep(PROBE_POLL_MS, undefined, { signal });
    }
  } catch (err) {
    await killTree(child);
    throw err;
  }
}

// The test run: startTtsService(), then its process tree is ended.
export async function probeTtsApi(dir: string, config: string, port: number, signal: AbortSignal): Promise<void> {
  const service = await startTtsService(dir, config, port, signal);
  await service.stop();
}
