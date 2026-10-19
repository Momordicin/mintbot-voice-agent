// One mintbot-voice-agent process: spawn, ready line, REST, SSE, shutdown.
// VoiceAgentSupervisor builds on this; use it directly only for tools and tests.
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import * as readline from "node:readline";

import { AgentHttpError, type AgentEvent, type ReadyLine } from "./protocol.ts";
import { readSse } from "./sse.ts";

export interface AgentProcessOptions {
  file: string; // the executable
  args: string[];
  readyTimeoutMs?: number; // default 60000 (loading the models takes a few seconds)
  onStderr?: (line: string) => void; // the agent logs to stderr
}

export interface AgentProcess {
  readonly child: ChildProcess;
  readonly ready: ReadyLine;
  readonly baseUrl: string;
  // Resolves with the exit code (null when killed) once the process has exited.
  readonly exited: Promise<number | null>;
  call<T = any>(method: "GET" | "POST" | "DELETE", path: string): Promise<T>;
  // Connects to the agent's SSE stream; resolves once connected. The stream ends
  // when the connection closes or `signal` aborts.
  openEvents(signal?: AbortSignal): Promise<AsyncGenerator<AgentEvent>>;
  // Same, connecting on first iteration.
  events(signal?: AbortSignal): AsyncGenerator<AgentEvent>;
  // POST /v1/shutdown, then kill if it has not exited within timeoutMs.
  shutdown(timeoutMs?: number): Promise<void>;
}

function forwardLines(stream: NodeJS.ReadableStream | null, onLine: (line: string) => void) {
  if (!stream) return;
  const rl = readline.createInterface({ input: stream });
  rl.on("line", (line) => {
    if (line.trim()) onLine(line);
  });
}

export async function startAgentProcess(o: AgentProcessOptions): Promise<AgentProcess> {
  // A new token per process; it travels in the environment, never on the command line.
  const token = randomBytes(32).toString("hex");
  // Not detached: libuv puts the child in a kill-on-close job object, and the agent
  // also exits when its stdin pipe closes, so it never outlives this process.
  const child = spawn(o.file, o.args, {
    env: { ...process.env, MINTBOT_VOICE_TOKEN: token },
    stdio: ["pipe", "pipe", "pipe"],
    windowsHide: true,
  });
  const exited = new Promise<number | null>((resolve) => {
    child.once("exit", (code) => resolve(code));
    child.once("error", () => resolve(null));
  });
  forwardLines(child.stderr, o.onStderr ?? (() => {}));

  const timeoutMs = o.readyTimeoutMs ?? 60000;
  const ready = await new Promise<ReadyLine>((resolve, reject) => {
    const rl = readline.createInterface({ input: child.stdout! });
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`voice agent not ready after ${timeoutMs} ms`));
    }, timeoutMs);
    rl.once("line", (line) => {
      clearTimeout(timer);
      try {
        const msg = JSON.parse(line);
        if (msg.type === "ready") return resolve(msg);
      } catch {}
      child.kill();
      reject(new Error(`unexpected first line from the voice agent: ${line}`));
    });
    child.once("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(Object.assign(new Error(`voice agent exited before it was ready (code ${code})`), { exitCode: code }));
    });
  });
  child.stdout!.resume(); // nothing else is written to stdout; keep the pipe drained

  const baseUrl = `http://127.0.0.1:${ready.port}`;
  const headers = { Authorization: `Bearer ${token}` };

  const call = async <T>(method: "GET" | "POST" | "DELETE", path: string): Promise<T> => {
    const r = await fetch(baseUrl + path, { method, headers });
    const text = await r.text();
    let body: any = text;
    try {
      body = JSON.parse(text);
    } catch {}
    if (!r.ok) throw new AgentHttpError(method, path, r.status, body);
    return body as T;
  };

  const openEvents = async (signal?: AbortSignal) => {
    const r = await fetch(baseUrl + "/v1/events", { headers, signal });
    if (!r.ok || !r.body) throw new AgentHttpError("GET", "/v1/events", r.status, await r.text());
    return readSse(r.body);
  };

  async function* events(signal?: AbortSignal): AsyncGenerator<AgentEvent> {
    yield* await openEvents(signal);
  }

  const shutdown = async (timeout = 3000) => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    try {
      await fetch(baseUrl + "/v1/shutdown", { method: "POST", headers, signal: AbortSignal.timeout(timeout) });
    } catch {}
    const killTimer = setTimeout(() => child.kill(), timeout);
    await exited;
    clearTimeout(killTimer);
  };

  return { child, ready, baseUrl, exited, call, openEvents, events, shutdown };
}
