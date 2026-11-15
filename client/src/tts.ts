// The GPT-SoVITS speech-synthesis component: install the official Windows package,
// use a folder the user already has, report status, cancel, and update to the package
// the release manifest pins. The app only calls these commands and shows status().
//
//   const tts = createTtsManager({ root, manifest, onState: (s) => show(s) });
//   tts.install({ variant: "v2pro", root: ttsRoot }); // progress and outcome in status()
//   await tts.useExisting(dir);                       // rejects with TtsError
//   await tts.checkUpdate({ signal: AbortSignal.timeout(30_000) }); // reads the manifest: status().updateAvailable
//   tts.cancel();
//   await tts.dispose();                              // on app exit: cancels, ends child processes
//
// Package pins come from the manifest's `tts` section (tools/tts_packages.json). The
// install record is <root>/tts.json. A download goes to <install root>/.download/<file>.part
// (kept on cancel or failure, resumed with a Range request), is checked against the pin's
// SHA-256, unpacked with 7za into <install root>/.extract-tmp, test-run there
// (runtime/python.exe api_v2.py), marked, and only then renamed to
// <install root>/<file without .7z>. A final folder whose mark matches the pin is adopted
// without downloading again. A folder chosen with useExisting() is test-run but never
// written to.
import { createHash } from "node:crypto";
import fs from "node:fs";
import * as path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as WebReadableStream } from "node:stream/web";

import { readManifest, VoiceInstallError } from "./install.ts";
import { API_VERSION } from "./protocol.ts";
import { ProcessFailure, probeTtsApi, sevenZipExtract, sevenZipUnpackedSize } from "./tts-process.ts";

export type TtsVariant = "v2pro" | "v5";

// One GPT-SoVITS package in the manifest's `tts` section.
export interface TtsPackageInfo {
  url: string;
  file: string; // <name>.7z; the installed folder is <name>
  size: number;
  sha256: string;
  date: string;
}

export type TtsErrorCode =
  | "TTS_BUSY" // an install or a test run is already in progress
  | "TTS_DISK_SPACE" // details.requiredBytes / details.freeBytes
  | "TTS_TARGET_EXISTS" // the final folder exists without a matching install mark (details.dir)
  | "TTS_CHECKSUM" // the downloaded file is not the pinned one (details.expected / details.actual); the .part is deleted
  | "TTS_DOWNLOAD" // details.status, or details.message
  | "TTS_EXTRACT" // 7za failed (details.message, details.stderr) or the package lacks files (details.missing)
  | "TTS_PROBE" // the test run failed (details.message, details.stderr: last lines of its error output)
  | "TTS_EXISTING_INVALID" // the chosen folder lacks files (details.missing)
  | "TTS_FAILED" // anything else (details.message)
  | "TTS_MANIFEST" // the manifest cannot be read (details.code, details.message) or has no package for the variant
  | "TTS_DISPOSED"; // a command after dispose()

export class TtsError extends Error {
  readonly code: TtsErrorCode;
  readonly details: Record<string, unknown>;
  constructor(code: TtsErrorCode, details: Record<string, unknown> = {}) {
    super(code);
    this.name = "TtsError";
    this.code = code;
    this.details = details;
  }
}

export interface TtsInstallTarget {
  variant: TtsVariant;
  root: string;
}

export interface TtsInstallStatus {
  state: "idle" | "installing" | "failed";
  action?: "download" | "existing";
  target?: TtsInstallTarget; // download only
  phase?: "checking" | "download" | "verify" | "extract" | "probe";
  bytes?: number;
  total?: number;
  errorCode?: TtsErrorCode;
  errorDetails?: Record<string, unknown>;
}

// The last download that failed (not one that was cancelled), kept for a retry until a
// download starts, succeeds or is adopted.
export interface TtsLastDownload {
  target: TtsInstallTarget;
  errorCode: TtsErrorCode;
  errorDetails: Record<string, unknown>;
}

export interface TtsStatus {
  installed: boolean;
  source: "download" | "existing" | null;
  variant: TtsVariant | null; // null for an existing folder
  dir: string | null;
  // a downloaded package whose SHA-256 differs from the manifest's pin for its variant;
  // install() with the same variant updates it. False until checkUpdate() has succeeded.
  updateAvailable: boolean;
  install: TtsInstallStatus;
  lastDownload: TtsLastDownload | null;
}

export interface TtsManagerOptions {
  root: string; // the voice install root (see installVoiceAgent); holds tts.json
  manifest: string; // the release manifest, path or URL, as for installVoiceAgent
  port?: number; // loopback port of the test run; default 18980
  // every change of status(); during a download, verify or extract at most every 500 ms
  onState?: (s: TtsStatus) => void;
}

export interface TtsManager {
  status(): TtsStatus;
  // Starts installing and returns at once; throws TtsError TTS_BUSY while busy. The
  // promise resolves when the install ends; its outcome is in status(). root: where
  // the package folder goes; default: the folder of the installed download, else <root>/tts.
  install(o: { variant: TtsVariant; root?: string }): Promise<void>;
  // Checks runtime/python.exe and api_v2.py, test-runs the folder and records it: true,
  // or false when cancelled. Rejects with TtsError (TTS_BUSY, TTS_EXISTING_INVALID,
  // TTS_PROBE); the failure stays in status().install.
  useExisting(dir: string): Promise<boolean>;
  // Reads the manifest's pins again and recomputes status().updateAvailable (onState when
  // it changes). Rejects with TtsError TTS_MANIFEST when the manifest cannot be read or
  // signal aborts the read (details.abortReason: the reason's name, e.g. "TimeoutError" for
  // AbortSignal.timeout), TTS_DISPOSED when dispose() ends it; status is then unchanged.
  checkUpdate(o?: { signal?: AbortSignal }): Promise<void>;
  cancel(): void;
  // Cancels, waits until the child processes have ended; later commands throw TTS_DISPOSED.
  dispose(): Promise<void>;
}

interface TtsRecord {
  source: "download" | "existing";
  variant: TtsVariant | null;
  dir: string;
  sha256: string | null;
}

const RECORD_FILE = "tts.json";
const MARKER_FILE = ".mintbot-install.json";
const REQUIRED_FILES = [path.join("runtime", "python.exe"), "api_v2.py"];
const NOTIFY_INTERVAL_MS = 500;
const DEFAULT_PORT = 18980;

class InstallFailure extends Error {
  readonly code: TtsErrorCode;
  readonly details: Record<string, unknown>;
  constructor(code: TtsErrorCode, details: Record<string, unknown>) {
    super(code);
    this.code = code;
    this.details = details;
  }
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
const exists = (target: string) => fs.existsSync(target);

function missingFiles(dir: string): string[] {
  return REQUIRED_FILES.filter((file) => !exists(path.join(dir, file))).map((file) => file.split(path.sep).join("/"));
}

function readRecord(file: string): TtsRecord | null {
  try {
    const r = JSON.parse(fs.readFileSync(file, "utf8"));
    return r && (r.source === "download" || r.source === "existing") && typeof r.dir === "string" ? r : null;
  } catch {
    return null;
  }
}

function detectInstalled(r: TtsRecord | null): boolean {
  if (r === null) return false;
  if (r.source === "download") return exists(path.join(r.dir, MARKER_FILE));
  return missingFiles(r.dir).length === 0;
}

async function fileSize(target: string): Promise<number> {
  try {
    return (await fs.promises.stat(target)).size;
  } catch {
    return 0;
  }
}

async function freeBytes(dir: string): Promise<number> {
  const stats = await fs.promises.statfs(dir);
  return Number(stats.bavail) * Number(stats.bsize);
}

async function readMarker(dir: string): Promise<{ variant?: unknown; sha256?: unknown } | null> {
  try {
    return JSON.parse(await fs.promises.readFile(path.join(dir, MARKER_FILE), "utf-8"));
  } catch {
    return null;
  }
}

// The package folder inside .extract-tmp: itself, or its single subfolder.
async function extractedContentDir(extractDir: string): Promise<string> {
  if (missingFiles(extractDir).length === 0) return extractDir;
  const entries = await fs.promises.readdir(extractDir, { withFileTypes: true });
  if (entries.length === 1 && entries[0].isDirectory()) {
    const inner = path.join(extractDir, entries[0].name);
    const missing = missingFiles(inner);
    if (missing.length === 0) return inner;
    throw new InstallFailure("TTS_EXTRACT", { missing });
  }
  throw new InstallFailure("TTS_EXTRACT", { missing: missingFiles(extractDir) });
}

export function createTtsManager(o: TtsManagerOptions): TtsManager {
  const recordPath = path.join(o.root, RECORD_FILE);
  const port = o.port ?? DEFAULT_PORT;
  let record = readRecord(recordPath);
  let installed = detectInstalled(record);
  let pins: Partial<Record<TtsVariant, TtsPackageInfo>> | null = null;
  let install: TtsInstallStatus = { state: "idle" };
  let activeAction: Pick<TtsInstallStatus, "action" | "target"> = {};
  let lastDownload: TtsLastDownload | null = null;
  let abort: AbortController | null = null;
  let task: Promise<unknown> = Promise.resolve();
  let disposed = false;
  const lifetime = new AbortController(); // ends a checkUpdate() in progress on dispose()
  let checking: Promise<unknown> = Promise.resolve();
  let lastNotifyAt = 0;

  function status(): TtsStatus {
    const pin = record?.variant ? pins?.[record.variant] : undefined;
    return {
      installed,
      source: record?.source ?? null,
      variant: record?.variant ?? null,
      dir: record?.dir ?? null,
      updateAvailable: installed && record?.source === "download" && pin !== undefined && pin.sha256 !== record.sha256,
      install: { ...install },
      lastDownload,
    };
  }

  function notify(): void {
    lastNotifyAt = Date.now();
    o.onState?.(status());
  }

  function setProgress(phase: NonNullable<TtsInstallStatus["phase"]>, bytes?: number, total?: number): void {
    const changed = install.phase !== phase;
    install = { state: "installing", ...activeAction, phase };
    if (bytes !== undefined) install.bytes = bytes;
    if (total !== undefined) install.total = total;
    if (changed || Date.now() - lastNotifyAt >= NOTIFY_INTERVAL_MS) notify();
  }

  async function loadPins(signal: AbortSignal): Promise<Partial<Record<TtsVariant, TtsPackageInfo>>> {
    const m = await readManifest({ manifest: o.manifest, root: o.root, apiVersion: API_VERSION, signal });
    return m.tts ?? {};
  }

  // Windows paths are case-insensitive: the same folder may be spelled differently.
  async function sameFolder(a: string, b: string): Promise<boolean> {
    try {
      return (await fs.promises.realpath(a)) === (await fs.promises.realpath(b));
    } catch {
      return path.relative(path.resolve(a).toLowerCase(), path.resolve(b).toLowerCase()) === "";
    }
  }

  async function saveRecord(next: TtsRecord): Promise<void> {
    await fs.promises.mkdir(o.root, { recursive: true });
    await fs.promises.writeFile(recordPath + ".tmp", JSON.stringify(next, null, 1));
    await fs.promises.rename(recordPath + ".tmp", recordPath);
    record = next;
    installed = true;
  }

  async function download(pkg: TtsPackageInfo, partPath: string, signal: AbortSignal): Promise<void> {
    let start = await fileSize(partPath);
    if (start > pkg.size) {
      await fs.promises.rm(partPath, { force: true });
      start = 0;
    }
    setProgress("download", start, pkg.size);
    if (start === pkg.size) return;
    let response: Response;
    try {
      response = await fetch(pkg.url, { headers: start > 0 ? { Range: `bytes=${start}-` } : {}, signal });
    } catch (err) {
      if (signal.aborted) throw err;
      throw new InstallFailure("TTS_DOWNLOAD", { message: message(err) });
    }
    if (response.status === 206) {
      const rangeStart = /^bytes (\d+)-/.exec(response.headers.get("content-range") ?? "")?.[1];
      if (Number(rangeStart) !== start) {
        await response.body?.cancel();
        throw new InstallFailure("TTS_DOWNLOAD", { message: "unexpected Content-Range", contentRange: response.headers.get("content-range") });
      }
    } else if (response.status === 200) {
      start = 0;
    } else {
      await response.body?.cancel();
      throw new InstallFailure("TTS_DOWNLOAD", { status: response.status });
    }
    if (!response.body) throw new InstallFailure("TTS_DOWNLOAD", { message: "empty response body" });
    let bytes = start;
    const counter = new Transform({
      transform(chunk: Buffer, _encoding, callback) {
        bytes += chunk.length;
        setProgress("download", bytes, pkg.size);
        callback(null, chunk);
      },
    });
    try {
      await pipeline(Readable.fromWeb(response.body as unknown as WebReadableStream), counter,
                     fs.createWriteStream(partPath, { flags: start > 0 ? "a" : "w" }), { signal });
    } catch (err) {
      if (signal.aborted) throw err;
      throw new InstallFailure("TTS_DOWNLOAD", { message: message(err) });
    }
    if (bytes !== pkg.size) throw new InstallFailure("TTS_DOWNLOAD", { message: "incomplete download", bytes, total: pkg.size });
  }

  async function sha256File(target: string, total: number, signal: AbortSignal): Promise<string> {
    const hash = createHash("sha256");
    let bytes = 0;
    setProgress("verify", 0, total);
    for await (const chunk of fs.createReadStream(target, { signal })) {
      hash.update(chunk as Buffer);
      bytes += (chunk as Buffer).length;
      setProgress("verify", bytes, total);
    }
    return hash.digest("hex");
  }

  // Records the package; an update removes the folder of the replaced package of the same variant.
  async function finishInstall(variant: TtsVariant, pkg: TtsPackageInfo, finalDir: string, archivePath: string, downloadDir: string) {
    const previous = record;
    await saveRecord({ source: "download", variant, dir: finalDir, sha256: pkg.sha256 });
    lastDownload = null;
    if (previous?.source === "download" && previous.variant === variant && !(await sameFolder(previous.dir, finalDir)))
      await fs.promises.rm(previous.dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }).catch(() => {});
    await fs.promises.rm(archivePath, { force: true }).catch(() => {});
    await fs.promises.rmdir(downloadDir).catch(() => {});
  }

  async function runInstall(variant: TtsVariant, root: string, signal: AbortSignal): Promise<void> {
    let outcome: TtsInstallStatus = { state: "idle" };
    try {
      setProgress("checking");
      let pkg: TtsPackageInfo | undefined;
      try {
        const read = await loadPins(signal);
        if (pins !== null) pins = read; // updateAvailable stays false until checkUpdate() has succeeded
        pkg = read[variant];
      } catch (err) {
        if (signal.aborted) throw err;
        throw new InstallFailure("TTS_MANIFEST", { code: err instanceof VoiceInstallError ? err.code : undefined, message: message(err) });
      }
      if (!pkg) throw new InstallFailure("TTS_MANIFEST", { message: `the manifest has no tts.${variant} package` });
      const downloadDir = path.join(root, ".download");
      const archivePath = path.join(downloadDir, pkg.file);
      const partPath = `${archivePath}.part`;
      const extractDir = path.join(root, ".extract-tmp");
      const finalDir = path.join(root, pkg.file.replace(/\.7z$/, ""));

      await fs.promises.mkdir(root, { recursive: true });
      await fs.promises.rm(extractDir, { recursive: true, force: true });
      if (exists(finalDir)) {
        const marker = await readMarker(finalDir);
        if (marker?.variant !== variant || marker?.sha256 !== pkg.sha256) throw new InstallFailure("TTS_TARGET_EXISTS", { dir: finalDir });
        await finishInstall(variant, pkg, finalDir, archivePath, downloadDir);
        return;
      }
      const archiveReady = exists(archivePath);
      const downloaded = archiveReady ? await fileSize(archivePath) : await fileSize(partPath);
      const requiredBytes = pkg.size - downloaded + pkg.size * 2;
      const free = await freeBytes(root);
      if (free < requiredBytes) throw new InstallFailure("TTS_DISK_SPACE", { requiredBytes, freeBytes: free });

      if (!archiveReady) {
        await fs.promises.mkdir(downloadDir, { recursive: true });
        await download(pkg, partPath, signal);
        const actual = await sha256File(partPath, pkg.size, signal);
        if (actual !== pkg.sha256) {
          await fs.promises.rm(partPath, { force: true });
          throw new InstallFailure("TTS_CHECKSUM", { expected: pkg.sha256, actual });
        }
        await fs.promises.rename(partPath, archivePath);
      }

      setProgress("extract", 0);
      let unpacked: number;
      try {
        unpacked = await sevenZipUnpackedSize(archivePath, signal);
      } catch (err) {
        if (signal.aborted) throw err;
        throw new InstallFailure("TTS_EXTRACT", { message: message(err) });
      }
      const freeForExtract = await freeBytes(root);
      if (freeForExtract < unpacked) throw new InstallFailure("TTS_DISK_SPACE", { requiredBytes: unpacked, freeBytes: freeForExtract });
      try {
        setProgress("extract", 0, unpacked);
        try {
          await sevenZipExtract(archivePath, extractDir, signal, (percent) => setProgress("extract", Math.round((unpacked * percent) / 100), unpacked));
        } catch (err) {
          if (signal.aborted) throw err;
          throw new InstallFailure("TTS_EXTRACT", { message: message(err), stderr: err instanceof ProcessFailure ? err.stderr : [] });
        }
        const contentDir = await extractedContentDir(extractDir);

        setProgress("probe");
        try {
          await probeTtsApi(contentDir, port, signal);
        } catch (err) {
          if (signal.aborted) throw err;
          throw new InstallFailure("TTS_PROBE", { message: message(err), stderr: err instanceof ProcessFailure ? err.stderr : [] });
        }
        signal.throwIfAborted();

        await fs.promises.writeFile(path.join(contentDir, MARKER_FILE),
                                    JSON.stringify({ variant, sha256: pkg.sha256, installedAt: new Date().toISOString() }, null, 2));
        await fs.promises.rename(contentDir, finalDir);
      } finally {
        await fs.promises.rm(extractDir, { recursive: true, force: true });
      }
      await finishInstall(variant, pkg, finalDir, archivePath, downloadDir);
    } catch (err) {
      if (!signal.aborted) {
        const failure = err instanceof InstallFailure
          ? { errorCode: err.code, errorDetails: err.details }
          : { errorCode: "TTS_FAILED" as const, errorDetails: { message: message(err) } };
        outcome = { state: "failed", ...activeAction, ...failure };
        lastDownload = { target: { variant, root }, ...failure };
      }
    } finally {
      abort = null;
      install = outcome;
      notify();
    }
  }

  function startInstall({ variant, root }: { variant: TtsVariant; root?: string }): Promise<void> {
    if (disposed) throw new TtsError("TTS_DISPOSED");
    if (install.state === "installing") throw new TtsError("TTS_BUSY");
    const resolved = path.resolve(root ?? (record?.source === "download" ? path.dirname(record.dir) : path.join(o.root, "tts")));
    const controller = new AbortController();
    abort = controller;
    activeAction = { action: "download", target: { variant, root: resolved } };
    lastDownload = null;
    install = { state: "installing", ...activeAction, phase: "checking" };
    notify();
    const run = runInstall(variant, resolved, controller.signal).catch(() => {});
    task = run;
    return run;
  }

  async function useExisting(dir: string): Promise<boolean> {
    const resolved = path.resolve(dir);
    if (disposed) throw new TtsError("TTS_DISPOSED");
    if (install.state === "installing") throw new TtsError("TTS_BUSY");
    activeAction = { action: "existing" };
    const missing = missingFiles(resolved);
    if (missing.length > 0) {
      install = { state: "failed", action: "existing", errorCode: "TTS_EXISTING_INVALID", errorDetails: { missing } };
      notify();
      throw new TtsError("TTS_EXISTING_INVALID", { missing });
    }
    const controller = new AbortController();
    abort = controller;
    const run = (async () => {
      setProgress("probe");
      let outcome: TtsInstallStatus = { state: "idle" };
      try {
        try {
          await probeTtsApi(resolved, port, controller.signal);
        } catch (err) {
          if (controller.signal.aborted) return false;
          const details = { message: message(err), stderr: err instanceof ProcessFailure ? err.stderr : [] };
          outcome = { state: "failed", action: "existing", errorCode: "TTS_PROBE", errorDetails: details };
          throw new TtsError("TTS_PROBE", details);
        }
        if (controller.signal.aborted) return false;
        try {
          await saveRecord({ source: "existing", variant: null, dir: resolved, sha256: null });
        } catch (err) {
          const details = { message: message(err) };
          outcome = { state: "failed", action: "existing", errorCode: "TTS_FAILED", errorDetails: details };
          throw new TtsError("TTS_FAILED", details);
        }
        return true;
      } finally {
        abort = null;
        install = outcome;
        notify();
      }
    })();
    task = run.catch(() => {});
    return run;
  }

  async function checkUpdate(c: { signal?: AbortSignal } = {}): Promise<void> {
    if (disposed) throw new TtsError("TTS_DISPOSED");
    const run = loadPins(c.signal ? AbortSignal.any([lifetime.signal, c.signal]) : lifetime.signal);
    checking = run.catch(() => {});
    let read: Partial<Record<TtsVariant, TtsPackageInfo>>;
    try {
      read = await run;
    } catch (err) {
      if (lifetime.signal.aborted) throw new TtsError("TTS_DISPOSED");
      const details: Record<string, unknown> = { code: err instanceof VoiceInstallError ? err.code : undefined, message: message(err) };
      if (c.signal?.aborted) {
        const reason: unknown = c.signal.reason;
        details.abortReason = reason instanceof Error ? reason.name : String(reason);
      }
      throw new TtsError("TTS_MANIFEST", details);
    }
    const before = status().updateAvailable;
    pins = read;
    if (status().updateAvailable !== before) notify();
  }

  function cancel(): void {
    abort?.abort();
  }

  return {
    status,
    install: startInstall,
    useExisting,
    checkUpdate,
    cancel,
    async dispose() {
      disposed = true;
      lifetime.abort();
      cancel();
      await Promise.all([checking, task]);
    },
  };
}
