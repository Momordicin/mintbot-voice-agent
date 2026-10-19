// Installing a mintbot-voice-agent release (tools/package_release.py) into an
// app-owned directory, Windows x64.
//
// Layout under root (programs and models are versioned separately, so an
// update of the program does not download the models again):
//   agent/<version>/mintbot-voice-agent.exe ...
//   models/<models version>/*.gguf ...
//   current.json        what installVoiceAgent() last completed
// Archives are verified (SHA-256) before extraction and every extracted file
// after it; a version directory only appears, by rename, once complete, so an
// interrupted or cancelled install never leaves a broken version behind, and
// the next install deletes what an interrupted one left in its temporary
// directories. Install or repair while no agent from that version runs
// (Windows keeps its files locked). Keep the voiceprint outside root: it is
// user data, not part of an install.
import { execFile } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { pipeline } from "node:stream/promises";
import { promisify } from "node:util";

import { extractZip } from "./unzip.ts";

const run = promisify(execFile);

export interface PackageInfo {
  version: string;
  file: string;
  // Where to download it; default: `file` next to the manifest (a GitHub release).
  // Lets the models be hosted elsewhere, e.g. a mirror, without a client change.
  url?: string;
  size: number;
  sha256: string;
  files: Record<string, string>; // archive path -> sha256
}

export interface Manifest {
  schema: number;
  name: string;
  version: string;
  api_version: number;
  platform: string;
  entry: string;
  models: { vad: string; asr: string; speaker: string };
  packages: { agent: PackageInfo; models: PackageInfo };
}

export interface Installation {
  version: string;
  apiVersion: number;
  exe: string;
  vad: string;
  asr: string;
  speaker: string;
}

// Progress of one package. A package that needs installing goes
//   download (only when fetched from a URL: bytes = 0 first, then at most every 200 ms while
//             bytes arrive, and bytes = total as soon as it ends) -> verify -> extract -> done,
// an intact one reports a single skip. models comes before agent.
export interface InstallProgress {
  package: "agent" | "models";
  phase: "skip" | "download" | "verify" | "extract" | "done";
  bytes?: number; // download only: bytes received so far
  total?: number; // download only: the size in the manifest
}

export interface InstallOptions {
  manifest: string; // local path or http(s) URL; package files are resolved relative to it
  root: string; // install directory
  apiVersion: number; // the API the app speaks; a manifest with another one is refused
  onProgress?: (msg: string) => void; // human-readable log lines
  onStep?: (p: InstallProgress) => void; // structured progress, e.g. for a progress bar
  // Cancels the install at any point; the error then has name "AbortError" and code
  // "ABORTED". Installed versions and current.json are left as they were.
  signal?: AbortSignal;
}

export type InstallErrorCode =
  | "NETWORK_ERROR" // the request failed or the connection broke
  | "HTTP_STATUS" // the server answered with another status than 200 (details.status)
  | "SHA256_MISMATCH" // an archive or an extracted file is not what the manifest says (details.file)
  | "API_VERSION_MISMATCH" // details.expected (the app's) / details.actual (the release's)
  | "ENOSPC_PRECHECK" // not enough free space (details.required / details.available, bytes)
  | "ABORTED" // options.signal was aborted
  | "INVALID_MANIFEST" // not a mintbot-voice-agent manifest
  | "EXTRACT_FAILED" // the archive could not be unpacked
  | "VERSION_MISMATCH" // the installed executable reports another version than the manifest
  | "UNSUPPORTED_PLATFORM"; // not Windows

export class VoiceInstallError extends Error {
  readonly code: InstallErrorCode;
  readonly details: Record<string, unknown>;
  constructor(code: InstallErrorCode, message: string, details: Record<string, unknown> = {}, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = code === "ABORTED" ? "AbortError" : "VoiceInstallError";
    this.code = code;
    this.details = details;
  }
}

const isUrl = (s: string) => /^https?:\/\//i.test(s);
const TMP_DIR = /^\..+\.tmp-[0-9a-f]+$/; // .<version>.tmp-<random>, see installPackage
const PROGRESS_MS = 200; // download callbacks: at most 5 per second, whatever the link speed

function aborted(o: InstallOptions): VoiceInstallError {
  return new VoiceInstallError("ABORTED", "voice agent install cancelled", {}, o.signal?.reason);
}

function checkAbort(o: InstallOptions) {
  if (o.signal?.aborted) throw aborted(o);
}

// Any failure while the signal is aborted is the cancellation, whatever it looks like.
function asInstallError(o: InstallOptions, e: unknown, fallback: () => VoiceInstallError): VoiceInstallError {
  if (o.signal?.aborted) return aborted(o);
  return e instanceof VoiceInstallError ? e : fallback();
}

async function sha256File(file: string, o: InstallOptions): Promise<string> {
  const h = createHash("sha256");
  await pipeline(createReadStream(file), h, { signal: o.signal });
  return h.digest("hex");
}

async function fetchOk(url: string, o: InstallOptions): Promise<Response> {
  let r: Response;
  try {
    r = await fetch(url, { signal: o.signal });
  } catch (e) {
    throw asInstallError(o, e, () => new VoiceInstallError("NETWORK_ERROR", `GET ${url}: ${(e as Error).message}`, { url }, e));
  }
  if (r.status !== 200) {
    await r.body?.cancel().catch(() => {});
    throw new VoiceInstallError("HTTP_STATUS", `GET ${url}: HTTP ${r.status}`, { url, status: r.status });
  }
  return r;
}

async function readManifest(o: InstallOptions): Promise<Manifest> {
  let text: string;
  if (isUrl(o.manifest)) {
    const r = await fetchOk(o.manifest, o);
    try {
      text = await r.text();
    } catch (e) {
      throw asInstallError(o, e, () => new VoiceInstallError("NETWORK_ERROR", `GET ${o.manifest}: ${(e as Error).message}`, { url: o.manifest }, e));
    }
  } else {
    text = await fs.readFile(o.manifest, { encoding: "utf8", signal: o.signal }).catch((e) => {
      throw asInstallError(o, e, () => new VoiceInstallError("INVALID_MANIFEST", `cannot read ${o.manifest}: ${e.message}`, { path: o.manifest }, e));
    });
  }
  let m: Manifest;
  try {
    m = JSON.parse(text);
  } catch (e) {
    throw new VoiceInstallError("INVALID_MANIFEST", `${o.manifest} is not JSON`, { path: o.manifest }, e);
  }
  if (m?.schema !== 1 || m.name !== "mintbot-voice-agent")
    throw new VoiceInstallError("INVALID_MANIFEST", `not a mintbot-voice-agent manifest: ${o.manifest}`, { path: o.manifest });
  return m;
}

// Every file of the package present with the right hash.
async function verifyDir(dir: string, pkg: PackageInfo, o: InstallOptions): Promise<boolean> {
  for (const [name, hash] of Object.entries(pkg.files)) {
    try {
      if ((await sha256File(path.join(dir, ...name.split("/")), o)) !== hash) return false;
    } catch (e) {
      checkAbort(o);
      return false;
    }
  }
  return true;
}

// The first extracted file that differs from the manifest, or null.
async function firstMismatch(dir: string, pkg: PackageInfo, o: InstallOptions): Promise<string | null> {
  for (const [name, hash] of Object.entries(pkg.files)) {
    let got = "";
    try {
      got = await sha256File(path.join(dir, ...name.split("/")), o);
    } catch {
      checkAbort(o);
    }
    if (got !== hash) return name;
  }
  return null;
}

// Removes what an install killed half-way left behind: agent/.<v>.tmp-*, models/.<n>.tmp-*.
async function removeStaleTemp(root: string, say: (m: string) => void) {
  for (const kind of ["agent", "models"]) {
    const dir = path.join(root, kind);
    for (const name of await fs.readdir(dir).catch(() => [] as string[])) {
      if (!TMP_DIR.test(name)) continue;
      await fs.rm(path.join(dir, name), { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }).catch(() => {});
      say(`removed leftover ${path.join(kind, name)}`);
    }
  }
}

async function download(url: string, file: string, pkg: PackageInfo, kind: InstallProgress["package"], o: InstallOptions) {
  const r = await fetchOk(url, o);
  const step = o.onStep ?? (() => {});
  let bytes = 0;
  let lastBytes = 0;
  let lastTime = Date.now();
  step({ package: kind, phase: "download", bytes: 0, total: pkg.size });
  const out = createWriteStream(file);
  try {
    // bytes = 0 at the start, then at least 200 ms apart and only when new bytes arrived,
    // and once more as soon as the download ends
    await pipeline(
      r.body as unknown as AsyncIterable<Uint8Array>,
      async function* (chunks: AsyncIterable<Uint8Array>) {
        for await (const c of chunks) {
          bytes += c.length;
          const now = Date.now();
          if (now - lastTime >= PROGRESS_MS && bytes > lastBytes) {
            lastBytes = bytes;
            lastTime = now;
            step({ package: kind, phase: "download", bytes, total: pkg.size });
          }
          yield c;
        }
      },
      out,
      { signal: o.signal },
    );
  } catch (e) {
    throw asInstallError(o, e, () => new VoiceInstallError("NETWORK_ERROR", `GET ${url}: ${(e as Error).message}`, { url }, e));
  }
  if (bytes !== lastBytes) step({ package: kind, phase: "download", bytes, total: pkg.size });
}

async function installPackage(o: InstallOptions, kind: InstallProgress["package"], pkg: PackageInfo, dest: string) {
  const say = o.onProgress ?? (() => {});
  const step = o.onStep ?? (() => {});
  const parent = path.dirname(dest);
  await fs.mkdir(parent, { recursive: true });
  const tmp = path.join(parent, `.${path.basename(dest)}.tmp-${randomBytes(4).toString("hex")}`);
  await fs.mkdir(tmp);
  try {
    const remote = Boolean(pkg.url) || isUrl(o.manifest);
    let archive = path.join(path.dirname(o.manifest), pkg.file);
    if (remote) {
      // relative to the manifest's own URL: works for releases/download/<tag>/ and for
      // releases/latest/download/ (GitHub redirects both; fetch follows)
      const url = pkg.url ?? new URL(pkg.file, o.manifest).href;
      say(`${pkg.file}: downloading ${(pkg.size / 2 ** 20).toFixed(1)} MB`);
      archive = path.join(tmp, pkg.file);
      await download(url, archive, pkg, kind, o);
    }
    checkAbort(o);
    step({ package: kind, phase: "verify" });
    let hash: string;
    try {
      hash = await sha256File(archive, o);
    } catch (e) {
      throw asInstallError(o, e, () => new VoiceInstallError("SHA256_MISMATCH", `${pkg.file}: cannot read (${(e as Error).message})`, { file: pkg.file }, e));
    }
    if (hash !== pkg.sha256)
      throw new VoiceInstallError("SHA256_MISMATCH", `${pkg.file}: SHA-256 mismatch`, { file: pkg.file, expected: pkg.sha256, actual: hash });
    step({ package: kind, phase: "extract" });
    const out = path.join(tmp, "x");
    await fs.mkdir(out);
    try {
      // in-process, not tar.exe: that one fails for paths outside the ANSI code page
      await extractZip(archive, out, o.signal);
    } catch (e) {
      throw asInstallError(o, e, () => new VoiceInstallError("EXTRACT_FAILED", `${pkg.file}: cannot extract (${(e as Error).message})`, { file: pkg.file }, e));
    }
    const bad = await firstMismatch(out, pkg, o);
    if (bad) throw new VoiceInstallError("SHA256_MISMATCH", `${pkg.file}: extracted ${bad} does not match the manifest`, { file: bad });
    checkAbort(o); // from here on the version directory is replaced without interruption
    await fs.rm(dest, { recursive: true, force: true }); // a damaged earlier copy
    await fs.rename(out, dest);
    say(`${pkg.file}: installed to ${dest}`);
    step({ package: kind, phase: "done" });
  } finally {
    // a write stream cut off by a cancellation may still hold its file for a moment
    await fs.rm(tmp, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }).catch(() => {});
  }
}

function installation(root: string, m: Manifest): Installation {
  const agentDir = path.join(root, "agent", m.packages.agent.version);
  const modelDir = path.join(root, "models", m.packages.models.version);
  return {
    version: m.version,
    apiVersion: m.api_version,
    exe: path.join(agentDir, m.entry),
    vad: path.join(modelDir, m.models.vad),
    asr: path.join(modelDir, m.models.asr),
    speaker: path.join(modelDir, m.models.speaker),
  };
}

// Errors are VoiceInstallError (see InstallErrorCode); a cancellation has name "AbortError".
export async function installVoiceAgent(o: InstallOptions): Promise<Installation> {
  if (process.platform !== "win32")
    throw new VoiceInstallError("UNSUPPORTED_PLATFORM", "mintbot-voice-agent releases are Windows x64 only", { platform: process.platform });
  const say = o.onProgress ?? (() => {});
  const step = o.onStep ?? (() => {});
  checkAbort(o);
  await fs.mkdir(o.root, { recursive: true });
  await removeStaleTemp(o.root, say);
  const m = await readManifest(o);
  if (m.api_version !== o.apiVersion)
    throw new VoiceInstallError("API_VERSION_MISMATCH", `voice agent ${m.version} speaks API ${m.api_version}, this app needs ${o.apiVersion}`,
                                { expected: o.apiVersion, actual: m.api_version });
  const inst = installation(o.root, m);
  const pkgs = [
    { kind: "models" as const, pkg: m.packages.models, dest: path.dirname(inst.vad) },
    { kind: "agent" as const, pkg: m.packages.agent, dest: path.dirname(inst.exe) },
  ];

  // what needs installing, and room for it, before anything is downloaded
  const todo: typeof pkgs = [];
  for (const p of pkgs) {
    if (await verifyDir(p.dest, p.pkg, o)) {
      say(`${p.pkg.file}: already installed`);
      step({ package: p.kind, phase: "skip" });
    } else {
      todo.push(p);
    }
  }
  const required = todo.reduce((n, p) => n + p.pkg.size * (p.pkg.url || isUrl(o.manifest) ? 2 : 1), 0); // zip + files
  if (required > 0) {
    const s = await fs.statfs(o.root);
    const available = Number(s.bavail) * Number(s.bsize);
    if (available < required)
      throw new VoiceInstallError("ENOSPC_PRECHECK",
                                  `not enough free space in ${o.root}: ${(required / 2 ** 20).toFixed(0)} MB needed, ${(available / 2 ** 20).toFixed(0)} MB free`,
                                  { required, available, root: o.root });
  }
  for (const p of todo) await installPackage(o, p.kind, p.pkg, p.dest);

  checkAbort(o);
  let v: { version?: string; api_version?: number };
  try {
    v = JSON.parse((await run(inst.exe, ["--version"], { signal: o.signal, windowsHide: true })).stdout);
  } catch (e) {
    throw asInstallError(o, e, () => new VoiceInstallError("VERSION_MISMATCH", `cannot run ${inst.exe} --version: ${(e as Error).message}`, { exe: inst.exe }, e));
  }
  if (v.version !== m.version || v.api_version !== m.api_version)
    throw new VoiceInstallError("VERSION_MISMATCH", `installed executable reports ${JSON.stringify(v)}, manifest says ${m.version} / API ${m.api_version}`,
                                { expected: m.version, actual: v.version });

  // current.json keeps the manifest, so readInstallation() can re-verify without the release files
  checkAbort(o);
  const current = path.join(o.root, "current.json");
  await fs.writeFile(current + ".tmp", JSON.stringify({ manifest: m }, null, 1));
  await fs.rename(current + ".tmp", current);
  return inst;
}

export interface ReadInstallationOptions {
  apiVersion: number;
  verify?: boolean; // re-hash every installed file (about 1 s for the models); default false
}

// The installation installVoiceAgent() last completed under root, without network
// access: null when there is none, it speaks another API, or a file is missing
// (or, with verify, damaged).
export async function readInstallation(root: string, o: ReadInstallationOptions): Promise<Installation | null> {
  let m: Manifest;
  try {
    m = JSON.parse(await fs.readFile(path.join(root, "current.json"), "utf8")).manifest;
  } catch {
    return null;
  }
  if (!m || m.api_version !== o.apiVersion) return null;
  const inst = installation(root, m);
  const io: InstallOptions = { manifest: "", root, apiVersion: o.apiVersion };
  for (const [pkg, dir] of [[m.packages.agent, path.dirname(inst.exe)], [m.packages.models, path.dirname(inst.vad)]] as const) {
    if (o.verify) {
      if (!(await verifyDir(dir, pkg, io))) return null;
    } else {
      for (const name of Object.keys(pkg.files)) {
        try {
          await fs.access(path.join(dir, ...name.split("/")));
        } catch {
          return null;
        }
      }
    }
  }
  return inst;
}

// Version directories under root other than the ones current.json points at
// (old installs, safe to delete once no agent from them is running).
export async function staleVersions(root: string): Promise<string[]> {
  const m: Manifest = JSON.parse(await fs.readFile(path.join(root, "current.json"), "utf8")).manifest;
  const out: string[] = [];
  for (const [kind, keep] of [["agent", m.packages.agent.version], ["models", m.packages.models.version]] as const) {
    for (const d of await fs.readdir(path.join(root, kind)).catch(() => [] as string[]))
      if (d !== keep && !TMP_DIR.test(d)) out.push(path.join(root, kind, d));
  }
  return out;
}
