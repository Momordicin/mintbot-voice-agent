// createTtsManager: free space, resumed download, checksum, cancellation, adoption of an
// interrupted install, leftover .extract-tmp, missing manifest section, update check and
// update, commands after dispose, use of an existing folder, status on start. fetch, statfs, 7za and the test run are mocked;
// everything is written under os.tmpdir().
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, it, mock } from "node:test";

class ProcessFailure extends Error {
  readonly stderr: string[];
  constructor(message: string, stderr: string[]) {
    super(message);
    this.stderr = stderr;
  }
}
const sevenZipUnpackedSize = mock.fn(async (_archive: string, _signal: AbortSignal) => 1000);
const sevenZipExtract = mock.fn(async (_archive: string, _dest: string, _signal: AbortSignal, _onPercent: (p: number) => void) => {});
const probeTtsApi = mock.fn(async (_dir: string, _port: number, _signal: AbortSignal) => {});
mock.module("../src/tts-process.ts", { namedExports: { ProcessFailure, sevenZipUnpackedSize, sevenZipExtract, probeTtsApi } });
const { createTtsManager, TtsError } = await import("../src/index.ts");
type TtsManager = ReturnType<typeof createTtsManager>;

const CONTENT = Buffer.from("fake GPT-SoVITS archive bytes ".repeat(40));
const pin = (name: string, content: Buffer) => ({
  url: `https://example.invalid/${name}.7z`,
  file: `${name}.7z`,
  size: content.length,
  sha256: createHash("sha256").update(content).digest("hex"),
  date: "2026-01-01",
});
const FAKE = pin("GPT-SoVITS-test-1", CONTENT);

let tmp: string;
let voiceRoot: string;
let ttsRoot: string;
let manifest: string;
let managers: TtsManager[];
let fetchImpl: (url: string, init: RequestInit) => Promise<Response>;
let fetchCalls: string[];
let freeSpace: () => number;

const downloadDir = () => path.join(ttsRoot, ".download");
const partPath = () => path.join(downloadDir(), `${FAKE.file}.part`);
const archivePath = () => path.join(downloadDir(), FAKE.file);
const extractDir = () => path.join(ttsRoot, ".extract-tmp");
const finalDir = (name = "GPT-SoVITS-test-1") => path.join(ttsRoot, name);

function writeGptSovits(dir: string): void {
  fs.mkdirSync(path.join(dir, "runtime"), { recursive: true });
  fs.writeFileSync(path.join(dir, "runtime", "python.exe"), "");
  fs.writeFileSync(path.join(dir, "api_v2.py"), "");
}

function writeManifest(tts?: object): void {
  fs.writeFileSync(manifest, JSON.stringify({ schema: 1, name: "mintbot-voice-agent", ...(tts ? { tts } : {}) }));
}

function manager(onState?: (s: ReturnType<TtsManager["status"]>) => void): TtsManager {
  const m = createTtsManager({ root: voiceRoot, manifest, port: 18999, onState });
  managers.push(m);
  return m;
}

async function waitFor(cond: () => boolean): Promise<void> {
  for (const end = Date.now() + 5000; !cond(); await new Promise((r) => setTimeout(r, 10)))
    if (Date.now() > end) assert.fail("timed out");
}

async function install(m = manager(), variant: "v2pro" | "v5" = "v2pro") {
  await m.install({ variant, root: ttsRoot });
  return m.status();
}

const stalledBody = (bytes: Buffer) =>
  new ReadableStream({
    start(c) {
      c.enqueue(new Uint8Array(bytes));
    },
    pull: () => new Promise(() => {}),
  });

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "voice-tts-"));
  voiceRoot = path.join(tmp, "voice");
  ttsRoot = path.join(tmp, "tts");
  manifest = path.join(tmp, "manifest.json");
  writeManifest({ v2pro: FAKE, v5: pin("GPT-SoVITS-test-5", CONTENT) });
  managers = [];
  fetchCalls = [];
  fetchImpl = async () => new Response(CONTENT, { status: 200 });
  mock.method(globalThis, "fetch", (url: string, init: RequestInit) => {
    fetchCalls.push(url);
    return fetchImpl(url, init);
  });
  freeSpace = () => Number.MAX_SAFE_INTEGER;
  mock.method(fs.promises, "statfs", async () => ({ bavail: freeSpace(), bsize: 1 }));
  sevenZipUnpackedSize.mock.resetCalls();
  sevenZipExtract.mock.resetCalls();
  probeTtsApi.mock.resetCalls();
  sevenZipExtract.mock.mockImplementation(async (_a, dest) => writeGptSovits(path.join(dest, path.basename(_a, ".7z"))));
  probeTtsApi.mock.mockImplementation(async () => {});
});

afterEach(async () => {
  await Promise.all(managers.map((m) => m.dispose()));
  mock.restoreAll();
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe("tts install", () => {
  for (const variant of ["v2pro", "v5"] as const) {
    it(`${variant} checks space against its own size and downloads its own URL`, async () => {
      const size = 8_000_000_000 + (variant === "v5" ? 1 : 0);
      const p = { ...pin(`GPT-SoVITS-${variant}`, CONTENT), size };
      writeManifest({ [variant]: p });
      freeSpace = () => size * 3 - 1;
      let s = await install(manager(), variant);
      assert.deepEqual(s.install, { state: "failed", action: "download", target: { variant, root: ttsRoot }, errorCode: "TTS_DISK_SPACE",
                                    errorDetails: { requiredBytes: size * 3, freeBytes: size * 3 - 1 } });
      assert.deepEqual(fetchCalls, []);
      assert.equal(fs.existsSync(downloadDir()), false);
      freeSpace = () => size * 3;
      fetchImpl = async () => new Response(null, { status: 404 });
      s = await install(manager(), variant);
      assert.deepEqual(fetchCalls, [p.url]);
      assert.equal(s.install.errorCode, "TTS_DOWNLOAD");
      assert.deepEqual(s.install.errorDetails, { status: 404 });
    });
  }

  it("counts bytes already in the .part file when checking free space", async () => {
    fs.mkdirSync(downloadDir(), { recursive: true });
    fs.writeFileSync(partPath(), CONTENT.subarray(0, 100));
    freeSpace = () => FAKE.size * 3 - 101;
    const s = await install();
    assert.equal(s.install.errorCode, "TTS_DISK_SPACE");
    assert.equal(s.install.errorDetails?.requiredBytes, FAKE.size * 3 - 100);
    assert.deepEqual(fetchCalls, []);
  });

  it("resumes from an existing .part with a Range request and finishes the install", async () => {
    fs.mkdirSync(downloadDir(), { recursive: true });
    fs.writeFileSync(partPath(), CONTENT.subarray(0, 300));
    fetchImpl = async (_url, init) => {
      assert.equal((init.headers as Record<string, string>).Range, "bytes=300-");
      return new Response(CONTENT.subarray(300), { status: 206, headers: { "content-range": `bytes 300-${FAKE.size - 1}/${FAKE.size}` } });
    };
    const s = await install();
    assert.deepEqual(s, { installed: true, source: "download", variant: "v2pro", dir: finalDir(), updateAvailable: false,
                          install: { state: "idle" }, lastDownload: null });
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(voiceRoot, "tts.json"), "utf8")),
                     { source: "download", variant: "v2pro", dir: finalDir(), sha256: FAKE.sha256 });
    assert.deepEqual(probeTtsApi.mock.calls[0].arguments.slice(0, 2), [path.join(extractDir(), "GPT-SoVITS-test-1"), 18999]);
    const marker = JSON.parse(fs.readFileSync(path.join(finalDir(), ".mintbot-install.json"), "utf8"));
    assert.deepEqual([marker.variant, marker.sha256], ["v2pro", FAKE.sha256]);
    for (const gone of [archivePath(), partPath(), extractDir()]) assert.equal(fs.existsSync(gone), false, gone);
  });

  it("deletes the .part and fails with TTS_CHECKSUM when the hash does not match", async () => {
    const corrupt = Buffer.from(CONTENT);
    corrupt[0] ^= 0xff;
    fetchImpl = async () => new Response(corrupt, { status: 200 });
    const s = await install();
    assert.equal(s.install.errorCode, "TTS_CHECKSUM");
    assert.equal(fs.existsSync(partPath()), false);
    assert.equal(sevenZipExtract.mock.callCount(), 0);
  });

  it("keeps the .part on cancel; lastDownload is set by a failure, cleared by a new start, not set by a cancel", async () => {
    const m = manager();
    fetchImpl = async () => new Response(null, { status: 503 });
    assert.equal((await install(m)).lastDownload?.errorCode, "TTS_DOWNLOAD");
    fetchImpl = async () => new Response(stalledBody(CONTENT.subarray(0, 200)), { status: 200 });
    const run = m.install({ variant: "v2pro", root: ttsRoot });
    assert.equal(m.status().install.state, "installing");
    assert.equal(m.status().lastDownload, null);
    assert.throws(() => m.install({ variant: "v2pro", root: ttsRoot }), (e) => e instanceof TtsError && e.code === "TTS_BUSY");
    await waitFor(() => m.status().install.bytes === 200);
    m.cancel();
    await run;
    assert.equal(m.status().installed, false);
    assert.deepEqual(m.status().install, { state: "idle" });
    assert.equal(m.status().lastDownload, null);
    assert.equal(fs.statSync(partPath()).size, 200);
  });

  it("removes .extract-tmp but keeps the archive when cancelled during extraction", async () => {
    fs.mkdirSync(downloadDir(), { recursive: true });
    fs.writeFileSync(archivePath(), CONTENT);
    sevenZipExtract.mock.mockImplementation((_a, dest, signal) => {
      writeGptSovits(path.join(dest, "GPT-SoVITS-test-1"));
      return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
    });
    const m = manager();
    const run = m.install({ variant: "v2pro", root: ttsRoot });
    await waitFor(() => sevenZipExtract.mock.callCount() === 1);
    m.cancel();
    await run;
    assert.deepEqual(m.status().install, { state: "idle" });
    assert.deepEqual(fetchCalls, []);
    assert.equal(fs.existsSync(extractDir()), false);
    assert.equal(fs.existsSync(finalDir()), false);
    assert.equal(fs.existsSync(archivePath()), true);
  });

  it("reports TTS_PROBE with stderr lines and removes the extracted folder when the test run fails", async () => {
    probeTtsApi.mock.mockImplementation(async () => {
      throw new ProcessFailure("api_v2.py exited with code 1", ["Traceback", "CUDA error"]);
    });
    const s = await install();
    assert.deepEqual(s.install, { state: "failed", action: "download", target: { variant: "v2pro", root: ttsRoot }, errorCode: "TTS_PROBE",
                                  errorDetails: { message: "api_v2.py exited with code 1", stderr: ["Traceback", "CUDA error"] } });
    assert.equal(fs.existsSync(finalDir()), false);
    assert.equal(fs.existsSync(extractDir()), false);
    assert.equal(fs.existsSync(archivePath()), true);
    assert.equal(fs.existsSync(path.join(voiceRoot, "tts.json")), false);
  });

  for (const [name, marker, adopted] of [
    ["matching marker is adopted", { variant: "v2pro", sha256: FAKE.sha256 }, true],
    ["marker of another variant is refused", { variant: "v5", sha256: FAKE.sha256 }, false],
    ["folder without marker is refused", null, false],
  ] as const) {
    it(`existing final folder: ${name}`, async () => {
      writeGptSovits(finalDir());
      if (marker) fs.writeFileSync(path.join(finalDir(), ".mintbot-install.json"), JSON.stringify(marker));
      fs.mkdirSync(downloadDir(), { recursive: true });
      fs.writeFileSync(archivePath(), CONTENT);
      const s = await install();
      if (adopted) {
        assert.deepEqual(s, { installed: true, source: "download", variant: "v2pro", dir: finalDir(), updateAvailable: false,
                              install: { state: "idle" }, lastDownload: null });
        assert.equal(fs.existsSync(archivePath()), false);
      } else {
        assert.equal(s.install.errorCode, "TTS_TARGET_EXISTS");
        assert.deepEqual(s.install.errorDetails, { dir: finalDir() });
        assert.equal(s.installed, false);
        assert.equal(fs.existsSync(archivePath()), true);
      }
      assert.deepEqual(fetchCalls, []);
      assert.equal(sevenZipExtract.mock.callCount(), 0);
    });
  }

  it("clears a leftover .extract-tmp before checking space and extracts again", async () => {
    fs.mkdirSync(downloadDir(), { recursive: true });
    fs.writeFileSync(archivePath(), CONTENT);
    writeGptSovits(path.join(extractDir(), "GPT-SoVITS-test-1"));
    fs.writeFileSync(path.join(extractDir(), "leftover.txt"), "");
    freeSpace = () => (fs.existsSync(extractDir()) ? 0 : Number.MAX_SAFE_INTEGER);
    sevenZipExtract.mock.mockImplementation(async (_a, dest) => {
      assert.equal(fs.existsSync(dest), false);
      writeGptSovits(path.join(dest, "GPT-SoVITS-test-1"));
    });
    const s = await install();
    assert.equal(s.installed, true);
    assert.equal(sevenZipExtract.mock.callCount(), 1);
    assert.equal(fs.existsSync(extractDir()), false);
    assert.equal(fs.existsSync(path.join(finalDir(), ".mintbot-install.json")), true);
  });

  it("fails with TTS_MANIFEST when the manifest has no tts section", async () => {
    writeManifest();
    const s = await install();
    assert.equal(s.install.errorCode, "TTS_MANIFEST");
    assert.equal(s.lastDownload?.errorCode, "TTS_MANIFEST");
    assert.deepEqual(fetchCalls, []);
    assert.equal(fs.existsSync(ttsRoot), false);
  });

  it("offers an update when the pin changes; a failed update keeps the old folder, a successful one replaces it", async () => {
    await install();
    const next = Buffer.from("next GPT-SoVITS package ".repeat(30));
    writeManifest({ v2pro: pin("GPT-SoVITS-test-2", next) });
    const m = manager();
    assert.equal(m.status().updateAvailable, false);
    await m.checkUpdate();
    assert.equal(m.status().updateAvailable, true);
    fetchImpl = async () => new Response(null, { status: 503 });
    await m.install({ variant: "v2pro" });
    assert.equal(m.status().install.errorCode, "TTS_DOWNLOAD");
    assert.deepEqual([m.status().dir, m.status().updateAvailable, fs.existsSync(finalDir())], [finalDir(), true, true]);
    fetchImpl = async () => new Response(next, { status: 200 });
    await m.install({ variant: "v2pro" });
    assert.deepEqual(m.status(), { installed: true, source: "download", variant: "v2pro", dir: finalDir("GPT-SoVITS-test-2"),
                                   updateAvailable: false, install: { state: "idle" }, lastDownload: null });
    assert.equal(fs.existsSync(finalDir()), false);
    assert.equal(fs.existsSync(path.join(finalDir("GPT-SoVITS-test-2"), ".mintbot-install.json")), true);
  });

  it("reads no manifest at construction; checkUpdate() reads it, sets updateAvailable and notifies", async () => {
    await install();
    fetchCalls = [];
    const next = pin("GPT-SoVITS-test-2", Buffer.from("next"));
    manifest = "https://example.invalid/manifest.json";
    fetchImpl = async () => new Response(JSON.stringify({ schema: 1, name: "mintbot-voice-agent", tts: { v2pro: next } }), { status: 200 });
    const seen: boolean[] = [];
    const m = manager((s) => seen.push(s.updateAvailable));
    await new Promise((r) => setTimeout(r, 50));
    assert.deepEqual(fetchCalls, []);
    assert.equal(m.status().updateAvailable, false);
    await m.checkUpdate();
    assert.deepEqual(fetchCalls, [manifest]);
    assert.deepEqual(seen, [true]);
    fetchImpl = async () => new Response(null, { status: 500 });
    await assert.rejects(m.checkUpdate(), (e) => e instanceof TtsError && e.code === "TTS_MANIFEST");
  });

  it("checkUpdate(): a caller timeout rejects TTS_MANIFEST, dispose() rejects TTS_DISPOSED; status unchanged", async () => {
    await install();
    fetchCalls = [];
    manifest = "https://example.invalid/manifest.json";
    fetchImpl = (_url, init) => new Promise((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
    const m = manager();
    const before = m.status();
    const keepAlive = setTimeout(() => {}, 5000); // AbortSignal.timeout's timer does not keep the process alive
    await assert.rejects(m.checkUpdate({ signal: AbortSignal.timeout(50) }),
                         (e) => e instanceof TtsError && e.code === "TTS_MANIFEST" && e.details.abortReason === "TimeoutError");
    clearTimeout(keepAlive);
    assert.deepEqual(m.status(), before);
    const pending = m.checkUpdate({ signal: new AbortController().signal });
    await waitFor(() => fetchCalls.length === 2);
    await m.dispose();
    await assert.rejects(pending, (e) => e instanceof TtsError && e.code === "TTS_DISPOSED");
    assert.deepEqual(m.status(), before);
  });

  it("adopting the same folder through a differently cased root keeps it and updates the record", async () => {
    await install();
    const upper = ttsRoot.toUpperCase();
    const m = manager();
    await m.install({ variant: "v2pro", root: upper });
    assert.equal(m.status().dir, path.join(upper, "GPT-SoVITS-test-1"));
    assert.equal(m.status().installed, true);
    assert.equal(fs.existsSync(path.join(finalDir(), ".mintbot-install.json")), true);
    assert.equal(JSON.parse(fs.readFileSync(path.join(voiceRoot, "tts.json"), "utf8")).dir, path.join(upper, "GPT-SoVITS-test-1"));
  });

  it("commands after dispose() throw TTS_DISPOSED and start nothing", async () => {
    const dir = path.join(tmp, "user");
    writeGptSovits(dir);
    const m = manager();
    await m.dispose();
    const disposed = (e: unknown) => e instanceof TtsError && e.code === "TTS_DISPOSED";
    assert.throws(() => m.install({ variant: "v2pro", root: ttsRoot }), disposed);
    await assert.rejects(m.useExisting(dir), disposed);
    await assert.rejects(m.checkUpdate(), disposed);
    assert.deepEqual(fetchCalls, []);
    assert.deepEqual([sevenZipExtract.mock.callCount(), sevenZipUnpackedSize.mock.callCount(), probeTtsApi.mock.callCount()], [0, 0, 0]);
    assert.deepEqual(m.status().install, { state: "idle" });
  });

  it("dispose() cancels a running install and waits for it", async () => {
    fetchImpl = async () => new Response(stalledBody(CONTENT.subarray(0, 50)), { status: 200 });
    const m = manager();
    m.install({ variant: "v2pro", root: ttsRoot });
    await waitFor(() => m.status().install.bytes === 50);
    await m.dispose();
    assert.deepEqual(m.status().install, { state: "idle" });
  });
});

describe("tts use existing", () => {
  it("keeps the failed download for retry after a use-existing attempt fails", async () => {
    const m = manager();
    fetchImpl = async () => new Response(null, { status: 503 });
    await install(m);
    const lastDownload = { target: { variant: "v2pro", root: ttsRoot }, errorCode: "TTS_DOWNLOAD", errorDetails: { status: 503 } };
    assert.deepEqual(m.status().lastDownload, lastDownload);
    await assert.rejects(m.useExisting(tmp), (e) => e instanceof TtsError && e.code === "TTS_EXISTING_INVALID");
    assert.deepEqual([m.status().install.state, m.status().install.action], ["failed", "existing"]);
    assert.deepEqual(m.status().lastDownload, lastDownload);
  });

  it("lists missing files without a test run", async () => {
    const dir = path.join(tmp, "user");
    fs.mkdirSync(dir);
    fs.writeFileSync(path.join(dir, "api_v2.py"), "");
    const m = manager();
    await assert.rejects(m.useExisting(dir), (e) => e instanceof TtsError && e.code === "TTS_EXISTING_INVALID"
                                                    && JSON.stringify(e.details) === '{"missing":["runtime/python.exe"]}');
    assert.deepEqual(m.status().install, { state: "failed", action: "existing", errorCode: "TTS_EXISTING_INVALID",
                                           errorDetails: { missing: ["runtime/python.exe"] } });
    assert.equal(probeTtsApi.mock.callCount(), 0);
    assert.equal(fs.existsSync(path.join(voiceRoot, "tts.json")), false);
  });

  for (const probeOk of [false, true]) {
    it(`test run ${probeOk ? "succeeds: recorded" : "fails: kept in status"}, the folder is not written to`, async () => {
      const dir = path.join(tmp, "user");
      writeGptSovits(dir);
      if (!probeOk) probeTtsApi.mock.mockImplementation(async () => {
        throw new ProcessFailure("timeout", ["line"]);
      });
      const m = manager();
      if (probeOk) {
        assert.equal(await m.useExisting(dir), true);
        assert.deepEqual(m.status(), { installed: true, source: "existing", variant: null, dir, updateAvailable: false,
                                       install: { state: "idle" }, lastDownload: null });
      } else {
        await assert.rejects(m.useExisting(dir), (e) => e instanceof TtsError && e.code === "TTS_PROBE");
        assert.equal(m.status().installed, false);
        assert.deepEqual(m.status().install, { state: "failed", action: "existing", errorCode: "TTS_PROBE",
                                               errorDetails: { message: "timeout", stderr: ["line"] } });
      }
      assert.deepEqual(fs.readdirSync(dir).sort(), ["api_v2.py", "runtime"]);
    });
  }
});

describe("tts status on start", () => {
  for (const [name, source, present, installed] of [
    ["download with marker", "download", true, true],
    ["download without marker", "download", false, false],
    ["existing with files", "existing", true, true],
    ["existing with files removed", "existing", false, false],
  ] as const) {
    it(name, () => {
      const dir = path.join(tmp, "GPT-SoVITS");
      if (present && source === "download") {
        fs.mkdirSync(dir);
        fs.writeFileSync(path.join(dir, ".mintbot-install.json"), "{}");
      } else if (present) {
        writeGptSovits(dir);
      }
      const variant = source === "download" ? "v5" : null;
      fs.mkdirSync(voiceRoot);
      fs.writeFileSync(path.join(voiceRoot, "tts.json"), JSON.stringify({ source, variant, dir, sha256: null }));
      assert.deepEqual(manager().status(), { installed, source, variant, dir, updateAvailable: false, install: { state: "idle" }, lastDownload: null });
    });
  }
});
