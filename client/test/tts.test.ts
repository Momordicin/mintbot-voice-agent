// createTtsManager: free space, resumed download, checksum, cancellation, adoption of an
// interrupted install, leftover .extract-tmp, missing manifest section, update check and
// update, commands after dispose, use of an existing folder, status on start, the service
// and its model, its config file, synthesis. fetch, statfs, 7za, the test run and the service are mocked;
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
let configs: string[] = [];
const readConfig = (config: string) => configs.push(fs.readFileSync(config, "utf8"));
const probe = async (_dir: string, config: string, _port: number, _signal: AbortSignal) => {
  readConfig(config);
};
const probeTtsApi = mock.fn(probe);
function fakeService() {
  let exit: (message: string) => void = () => {};
  const exited = new Promise<string>((resolve) => {
    exit = resolve;
  });
  return { url: "http://127.0.0.1:18999/", stderr: ["boot"], exited, exit, stop: mock.fn(async () => exit("api_v2.py exited with code 1")) };
}
let services: ReturnType<typeof fakeService>[] = [];
const startService = async (_dir: string, config: string, _port: number, _signal: AbortSignal) => {
  readConfig(config);
  const s = fakeService();
  services.push(s);
  return s;
};
const startTtsService = mock.fn(startService);
mock.module("../src/tts-process.ts", { namedExports: { ProcessFailure, sevenZipUnpackedSize, sevenZipExtract, probeTtsApi, startTtsService } });
const { createTtsManager, TtsError } = await import("../src/index.ts");
type TtsManager = ReturnType<typeof createTtsManager>;
const STOPPED = { state: "stopped", model: null, modelState: "none" };
const INFER_CONFIG = "custom:\n  device: cuda\n  is_half: true\n  version: v2\n";

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
const configPath = () => path.join(voiceRoot, "tts-infer.yaml");

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
  startTtsService.mock.resetCalls();
  services = [];
  configs = [];
  sevenZipExtract.mock.mockImplementation(async (_a, dest) => writeGptSovits(path.join(dest, path.basename(_a, ".7z"))));
  probeTtsApi.mock.mockImplementation(probe);
  startTtsService.mock.mockImplementation(startService);
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
                          install: { state: "idle" }, lastDownload: null, service: STOPPED });
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(voiceRoot, "tts.json"), "utf8")),
                     { source: "download", variant: "v2pro", dir: finalDir(), sha256: FAKE.sha256 });
    assert.deepEqual(probeTtsApi.mock.calls[0].arguments.slice(0, 3), [path.join(extractDir(), "GPT-SoVITS-test-1"), configPath(), 18999]);
    assert.deepEqual(configs, [INFER_CONFIG]);
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
                              install: { state: "idle" }, lastDownload: null, service: STOPPED });
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
                                   updateAvailable: false, install: { state: "idle" }, lastDownload: null, service: STOPPED });
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
    assert.throws(() => m.startService(null), disposed);
    await assert.rejects(m.useExisting(dir), disposed);
    await assert.rejects(m.checkUpdate(), disposed);
    assert.deepEqual(fetchCalls, []);
    assert.deepEqual([sevenZipExtract.mock.callCount(), sevenZipUnpackedSize.mock.callCount(), probeTtsApi.mock.callCount(),
                      startTtsService.mock.callCount()], [0, 0, 0, 0]);
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
                                       install: { state: "idle" }, lastDownload: null, service: STOPPED });
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
      assert.deepEqual(manager().status(), { installed, source, variant, dir, updateAvailable: false, install: { state: "idle" }, lastDownload: null,
                                             service: STOPPED });
    });
  }
});

describe("tts service", () => {
  const MODEL = { gptWeights: "C:\\voices\\a b.ckpt", sovitsWeights: "C:\\voices\\a b.pth" };
  const OTHER = { gptWeights: "C:\\voices\\other.ckpt", sovitsWeights: "C:\\voices\\other.pth" };
  const weightsUrl = (endpoint: string, file: string) => `http://127.0.0.1:18999/${endpoint}?weights_path=${encodeURIComponent(file)}`;
  const weightsUrls = (model: typeof MODEL) => [weightsUrl("set_gpt_weights", model.gptWeights), weightsUrl("set_sovits_weights", model.sovitsWeights)];
  const serviceDir = () => path.join(tmp, "GPT-SoVITS");
  const userDir = () => path.join(tmp, "user");
  const request = { text: "你好", textLang: "zh", refAudio: "C:\\voices\\ref.wav", refText: "参考", promptLang: "zh" };
  const notReady = (e: unknown) => e instanceof TtsError && e.code === "TTS_NOT_READY";

  function installedManager(onState?: (s: ReturnType<TtsManager["status"]>) => void): TtsManager {
    writeGptSovits(serviceDir());
    fs.mkdirSync(voiceRoot, { recursive: true });
    fs.writeFileSync(path.join(voiceRoot, "tts.json"), JSON.stringify({ source: "existing", variant: null, dir: serviceDir(), sha256: null }));
    return manager(onState);
  }

  for (const [name, setup, expected] of [
    ["starts and keeps the process", () => installedManager(), { state: "running", model: null, modelState: "none" }],
    ["port in use: TTS_SERVICE", () => {
      startTtsService.mock.mockImplementation(async () => {
        throw new ProcessFailure("port 18999 is already in use", []);
      });
      return installedManager();
    }, { state: "failed", model: null, modelState: "none", errorCode: "TTS_SERVICE", errorDetails: { message: "port 18999 is already in use", stderr: [] } }],
    ["not installed: TTS_NOT_INSTALLED", () => manager(), { state: "failed", model: null, modelState: "none", errorCode: "TTS_NOT_INSTALLED", errorDetails: {} }],
  ] as const) {
    it(`start: ${name}`, async () => {
      const m = setup();
      await m.startService(null);
      assert.deepEqual(m.status().service, expected);
      assert.deepEqual(fetchCalls, []);
      if (expected.state === "running") assert.deepEqual(startTtsService.mock.calls[0].arguments.slice(0, 3), [serviceDir(), configPath(), 18999]);
      if (name.startsWith("not installed")) assert.equal(startTtsService.mock.callCount(), 0);
    });
  }

  for (const [name, status, expected] of [
    ["both weights load", 200, { state: "running", model: MODEL, modelState: "loaded" }],
    ["a weight endpoint fails: TTS_MODEL", 400, { state: "running", model: MODEL, modelState: "failed", errorCode: "TTS_MODEL",
                                                 errorDetails: { status: 400, message: "change sovits weight failed", exception: "bad file" } }],
  ] as const) {
    it(`model: ${name}; null loads nothing`, async () => {
      fetchImpl = async (url) => url.includes("set_sovits_weights") && status !== 200
        ? new Response(JSON.stringify({ message: "change sovits weight failed", Exception: "bad file" }), { status })
        : new Response("success", { status: 200 });
      const m = installedManager();
      await m.startService(MODEL);
      assert.deepEqual(fetchCalls, weightsUrls(MODEL));
      assert.deepEqual(m.status().service, expected);
      await m.loadModel(null);
      assert.deepEqual(m.status().service, { state: "running", model: null, modelState: "none" });
      assert.equal(fetchCalls.length, 2);
    });
  }

  it("model: the last loadModel wins over an earlier one still in flight", async () => {
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fetchImpl = async (url) => {
      if (url === weightsUrl("set_gpt_weights", MODEL.gptWeights)) {
        await held;
        return new Response(JSON.stringify({ message: "change gpt weight failed" }), { status: 400 });
      }
      return new Response("success", { status: 200 });
    };
    const seen: string[] = [];
    const m = installedManager((s) => seen.push(s.service.modelState));
    await m.startService(null);
    const first = m.loadModel(MODEL);
    await waitFor(() => fetchCalls.length === 1);
    const second = m.loadModel(OTHER);
    release();
    await Promise.all([first, second]);
    assert.deepEqual(fetchCalls, [weightsUrl("set_gpt_weights", MODEL.gptWeights), ...weightsUrls(OTHER)]);
    assert.deepEqual(m.status().service, { state: "running", model: OTHER, modelState: "loaded" });
    assert.equal(seen.includes("failed"), false);
  });

  it("model: loadModel while running reports loading at once, so synthesize waits for it; null is none at once", async () => {
    const seen: ReturnType<TtsManager["status"]>["service"][] = [];
    const m = installedManager((s) => seen.push(s.service));
    await m.startService(MODEL);
    const loading = m.loadModel(OTHER);
    assert.deepEqual(m.status().service, { state: "running", model: OTHER, modelState: "loading" });
    await assert.rejects(m.synthesize(request), notReady);
    await loading;
    assert.deepEqual(m.status().service, { state: "running", model: OTHER, modelState: "loaded" });
    assert.equal(seen.find((s) => s.model === OTHER)?.modelState, "loading");
    const none = m.loadModel(null);
    assert.deepEqual(m.status().service, { state: "running", model: null, modelState: "none" });
    await none;
  });

  it("synthesize: TTS_NOT_READY while stopped or without a model; POSTs the request once loaded, returns it normalized", async () => {
    const m = installedManager();
    await assert.rejects(m.synthesize(request), notReady);
    await m.startService(null);
    await assert.rejects(m.synthesize(request), notReady);
    assert.deepEqual(fetchCalls, []);
    const wav = (sample: number) => {
      const bytes = Buffer.alloc(52);
      bytes.write("RIFF", 0, "latin1");
      bytes.writeUInt32LE(44, 4);
      bytes.write("WAVEfmt ", 8, "latin1");
      bytes.writeUInt32LE(16, 16);
      bytes.writeUInt16LE(1, 20);
      bytes.writeUInt16LE(1, 22);
      bytes.writeUInt32LE(32000, 24);
      bytes.writeUInt32LE(64000, 28);
      bytes.writeUInt16LE(2, 32);
      bytes.writeUInt16LE(16, 34);
      bytes.write("data", 36, "latin1");
      bytes.writeUInt32LE(8, 40);
      [sample, -sample, sample, -sample].forEach((s, i) => bytes.writeInt16LE(s, 44 + i * 2));
      return bytes;
    };
    let body: unknown;
    fetchImpl = async (_url, init) => {
      body = init.body === undefined ? undefined : JSON.parse(String(init.body));
      return new Response(wav(1000), { status: 200 });
    };
    await m.loadModel(MODEL);
    assert.deepEqual(await m.synthesize(request), wav(3277));
    assert.equal(fetchCalls.at(-1), "http://127.0.0.1:18999/tts");
    assert.deepEqual(body, { text: "你好", text_lang: "zh", ref_audio_path: "C:\\voices\\ref.wav", prompt_text: "参考", prompt_lang: "zh",
                             media_type: "wav", streaming_mode: false });
  });

  for (const [name, abort, respond, expected] of [
    ["non-200: TTS_SYNTHESIS with the exception", false,
     async () => new Response(JSON.stringify({ message: "tts failed", Exception: "CUDA out of memory" }), { status: 400 }),
     { status: 400, message: "tts failed", exception: "CUDA out of memory" }],
    ["aborted by signal: TTS_SYNTHESIS with abortReason", true,
     (init: RequestInit) => new Promise<Response>((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason))),
     { message: "This operation was aborted", abortReason: "AbortError" }],
  ] as const) {
    it(`synthesize: ${name}`, async () => {
      const m = installedManager();
      await m.startService(MODEL);
      fetchImpl = async (_url, init) => respond(init);
      const controller = new AbortController();
      const pending = m.synthesize({ ...request, signal: controller.signal });
      if (abort) {
        await waitFor(() => fetchCalls.at(-1) === "http://127.0.0.1:18999/tts");
        controller.abort();
      }
      await assert.rejects(pending, (e) => e instanceof TtsError && e.code === "TTS_SYNTHESIS" && JSON.stringify(e.details) === JSON.stringify(expected));
    });
  }

  for (const [name, downloadStatus, run, restartedIn] of [
    ["install", 200, (m: TtsManager) => m.install({ variant: "v2pro", root: ttsRoot }), finalDir],
    ["a failed install", 503, (m: TtsManager) => m.install({ variant: "v2pro", root: ttsRoot }), serviceDir],
    ["useExisting", 200, (m: TtsManager) => m.useExisting(userDir()), userDir],
  ] as const) {
    it(`${name} while running stops the service first, then starts it again with its model`, async () => {
      writeGptSovits(userDir());
      const m = installedManager();
      await m.startService(MODEL);
      const statesAtWork: string[] = [];
      fetchImpl = async (url) => {
        if (url !== FAKE.url) return new Response("success", { status: 200 });
        statesAtWork.push(m.status().service.state);
        return new Response(downloadStatus === 200 ? CONTENT : null, { status: downloadStatus });
      };
      probeTtsApi.mock.mockImplementation(async () => {
        statesAtWork.push(m.status().service.state);
      });
      await run(m);
      await waitFor(() => m.status().service.modelState === "loaded");
      assert.equal(statesAtWork.length > 0 && statesAtWork.every((state) => state === "stopped"), true);
      assert.equal(services[0].stop.mock.callCount(), 1);
      assert.deepEqual(startTtsService.mock.calls.map((c) => c.arguments[0]), [serviceDir(), restartedIn()]);
      assert.deepEqual(fetchCalls.filter((url) => url !== FAKE.url), [...weightsUrls(MODEL), ...weightsUrls(MODEL)]);
      assert.deepEqual(m.status().service, { state: "running", model: MODEL, modelState: "loaded" });
    });
  }

  it("every start and test run rewrites <root>/tts-infer.yaml and passes it; GPT-SoVITS folders are not written to", async () => {
    writeGptSovits(userDir());
    const m = installedManager();
    fs.writeFileSync(configPath(), "custom:\n  device: cpu\n");
    await m.startService(null);
    await m.stopService();
    fs.writeFileSync(configPath(), "broken");
    await m.startService(null);
    fs.writeFileSync(configPath(), "broken");
    assert.equal(await m.useExisting(userDir()), true);
    await waitFor(() => m.status().service.state === "running" && startTtsService.mock.callCount() === 3);
    assert.deepEqual(configs, [INFER_CONFIG, INFER_CONFIG, INFER_CONFIG, INFER_CONFIG]);
    for (const call of [...startTtsService.mock.calls, ...probeTtsApi.mock.calls]) assert.equal(call.arguments[1], configPath());
    for (const dir of [serviceDir(), userDir()]) assert.deepEqual(fs.readdirSync(dir).sort(), ["api_v2.py", "runtime"]);
  });

  it("an unexpected exit is failed with TTS_SERVICE and its stderr, and is not restarted", async () => {
    const m = installedManager();
    await m.startService(null);
    services[0].exit("api_v2.py exited with code 3");
    await waitFor(() => m.status().service.state === "failed");
    assert.deepEqual(m.status().service, { state: "failed", model: null, modelState: "none", errorCode: "TTS_SERVICE",
                                           errorDetails: { message: "api_v2.py exited with code 3", stderr: ["boot"] } });
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(startTtsService.mock.callCount(), 1);
  });

  for (const [command, starting] of [["stopService", true], ["dispose", false]] as const) {
    it(`${command}() ${starting ? "while starting leaves no running process" : "ends the process"}`, async () => {
      if (starting) startTtsService.mock.mockImplementation((dir, config, port, signal) =>
        new Promise((resolve) => signal.addEventListener("abort", () => resolve(startService(dir, config, port, signal)))));
      const m = installedManager();
      const start = m.startService(MODEL);
      if (starting) {
        await waitFor(() => startTtsService.mock.callCount() === 1);
        assert.equal(m.status().service.state, "starting");
      } else {
        await start;
      }
      await m[command]();
      await start;
      assert.equal(services.length, 1);
      assert.equal(services[0].stop.mock.callCount(), 1);
      assert.deepEqual(m.status().service, { state: "stopped", model: MODEL, modelState: "none" });
    });
  }

  it("useExisting right after an unawaited stopService() test-runs only once the stop has ended", async () => {
    writeGptSovits(userDir());
    const m = installedManager();
    await m.startService(null);
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let stopDone = false;
    services[0].stop.mock.mockImplementation(async () => {
      await held;
      stopDone = true;
    });
    const probedAfterStop: boolean[] = [];
    probeTtsApi.mock.mockImplementation(async () => {
      probedAfterStop.push(stopDone);
    });
    void m.stopService();
    const used = m.useExisting(userDir());
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(probeTtsApi.mock.callCount(), 0);
    release();
    assert.equal(await used, true);
    assert.deepEqual(probedAfterStop, [true]);
    assert.equal(startTtsService.mock.callCount(), 1);
  });

  it("loadModel(model) while starting loads the last recorded model once the start answers", async () => {
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    startTtsService.mock.mockImplementation(async (dir, config, port, signal) => {
      await held;
      return startService(dir, config, port, signal);
    });
    fetchImpl = async () => new Response("success", { status: 200 });
    const m = installedManager();
    const start = m.startService(OTHER);
    await waitFor(() => startTtsService.mock.callCount() === 1);
    assert.equal(m.status().service.state, "starting");
    const requested = m.loadModel(MODEL);
    release();
    await Promise.all([start, requested]);
    assert.deepEqual(fetchCalls, weightsUrls(MODEL));
    assert.deepEqual(m.status().service, { state: "running", model: MODEL, modelState: "loaded" });
  });

  it("startService(model) during an install starts the service when the install ends", async () => {
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    fetchImpl = async (url) => {
      if (url !== FAKE.url) return new Response("success", { status: 200 });
      await held;
      return new Response(CONTENT, { status: 200 });
    };
    const m = installedManager();
    const run = m.install({ variant: "v2pro", root: ttsRoot });
    await waitFor(() => fetchCalls.includes(FAKE.url));
    await m.startService(MODEL);
    assert.deepEqual(m.status().service, { state: "stopped", model: MODEL, modelState: "none" });
    assert.equal(startTtsService.mock.callCount(), 0);
    release();
    await run;
    await waitFor(() => m.status().service.modelState === "loaded");
    assert.deepEqual(startTtsService.mock.calls.map((c) => c.arguments[0]), [finalDir()]);
    assert.deepEqual(fetchCalls.filter((url) => url !== FAKE.url), weightsUrls(MODEL));
    assert.deepEqual(m.status().service, { state: "running", model: MODEL, modelState: "loaded" });
  });
});
