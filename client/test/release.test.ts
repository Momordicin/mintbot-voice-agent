// Against a real release (tools/package_release.py output): install, verify, repair,
// refuse tampered or incompatible packages, then supervise the installed agent on a
// replayed WAV, also over HTTP the way a GitHub release serves it. Skipped when no
// release is found, unless MINTBOT_TEST_REQUIRE=1 (release builds).
//   MINTBOT_TEST_MANIFEST  default: ../dist/<newest>/manifest.json
//   MINTBOT_TEST_WAV       default: SenseVoice's runtime/llama.cpp/tests/sample.wav
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import {
  API_VERSION,
  installVoiceAgent,
  readInstallation,
  staleVersions,
  VoiceAgentSupervisor,
  type AgentEvent,
  type Installation,
} from "../src/index.ts";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
function findManifest(): string | undefined {
  if (process.env.MINTBOT_TEST_MANIFEST) return process.env.MINTBOT_TEST_MANIFEST;
  const dist = path.join(repo, "dist");
  const versions = fs.existsSync(dist) ? fs.readdirSync(dist).filter((d) => fs.existsSync(path.join(dist, d, "manifest.json"))) : [];
  return versions.length ? path.join(dist, versions.sort().at(-1)!, "manifest.json") : undefined;
}
const manifest = findManifest();
const wav = process.env.MINTBOT_TEST_WAV ?? path.resolve(repo, "..", "SenseVoice", "runtime", "llama.cpp", "tests", "sample.wav");
const required = process.env.MINTBOT_TEST_REQUIRE === "1";
const skip = !manifest || !fs.existsSync(manifest) ? "no release found (run tools/package_release.py)" : false;
const wavSkip = fs.existsSync(wav) ? false : `no ${wav}`;
if (required && (skip || wavSkip)) throw new Error(`MINTBOT_TEST_REQUIRE=1 but ${skip || wavSkip}`);
const quiet = { info() {}, warn() {}, error() {} };

// Serves a release directory like github.com does: /<owner>/<repo>/releases/download/<tag>/<file>
// answers 302 to a storage host path, and releases/latest/download/<file> 302s to the tag.
async function githubLikeServer(dir: string, tag: string): Promise<{ base: string; close: () => Promise<void> }> {
  const server = http.createServer((req, res) => {
    const u = req.url ?? "";
    const latest = u.match(/^\/o\/r\/releases\/latest\/download\/(.+)$/);
    const tagged = u.match(/^\/o\/r\/releases\/download\/([^/]+)\/(.+)$/);
    const blob = u.match(/^\/storage\/(.+)$/);
    if (latest) {
      res.writeHead(302, { Location: `/o/r/releases/download/${tag}/${latest[1]}` });
      return res.end();
    }
    if (tagged && tagged[1] === tag) {
      res.writeHead(302, { Location: `/storage/${tagged[2]}?sig=x` });
      return res.end();
    }
    const file = blob && path.join(dir, decodeURIComponent(blob[1].split("?")[0]));
    if (file && fs.existsSync(file)) {
      res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": fs.statSync(file).size });
      return fs.createReadStream(file).pipe(res);
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as { port: number }).port;
  return { base: `http://127.0.0.1:${port}/o/r/releases`, close: () => new Promise((r) => server.close(() => r())) };
}

describe("release", { skip }, () => {
  let root: string;
  let inst: Installation;

  before(async () => {
    // a non-ASCII user name in %LOCALAPPDATA%, with characters outside any single ANSI
    // code page (Chinese and Korean), so nothing here may go through a narrow API
    root = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "voice-rel-")), "用户_张三_한글", "voice-agent");
    inst = await installVoiceAgent({ manifest: manifest!, root, apiVersion: API_VERSION });
  });
  after(() => fs.rmSync(path.dirname(path.dirname(root)), { recursive: true, force: true }));

  it("installs, and readInstallation finds the same paths", async () => {
    assert.ok(fs.existsSync(inst.exe) && fs.existsSync(inst.asr));
    assert.deepEqual(await readInstallation(root, { apiVersion: API_VERSION, verify: true }), inst);
    assert.equal(await readInstallation(root, { apiVersion: API_VERSION + 1 }), null);
    assert.deepEqual(await staleVersions(root), []);
  });

  it("skips an intact install", async () => {
    const msgs: string[] = [];
    await installVoiceAgent({ manifest: manifest!, root, apiVersion: API_VERSION, onProgress: (m) => msgs.push(m) });
    assert.equal(msgs.filter((m) => m.endsWith("already installed")).length, 2);
  });

  it("detects a damaged file and repairs it", async () => {
    const dll = path.join(path.dirname(inst.exe), "ggml-base.dll");
    const b = fs.readFileSync(dll);
    b[1000] ^= 0xff;
    fs.writeFileSync(dll, b);
    assert.equal(await readInstallation(root, { apiVersion: API_VERSION, verify: true }), null);
    const msgs: string[] = [];
    await installVoiceAgent({ manifest: manifest!, root, apiVersion: API_VERSION, onProgress: (m) => msgs.push(m) });
    assert.ok(msgs.some((m) => m.includes("win-x64.zip: installed")), "program package reinstalled");
    assert.ok(msgs.some((m) => m.includes("models") && m.endsWith("already installed")), "models untouched");
    assert.deepEqual(await readInstallation(root, { apiVersion: API_VERSION, verify: true }), inst);
  });

  it("refuses a tampered archive and leaves no partial version", async () => {
    const m = JSON.parse(fs.readFileSync(manifest!, "utf8"));
    const bad = fs.mkdtempSync(path.join(os.tmpdir(), "voice-bad-"));
    const root2 = path.join(bad, "root");
    try {
      fs.copyFileSync(manifest!, path.join(bad, "manifest.json"));
      const zip = fs.readFileSync(path.join(path.dirname(manifest!), m.packages.agent.file));
      zip[zip.length >> 1] ^= 0xff;
      fs.writeFileSync(path.join(bad, m.packages.agent.file), zip);
      fs.mkdirSync(path.join(root2, "models"), { recursive: true });
      // models already in place. Not fs.cpSync: on Windows, Node 22.22 crashes the process
      // (0xC0000409) when its source path is not ASCII; fs.promises.cp is fine.
      await fs.promises.cp(path.join(root, "models"), path.join(root2, "models"), { recursive: true });
      await assert.rejects(installVoiceAgent({ manifest: path.join(bad, "manifest.json"), root: root2, apiVersion: API_VERSION }),
                           /SHA-256 mismatch/);
      assert.deepEqual(fs.existsSync(path.join(root2, "agent")) ? fs.readdirSync(path.join(root2, "agent")) : [], []);
      assert.equal(fs.existsSync(path.join(root2, "current.json")), false);
    } finally {
      fs.rmSync(bad, { recursive: true, force: true });
    }
  });

  it("installs from release URLs with GitHub-style redirects (tagged and latest)", async () => {
    const m = JSON.parse(fs.readFileSync(manifest!, "utf8"));
    const srv = await githubLikeServer(path.dirname(manifest!), `v${m.version}`);
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "voice-http-"));
    try {
      for (const url of [`${srv.base}/download/v${m.version}/manifest.json`, `${srv.base}/latest/download/manifest.json`]) {
        const r = path.join(tmp, url.includes("latest") ? "latest" : "tagged");
        const got = await installVoiceAgent({ manifest: url, root: r, apiVersion: API_VERSION });
        assert.equal(got.version, m.version);
        assert.deepEqual(await readInstallation(r, { apiVersion: API_VERSION, verify: true }), got);
      }
    } finally {
      await srv.close();
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("refuses a manifest with another API version", async () => {
    await assert.rejects(installVoiceAgent({ manifest: manifest!, root, apiVersion: API_VERSION + 1 }), /speaks API/);
  });

  it("supervises the installed agent on a replayed WAV", { skip: wavSkip }, async () => {
    const events: AgentEvent[] = [];
    const speed = 4;
    const sup = new VoiceAgentSupervisor({
      installation: inst,
      args: ["--source", "wav:" + wav, "--replay-speed", String(speed)],
      log: quiet,
      onEvent: (e) => events.push(e),
    });
    try {
      await sup.start();
      assert.equal(sup.state.phase, "running");
      // 16 kHz mono 16-bit: the replay takes about (bytes / 32000) / speed seconds
      const end = Date.now() + (fs.statSync(wav).size / 32000 / speed) * 1000 + 30000;
      while (!events.some((e) => e.type === "device" && e.data.state === "ended")) {
        if (Date.now() > end) throw new Error("replay did not end");
        await new Promise((r) => setTimeout(r, 100));
      }
      for (;;) {
        const st = await sup.call("GET", "/v1/status");
        if (st.asr_queue === 0 && st.capture === "stopped") break;
        await new Promise((r) => setTimeout(r, 100));
      }
      const finals = events.filter((e) => e.type === "final");
      assert.ok(finals.length >= 1 && finals.every((e) => typeof e.data.text === "string" && e.data.text.length > 0));
    } finally {
      const pid = sup.state.pid;
      await sup.stop();
      if (pid) assert.throws(() => process.kill(pid, 0), "agent process is gone after stop()");
    }
  });
});
