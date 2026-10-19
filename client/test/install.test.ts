// installVoiceAgent: progress, cancellation, leftover cleanup, free-space check and
// error codes, against a real release served over HTTP. Skipped when no release is
// found, unless MINTBOT_TEST_REQUIRE=1.
//   MINTBOT_TEST_MANIFEST  default: ../dist/<newest>/manifest.json
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { API_VERSION, installVoiceAgent, VoiceInstallError, type InstallProgress } from "../src/index.ts";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
function findManifest(): string | undefined {
  if (process.env.MINTBOT_TEST_MANIFEST) return process.env.MINTBOT_TEST_MANIFEST;
  const dist = path.join(repo, "dist");
  const versions = fs.existsSync(dist) ? fs.readdirSync(dist).filter((d) => fs.existsSync(path.join(dist, d, "manifest.json"))) : [];
  return versions.length ? path.join(dist, versions.sort().at(-1)!, "manifest.json") : undefined;
}
const manifestPath = findManifest();
const skip = !manifestPath || !fs.existsSync(manifestPath) ? "no release found (run tools/package_release.py)" : false;
if (process.env.MINTBOT_TEST_REQUIRE === "1" && skip) throw new Error(`MINTBOT_TEST_REQUIRE=1 but ${skip}`);

// Serves named entries (a file on disk or a buffer) under /r/<name>, optionally throttled
// (all entries, or only those in `slow`), and records every request.
interface Entry { file?: string; body?: Buffer }
async function server(entries: Map<string, Entry>, bytesPerSecond = 0, slow?: Set<string>) {
  const requests: string[] = [];
  const srv = http.createServer(async (req, res) => {
    requests.push(req.url ?? "");
    const name = decodeURIComponent((req.url ?? "").replace(/^\/r\//, ""));
    const e = entries.get(name);
    if (!e) {
      res.writeHead(404);
      return res.end();
    }
    const data = e.body ?? fs.readFileSync(e.file!);
    res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Length": data.length });
    if (!bytesPerSecond || (slow && !slow.has(name))) return res.end(data);
    const slice = Math.max(1, Math.floor(bytesPerSecond / 20)); // 50 ms steps
    for (let i = 0; i < data.length && !res.destroyed; i += slice) {
      res.write(data.subarray(i, i + slice));
      await new Promise((r) => setTimeout(r, 50));
    }
    res.end();
  });
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
  const base = `http://127.0.0.1:${(srv.address() as { port: number }).port}/r/`;
  return { base, requests, close: () => new Promise<void>((r) => { srv.closeAllConnections(); srv.close(() => r()); }) };
}

describe("installVoiceAgent", { skip }, () => {
  const relDir = path.dirname(manifestPath ?? ".");
  let m: any;
  let tmp: string;
  let entries: Map<string, Entry>;
  const fresh = (name: string) => path.join(tmp, name);
  // the release's manifest with some changes, published under a new name
  const variant = (name: string, change: (m: any) => void) => {
    const v = JSON.parse(JSON.stringify(m));
    change(v);
    entries.set(name, { body: Buffer.from(JSON.stringify(v)) });
    return name;
  };
  const codeOf = async (p: Promise<unknown>) => {
    try {
      await p;
    } catch (e) {
      assert.ok(e instanceof VoiceInstallError, `a VoiceInstallError, got ${e}`);
      return e;
    }
    assert.fail("expected the install to fail");
  };

  before(() => {
    m = JSON.parse(fs.readFileSync(manifestPath!, "utf8"));
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), "voice-inst-"));
    entries = new Map([
      ["manifest.json", { file: manifestPath! }],
      [m.packages.agent.file, { file: path.join(relDir, m.packages.agent.file) }],
      [m.packages.models.file, { file: path.join(relDir, m.packages.models.file) }],
    ]);
  });
  after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  // download callbacks of one package: bytes = 0 first, bytes = total last, in between at
  // least 200 ms apart and each with new bytes (the last one may follow sooner)
  const checkThrottle = (dl: { p: InstallProgress; t: number }[], size: number) => {
    assert.equal(dl[0].p.bytes, 0, "bytes = 0 at the start");
    assert.equal(dl.at(-1)!.p.bytes, size, "bytes = total at the end");
    for (let i = 1; i < dl.length; i++) {
      assert.ok(dl[i].p.bytes! > dl[i - 1].p.bytes!, "only when new bytes arrived");
      assert.equal(dl[i].p.total, size);
      // 195: Date.now() and timer granularity on Windows
      if (i < dl.length - 1) assert.ok(dl[i].t - dl[i - 1].t >= 195, `>= 200 ms apart (${dl[i].t - dl[i - 1].t} ms)`);
    }
  };

  it("reports progress in order, bytes rising to the manifest size; an intact install only skips", async () => {
    const srv = await server(entries);
    try {
      const steps: InstallProgress[] = [];
      const timed: { p: InstallProgress; t: number }[] = [];
      await installVoiceAgent({ manifest: srv.base + "manifest.json", root: fresh("progress"), apiVersion: API_VERSION,
                                onStep: (p) => { steps.push(p); timed.push({ p, t: Date.now() }); } });
      assert.deepEqual([...new Set(steps.map((s) => s.package))], ["models", "agent"], "models first");
      for (const kind of ["models", "agent"] as const) {
        const s = steps.filter((x) => x.package === kind);
        const phases = s.map((x) => x.phase).filter((p, i, a) => p !== a[i - 1]);
        assert.deepEqual(phases, ["download", "verify", "extract", "done"], `${kind} phases`);
        const dl = s.filter((x) => x.phase === "download");
        const size = m.packages[kind].size;
        assert.equal(dl[0].bytes, 0);
        assert.equal(dl.at(-1)!.bytes, size, "ends at the manifest size");
        assert.ok(dl.every((x, i) => x.total === size && (i === 0 || x.bytes! > dl[i - 1].bytes!)), "strictly rising");
        assert.ok(s.filter((x) => x.phase !== "download").every((x) => x.bytes === undefined), "bytes only while downloading");
        checkThrottle(timed.filter((x) => x.p.package === kind && x.p.phase === "download"), size);
      }
      const again: InstallProgress[] = [];
      await installVoiceAgent({ manifest: srv.base + "manifest.json", root: fresh("progress"), apiVersion: API_VERSION,
                                onStep: (p) => again.push(p) });
      assert.deepEqual(again, [{ package: "models", phase: "skip" }, { package: "agent", phase: "skip" }]);
    } finally {
      await srv.close();
    }
  });

  it("on a slow link: at most 5 download callbacks per second, 0 first and total last", async () => {
    const srv = await server(entries, 1024 * 1024, new Set([m.packages.agent.file])); // the program at 1 MB/s
    try {
      const timed: { p: InstallProgress; t: number }[] = [];
      await installVoiceAgent({ manifest: srv.base + "manifest.json", root: fresh("slow"), apiVersion: API_VERSION,
                                onStep: (p) => timed.push({ p, t: Date.now() }) });
      const dl = timed.filter((x) => x.p.package === "agent" && x.p.phase === "download");
      const seconds = (dl.at(-1)!.t - dl[0].t) / 1000;
      assert.ok(seconds > 3, `the throttled download took ${seconds.toFixed(1)} s`);
      assert.ok(dl.length >= 10, `${dl.length} callbacks`);
      assert.ok(dl.length <= seconds * 5 + 2, `${dl.length} callbacks in ${seconds.toFixed(1)} s`);
      checkThrottle(dl, m.packages.agent.size);
    } finally {
      await srv.close();
    }
  });

  it("from a local manifest: no download phase", async () => {
    const steps: InstallProgress[] = [];
    await installVoiceAgent({ manifest: manifestPath!, root: fresh("local"), apiVersion: API_VERSION, onStep: (p) => steps.push(p) });
    assert.deepEqual(steps.map((s) => `${s.package}:${s.phase}`),
                     ["models:verify", "models:extract", "models:done", "agent:verify", "agent:extract", "agent:done"]);
  });

  it("cancelled mid-download: AbortError, temp removed, the installed version and current.json untouched", async () => {
    const root = fresh("progress"); // installed by the first test
    const current = fs.readFileSync(path.join(root, "current.json"));
    const name = variant("next.json", (v) => { v.packages.agent.version = "9.9.9"; });
    const srv = await server(entries, 2 * 1024 * 1024);
    try {
      const ac = new AbortController();
      const e = await codeOf(installVoiceAgent({ manifest: srv.base + name, root, apiVersion: API_VERSION, signal: ac.signal,
        onStep: (p) => { if (p.package === "agent" && p.phase === "download" && (p.bytes ?? 0) > 0) ac.abort(); } }));
      assert.equal(e.code, "ABORTED");
      assert.equal(e.name, "AbortError");
      assert.deepEqual(fs.readdirSync(path.join(root, "agent")), [m.packages.agent.version], "no 9.9.9, no temp directory");
      assert.deepEqual(fs.readdirSync(path.join(root, "models")), [m.packages.models.version]);
      assert.deepEqual(fs.readFileSync(path.join(root, "current.json")), current);
    } finally {
      await srv.close();
    }
  });

  it("cancelled while verifying or extracting: AbortError and nothing left", async () => {
    for (const phase of ["verify", "extract"] as const) {
      const root = fresh(`cancel-${phase}`);
      const ac = new AbortController();
      // 100 ms into the phase: the 270 MB model archive is being hashed / tar is unpacking it
      const e = await codeOf(installVoiceAgent({ manifest: manifestPath!, root, apiVersion: API_VERSION, signal: ac.signal,
        onStep: (p) => { if (p.package === "models" && p.phase === phase) setTimeout(() => ac.abort(), 100); } }));
      assert.equal(e.code, "ABORTED", phase);
      assert.deepEqual(fs.readdirSync(path.join(root, "models")), [], `${phase}: no version, no temp directory`);
      assert.equal(fs.existsSync(path.join(root, "current.json")), false);
    }
    const e = await codeOf(installVoiceAgent({ manifest: manifestPath!, root: fresh("pre-aborted"), apiVersion: API_VERSION,
                                               signal: AbortSignal.abort() }));
    assert.equal(e.code, "ABORTED");
  });

  it("deletes temp directories an interrupted install left behind", async () => {
    const root = fresh("progress");
    const leftovers = [path.join(root, "agent", ".9.9.9.tmp-deadbeef"), path.join(root, "models", ".1.tmp-0123abcd")];
    for (const d of leftovers) {
      fs.mkdirSync(path.join(d, "x"), { recursive: true });
      fs.writeFileSync(path.join(d, "x", "big.bin"), Buffer.alloc(4 * 1024 * 1024));
    }
    const log: string[] = [];
    await installVoiceAgent({ manifest: manifestPath!, root, apiVersion: API_VERSION, onProgress: (l) => log.push(l) });
    for (const d of leftovers) assert.equal(fs.existsSync(d), false, d);
    assert.equal(log.filter((l) => l.startsWith("removed leftover")).length, 2);
  });

  it("not enough free space: ENOSPC_PRECHECK with the numbers, before any download", async () => {
    const name = variant("huge.json", (v) => { v.packages.models.version = "77"; v.packages.models.size = 2 ** 52; });
    const srv = await server(entries);
    try {
      const e = await codeOf(installVoiceAgent({ manifest: srv.base + name, root: fresh("progress"), apiVersion: API_VERSION }));
      assert.equal(e.code, "ENOSPC_PRECHECK");
      assert.equal(e.details.required, 2 ** 53, "zip + extracted files");
      assert.ok(typeof e.details.available === "number" && (e.details.available as number) > 0);
      assert.deepEqual(srv.requests, ["/r/" + name], "only the manifest was fetched");
    } finally {
      await srv.close();
    }
  });

  it("error codes: NETWORK_ERROR, HTTP_STATUS, SHA256_MISMATCH, API_VERSION_MISMATCH, INVALID_MANIFEST", async () => {
    const srv = await server(entries);
    try {
      const closed = await server(new Map());
      await closed.close();
      const net = await codeOf(installVoiceAgent({ manifest: closed.base + "manifest.json", root: fresh("e1"), apiVersion: API_VERSION }));
      assert.equal(net.code, "NETWORK_ERROR");

      const missing = variant("missing.json", (v) => { v.packages.agent.version = "8.8.8"; v.packages.agent.file = "gone.zip"; });
      const http404 = await codeOf(installVoiceAgent({ manifest: srv.base + missing, root: fresh("progress"), apiVersion: API_VERSION }));
      assert.equal(http404.code, "HTTP_STATUS");
      assert.equal(http404.details.status, 404);

      const zip = fs.readFileSync(path.join(relDir, m.packages.agent.file));
      zip[zip.length >> 1] ^= 0xff;
      entries.set("tampered.zip", { body: zip });
      const tampered = variant("tampered.json", (v) => { v.packages.agent.version = "6.6.6"; v.packages.agent.file = "tampered.zip"; });
      const sha = await codeOf(installVoiceAgent({ manifest: srv.base + tampered, root: fresh("progress"), apiVersion: API_VERSION }));
      assert.equal(sha.code, "SHA256_MISMATCH");

      const api = await codeOf(installVoiceAgent({ manifest: srv.base + "manifest.json", root: fresh("e2"), apiVersion: API_VERSION + 1 }));
      assert.equal(api.code, "API_VERSION_MISMATCH");
      assert.deepEqual([api.details.expected, api.details.actual], [API_VERSION + 1, m.api_version]);

      entries.set("junk.json", { body: Buffer.from("<html>not json</html>") });
      const junk = await codeOf(installVoiceAgent({ manifest: srv.base + "junk.json", root: fresh("e3"), apiVersion: API_VERSION }));
      assert.equal(junk.code, "INVALID_MANIFEST");

      // none of the failed installs left a version or temp directory behind
      assert.deepEqual(fs.readdirSync(path.join(fresh("progress"), "agent")), [m.packages.agent.version]);
    } finally {
      await srv.close();
    }
  });
});
