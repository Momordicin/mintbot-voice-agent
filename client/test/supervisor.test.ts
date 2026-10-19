// VoiceAgentSupervisor against test/fake-agent.mjs (no models, no microphone).
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { VoiceAgentSupervisor, type AgentEvent, type SupervisorOptions, type SupervisorState } from "../src/index.ts";

const FAKE = path.join(path.dirname(fileURLToPath(import.meta.url)), "fake-agent.mjs");
const installation = { version: "0.1.0", apiVersion: 1, exe: "unused.exe", vad: "v.gguf", asr: "a.gguf", speaker: "s.gguf" };
const quiet = { info() {}, warn() {}, error() {} };

let dir: string;
let sup: VoiceAgentSupervisor | null = null;
const FAKE_VARS = ["FAKE_EXIT", "FAKE_CRASH_RUNS", "FAKE_CRASH_MS", "FAKE_MIC_DENIED", "FAKE_SSE_DROP", "FAKE_START_DELAY"];

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "voice-sup-"));
  process.env.FAKE_STATE = path.join(dir, "launches");
  process.env.FAKE_LOG = path.join(dir, "log.jsonl");
  for (const v of FAKE_VARS) delete process.env[v];
});

afterEach(async () => {
  await sup?.stop();
  sup = null;
  fs.rmSync(dir, { recursive: true, force: true });
});

function fakeLog(): any[] {
  return fs.existsSync(process.env.FAKE_LOG!)
    ? fs.readFileSync(process.env.FAKE_LOG!, "utf8").trim().split("\n").map((l) => JSON.parse(l))
    : [];
}

async function until(what: string, pred: () => boolean, ms = 5000) {
  const end = Date.now() + ms;
  while (!pred()) {
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}

function make(extra: Partial<SupervisorOptions> = {}) {
  const events: AgentEvent[] = [];
  const states: SupervisorState[] = [];
  sup = new VoiceAgentSupervisor({
    installation,
    command: { file: process.execPath, args: [FAKE] },
    log: quiet,
    onEvent: (e) => events.push(e),
    onState: (s) => states.push(s),
    listenRetryMs: 50,
    restart: { initialDelayMs: 50, maxDelayMs: 200, maxFailures: 3 },
    ...extra,
  });
  return { sup, events, states };
}

describe("VoiceAgentSupervisor", () => {
  it("starts, listens and forwards events; passes the voiceprint flags, never --record", async () => {
    const { sup, events } = make({ voiceprintPath: "C:\\data\\owner.voiceprint.json", speakerThreshold: 0.5 });
    await sup.start();
    await until("a final event", () => events.some((e) => e.type === "final"));
    assert.equal(sup.state.phase, "running");
    assert.equal(sup.state.listening, true);
    assert.equal(sup.state.capture, "running");
    const args: string[] = fakeLog()[0].args;
    assert.deepEqual(args.slice(args.indexOf("--speaker-db"), args.indexOf("--speaker-db") + 2),
                     ["--speaker-db", "C:\\data\\owner.voiceprint.json"]);
    assert.deepEqual(args.slice(args.indexOf("--speaker-threshold"), args.indexOf("--speaker-threshold") + 2),
                     ["--speaker-threshold", "0.5"]);
    assert.ok(!args.includes("--record"));
    assert.equal(fakeLog()[0].token, true, "token is passed in the environment");
    assert.ok(!args.some((a) => /^[0-9a-f]{64}$/.test(a)), "token is not on the command line");
  });

  it("keeps retrying listen/start while the microphone is denied", async () => {
    process.env.FAKE_MIC_DENIED = "3";
    const { sup, states } = make();
    await sup.start();
    assert.equal(sup.state.error?.code, "mic_access_denied");
    await until("listening", () => sup.state.listening);
    assert.equal(sup.state.error, undefined);
    assert.ok(states.some((s) => s.error?.code === "mic_access_denied"));
    assert.equal(fakeLog().filter((l) => l.req?.startsWith("POST /v1/listen/start")).length, 4);
  });

  it("restarts a crashed agent with a new process and listens again", async () => {
    process.env.FAKE_CRASH_RUNS = "1";
    const { sup } = make();
    await sup.start();
    const first = sup.state.pid;
    await until("restart", () => sup.state.phase === "running" && sup.state.pid !== first && sup.state.listening);
    assert.equal(sup.state.restarts, 1);
    assert.equal(fakeLog().filter((l) => l.args).length, 2);
  });

  it("gives up after maxFailures consecutive crashes", async () => {
    process.env.FAKE_CRASH_RUNS = "99";
    process.env.FAKE_CRASH_MS = "10";
    const { sup } = make();
    await sup.start();
    await until("failed", () => sup.state.phase === "failed");
    assert.equal(fakeLog().filter((l) => l.args).length, 3);
    await new Promise((r) => setTimeout(r, 300));
    assert.equal(fakeLog().filter((l) => l.args).length, 3, "no further restarts");
  });

  it("does not restart after an invalid-arguments exit (code 2)", async () => {
    process.env.FAKE_EXIT = "2";
    const { sup } = make();
    await sup.start();
    assert.equal(sup.state.phase, "failed");
    await new Promise((r) => setTimeout(r, 200));
    assert.equal(fakeLog().filter((l) => l.args).length, 1);
  });

  it("reconnects a dropped event stream and emits a status snapshot", async () => {
    process.env.FAKE_SSE_DROP = "1";
    const { sup, events } = make({ listen: false });
    await sup.start();
    await until("snapshot", () => events.some((e) => e.type === "snapshot"));
    const snap = events.find((e) => e.type === "snapshot")!;
    assert.deepEqual(snap.data, { listening: false, capture: "stopped" });
    assert.equal(fakeLog().filter((l) => l.req === "GET /v1/events").length, 2);
  });

  it("stop() ends the process and no restart follows", async () => {
    const { sup } = make();
    await sup.start();
    const pid = sup.state.pid!;
    await sup.stop();
    assert.equal(sup.state.phase, "stopped");
    assert.throws(() => process.kill(pid, 0), "process is gone");
    await new Promise((r) => setTimeout(r, 200));
    assert.equal(fakeLog().filter((l) => l.args).length, 1);
    await assert.rejects(sup.call("GET", "/v1/status"), /stopped/);
  });

  it("stop() during start shuts the starting process down (listening tests follow)", async () => {
    const { sup } = make();
    const starting = sup.start();
    await sup.stop();
    await starting;
    // stop() waited for the start in flight, which shut its process down on becoming ready
    assert.ok(fakeLog().some((l) => l.req === "POST /v1/shutdown"), "the starting process was told to shut down");
    assert.equal(fakeLog().filter((l) => l.args).length, 1);
    assert.equal(sup.state.phase, "stopped");
  });
});

const reqs = (prefix: string) => fakeLog().filter((l) => typeof l.req === "string" && l.req.startsWith(prefix));
const starts = () => reqs("POST /v1/listen/start").length;
const stops = () => reqs("POST /v1/listen/stop").length;
const agentListening = async (s: VoiceAgentSupervisor) => (await s.call("GET", "/v1/status")).listening as boolean;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("setListening", () => {
  it("1. off while running: sends listen/stop, listening and listenWanted become false", async () => {
    const { sup, states } = make();
    await sup.start();
    assert.equal(sup.state.listening, true);
    sup.setListening(false);
    await until("not listening", () => !sup.state.listening);
    assert.equal(stops(), 1);
    assert.equal(sup.state.listenWanted, false);
    assert.equal(await agentListening(sup), false);
    assert.ok(states.some((s) => !s.listenWanted && s.listening), "listenWanted change was notified before listening");
  });

  it("2. off while listen/start is being retried (microphone blocked): retries stop for good", async () => {
    process.env.FAKE_MIC_DENIED = "999";
    const { sup } = make();
    await sup.start();
    await until("a few retries", () => starts() >= 3);
    assert.equal(sup.state.error?.code, "mic_access_denied");
    sup.setListening(false);
    const n = starts();
    await sleep(400); // eight retry periods
    assert.equal(starts(), n, "no listen/start after setListening(false)");
    assert.equal(sup.state.error, undefined, "the listen error is cleared");
    assert.equal(sup.state.listening, false);
  });

  it("3. crash restart while off: the new agent is not asked to listen", async () => {
    process.env.FAKE_CRASH_RUNS = "1";
    process.env.FAKE_CRASH_MS = "300";
    const { sup } = make();
    await sup.start();
    sup.setListening(false);
    const first = sup.state.pid;
    await until("restart", () => sup.state.phase === "running" && sup.state.pid !== first);
    await sleep(200);
    assert.equal(fakeLog().filter((l) => l.launch === 2 && l.req?.startsWith("POST /v1/listen/start")).length, 0);
    assert.equal(sup.state.listening, false);
    assert.equal(await agentListening(sup), false);
  });

  it("4. off then on: listening resumes", async () => {
    const { sup } = make();
    await sup.start();
    sup.setListening(false);
    await until("off", () => !sup.state.listening && stops() === 1);
    sup.setListening(true);
    await until("on again", () => sup.state.listening);
    assert.equal(starts(), 2);
    assert.equal(await agentListening(sup), true);
  });

  it("5. off while listen/start is in flight: listen/stop follows its answer, the microphone ends off", async () => {
    process.env.FAKE_START_DELAY = "300";
    const { sup } = make({ listen: false });
    await sup.start();
    sup.setListening(true);
    await until("listen/start sent", () => starts() === 1);
    sup.setListening(false); // the agent has already opened the device; its answer is still on the way
    await until("listen/stop sent", () => stops() === 1, 3000);
    await until("settled", () => !sup.state.listening);
    await sleep(100);
    const log = fakeLog().filter((l) => l.req?.startsWith("POST /v1/listen/"));
    assert.deepEqual(log.map((l) => l.req), ["POST /v1/listen/start", "POST /v1/listen/stop"]);
    assert.equal(await agentListening(sup), false);
  });

  it("6. rapid toggling (false, true, false / true, false, true): the last call wins", async () => {
    process.env.FAKE_START_DELAY = "50";
    for (const seq of [[false, true, false], [true, false, true]]) {
      const { sup } = make({ listen: !seq[0] });
      await sup.start();
      for (const on of seq) sup.setListening(on);
      const last = seq[seq.length - 1];
      await until("settled", () => sup.state.listening === last, 3000);
      await sleep(300); // nothing in flight may flip it afterwards
      assert.equal(sup.state.listening, last);
      assert.equal(sup.state.listenWanted, last);
      assert.equal(await agentListening(sup), last);
      await sup.stop();
    }
  });

  it("7. calls while not running only record the wish, applied when the agent is ready", async () => {
    const { sup } = make();
    sup.setListening(false); // stopped
    assert.equal(sup.state.listenWanted, false);
    await sup.start();
    assert.equal(starts(), 0);
    assert.equal(sup.state.listening, false);
    await sup.stop();
    sup.setListening(true); // stopped again: no request, no throw
    assert.equal(sup.state.listenWanted, true);
    await sup.start();
    await until("listening", () => sup.state.listening);
    assert.equal(starts(), 1);
  });

  it("7b. a wish set while restarting applies to the restarted agent", async () => {
    process.env.FAKE_CRASH_RUNS = "1";
    process.env.FAKE_CRASH_MS = "100";
    const { sup } = make({ listen: false, restart: { initialDelayMs: 300, maxDelayMs: 300, maxFailures: 3 } });
    await sup.start();
    await until("restarting", () => sup.state.phase === "restarting");
    sup.setListening(true);
    await until("listening after restart", () => sup.state.phase === "running" && sup.state.listening);
    assert.equal(fakeLog().filter((l) => l.launch === 2 && l.req?.startsWith("POST /v1/listen/start")).length, 1);
  });

  it("8. listen: false: never listens after start; listenWanted is false", async () => {
    const { sup } = make({ listen: false });
    assert.equal(sup.state.listenWanted, false);
    await sup.start();
    await sleep(200);
    assert.equal(starts(), 0);
    assert.equal(sup.state.listening, false);
    assert.equal(sup.state.listenWanted, false);
  });
});
