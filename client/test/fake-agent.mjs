// Stand-in for mintbot-voice-agent.exe in the supervisor tests: same ready line,
// token check, endpoints and stdin-EOF exit, with failures chosen by env:
//   FAKE_STATE        file counting launches (shared by all processes of one test)
//   FAKE_LOG          JSON lines: {launch, args} per start, {launch, req} per request
//   FAKE_EXIT         exit with this code before the ready line
//   FAKE_CRASH_RUNS   the first N launches crash (exit 3) FAKE_CRASH_MS after ready
//   FAKE_MIC_DENIED   the first N listen/start calls of each process fail (mic_access_denied)
//   FAKE_SSE_DROP     the first SSE connection of each process closes after one event
//   FAKE_START_DELAY  listen/start answers after this many ms (to test requests in flight)
import fs from "node:fs";
import http from "node:http";

const env = process.env;
const launch = (() => {
  const n = (fs.existsSync(env.FAKE_STATE) ? Number(fs.readFileSync(env.FAKE_STATE, "utf8")) : 0) + 1;
  fs.writeFileSync(env.FAKE_STATE, String(n));
  return n;
})();
const log = (o) => env.FAKE_LOG && fs.appendFileSync(env.FAKE_LOG, JSON.stringify({ launch, ...o }) + "\n");
log({ args: process.argv.slice(2), token: Boolean(env.MINTBOT_VOICE_TOKEN) });

if (env.FAKE_EXIT) process.exit(Number(env.FAKE_EXIT));
if (!env.MINTBOT_VOICE_TOKEN) process.exit(1);

let micDenied = Number(env.FAKE_MIC_DENIED ?? 0);
let dropFirstSse = env.FAKE_SSE_DROP === "1";
let listening = false;
const clients = new Set();
const send = (res, type, data) => res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
const broadcast = (type, data) => clients.forEach((c) => send(c, type, data));
const json = (res, code, body) => {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
};

const server = http.createServer((req, res) => {
  log({ req: `${req.method} ${req.url}` });
  if (req.url === "/v1/health") return json(res, 200, { status: "ok", version: "0.1.0", api_version: 1 });
  if (req.headers.authorization !== `Bearer ${env.MINTBOT_VOICE_TOKEN}`) return json(res, 401, { error: "unauthorized" });
  if (req.method === "GET" && req.url === "/v1/status") return json(res, 200, { listening, capture: listening ? "running" : "stopped" });
  if (req.method === "POST" && req.url.startsWith("/v1/listen/start")) {
    // like the agent: the device is opened when the request arrives, the answer may come later
    const denied = micDenied > 0;
    if (denied) micDenied--;
    else listening = true;
    setTimeout(() => {
      if (denied) return json(res, 500, { error: "capture start failed: Access denied. (-6)", code: "mic_access_denied" });
      json(res, 200, { listening });
      broadcast("capture", { state: "running", code: "", message: "" });
      broadcast("final", { text: "你好", start_ms: 0, end_ms: 1000 });
    }, Number(env.FAKE_START_DELAY ?? 0));
    return;
  }
  if (req.method === "POST" && req.url === "/v1/listen/stop") {
    const was = listening;
    listening = false;
    json(res, 200, { listening });
    if (was) broadcast("capture", { state: "stopped", code: "", message: "" });
    return;
  }
  if (req.method === "POST" && req.url === "/v1/shutdown") {
    json(res, 200, { ok: true });
    setTimeout(() => process.exit(0), 20);
    return;
  }
  if (req.method === "GET" && req.url === "/v1/events") {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    send(res, "status", { listening });
    if (dropFirstSse) {
      dropFirstSse = false;
      setTimeout(() => res.destroy(), 50);
      return;
    }
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }
  json(res, 404, { error: "not found" });
});

server.listen(0, "127.0.0.1", () => {
  process.stdout.write(JSON.stringify({ type: "ready", port: server.address().port, pid: process.pid,
                                        version: "0.1.0", api_version: 1 }) + "\n");
  if (launch <= Number(env.FAKE_CRASH_RUNS ?? 0)) setTimeout(() => process.exit(3), Number(env.FAKE_CRASH_MS ?? 100));
});
process.stdin.on("end", () => process.exit(0));
process.stdin.resume();
