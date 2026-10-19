// Keeps one mintbot-voice-agent running for the lifetime of the host process:
// start -> subscribe to events -> listen if wanted; restart after a crash with a
// growing delay; retry listening while the microphone is unavailable; reconnect
// the event stream and read a status snapshot when it drops. setListening()
// switches the microphone on and off for good: no restart, retry or reconnect
// opens it while listening is not wanted.
import { startAgentProcess, type AgentProcess } from "./agent.ts";
import type { Installation } from "./install.ts";
import { API_VERSION, AgentHttpError, type AgentEvent } from "./protocol.ts";

export type SupervisorPhase =
  | "stopped" // not started, or stop() completed
  | "starting" // process launched, waiting for its ready line
  | "running" // ready; see `listening` / `capture` for the microphone
  | "restarting" // the process exited unexpectedly; waiting before the next start
  | "failed"; // gave up: see `error`; call start() again to retry

export interface SupervisorState {
  phase: SupervisorPhase;
  pid?: number;
  version?: string;
  // what the host asked for (setListening / the `listen` option)
  listenWanted: boolean;
  // what the agent is doing: listening was requested and accepted
  listening: boolean;
  // the agent's last `capture` event: running | retrying | stopped
  capture?: string;
  // last problem, e.g. { code: "mic_access_denied" } while the microphone is blocked
  error?: { code?: string; message: string };
  restarts: number;
}

export interface Logger {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}

export interface SupervisorOptions {
  installation: Installation;
  // Owner voiceprint file (created by enrollment); omitted = no speaker gate,
  // every utterance is transcribed.
  voiceprintPath?: string;
  speakerThreshold?: number; // the user's own value; the agent's default is 0.5
  device?: number; // capture device index; omitted = the system default, following changes
  listen?: boolean; // initial listenWanted (default true); change it with setListening()
  args?: string[]; // further agent flags (see mintbot-voice-agent.exe --help)
  onEvent?: (ev: AgentEvent) => void;
  onState?: (s: SupervisorState) => void;
  log?: Logger;
  readyTimeoutMs?: number; // default 60000
  listenRetryMs?: number; // while listen/start fails (microphone blocked or absent); default 5000
  restart?: {
    initialDelayMs?: number; // default 2000, doubled per consecutive failure
    maxDelayMs?: number; // default 60000
    maxFailures?: number; // consecutive failures before giving up; default 5
    stableAfterMs?: number; // a run this long resets the failure count; default 60000
  };
  // Tests only: run this instead of installation.exe, with these leading arguments.
  command?: { file: string; args: string[] };
}

const consoleLog: Logger = {
  info: (m) => console.log(`[VoiceAgent] ${m}`),
  warn: (m) => console.warn(`[VoiceAgent] ${m}`),
  error: (m) => console.error(`[VoiceAgent] ${m}`),
};

// Exit code of the agent for invalid command-line arguments: restarting cannot help.
const EXIT_BAD_ARGS = 2;

// Listening bookkeeping of one agent process (a restarted agent gets a fresh one).
interface ListenLink {
  agent: AgentProcess;
  gen: number;
  busy: boolean; // a listen/start or listen/stop request is in flight
  maybeOn: boolean; // a listen/start was sent since the last successful listen/stop
  retry: ReturnType<typeof setTimeout> | null; // pending listen/start retry
}

export class VoiceAgentSupervisor {
  private readonly o: SupervisorOptions;
  private readonly log: Logger;
  private agent: AgentProcess | null = null;
  private wanted = false; // between start() and stop()
  private listenWanted: boolean;
  private gen = 0; // increments per launched process; stale callbacks compare against it
  private failures = 0;
  private timers = new Set<ReturnType<typeof setTimeout>>();
  private abort: AbortController | null = null;
  private launching: Promise<void> | null = null;
  private link: ListenLink | null = null;
  private s: SupervisorState;

  constructor(o: SupervisorOptions) {
    if (o.installation.apiVersion !== API_VERSION)
      throw new Error(`installation speaks API ${o.installation.apiVersion}, this client ${API_VERSION}`);
    this.o = o;
    this.log = o.log ?? consoleLog;
    this.listenWanted = o.listen ?? true;
    this.s = { phase: "stopped", listenWanted: this.listenWanted, listening: false, restarts: 0 };
  }

  get state(): SupervisorState {
    return { ...this.s };
  }

  // Launches the agent and resolves once the first attempt has finished (running, or
  // scheduled for a restart / failed: watch onState). Never rejects for agent problems.
  async start(): Promise<void> {
    if (this.wanted) return;
    this.wanted = true;
    this.failures = 0;
    this.set({ phase: "starting", error: undefined, restarts: 0 });
    await this.relaunch();
  }

  private relaunch(): Promise<void> {
    const p = this.launch().finally(() => {
      if (this.launching === p) this.launching = null;
    });
    this.launching = p;
    return p;
  }

  // Stops the agent (graceful shutdown, killed after 3 s) and cancels pending restarts.
  async stop(): Promise<void> {
    this.wanted = false;
    this.gen++;
    for (const t of this.timers) clearTimeout(t);
    this.timers.clear();
    this.link = null;
    this.abort?.abort();
    this.abort = null;
    const a = this.agent;
    this.agent = null;
    if (a) await a.shutdown();
    await this.launching; // a start in flight shuts its process down itself (gen changed)
    this.set({ phase: "stopped", pid: undefined, listening: false, capture: undefined });
  }

  // Whether the microphone should be used. Takes effect at once on a running agent
  // (on: listen/start, retried while it fails; off: cancels the retry and sends
  // listen/stop) and on every later start or restart; the last call wins. Never
  // throws; while the agent is not running only the wish is recorded.
  setListening(on: boolean): void {
    this.listenWanted = on;
    const patch: Partial<SupervisorState> = { listenWanted: on };
    if (!on) {
      this.cancelListenRetry();
      if (this.s.error?.code) patch.error = undefined; // a listen error no longer applies
    }
    this.set(patch);
    const l = this.link;
    if (l && this.s.phase === "running") void this.reconcileListening(l);
  }

  // REST call on the running agent, e.g. call("POST", "/v1/speaker/enroll/start").
  async call<T = any>(method: "GET" | "POST" | "DELETE", path: string): Promise<T> {
    if (!this.agent || this.s.phase !== "running") throw new Error(`voice agent is ${this.s.phase}`);
    return this.agent.call<T>(method, path);
  }

  private set(p: Partial<SupervisorState>) {
    const next = { ...this.s, ...p };
    if (JSON.stringify(next) === JSON.stringify(this.s)) return;
    this.s = next;
    try {
      this.o.onState?.(this.state);
    } catch (e) {
      this.log.error(`onState threw: ${e}`);
    }
  }

  private emit(ev: AgentEvent) {
    if (ev.type === "capture") this.set({ capture: ev.data.state });
    try {
      this.o.onEvent?.(ev);
    } catch (e) {
      this.log.error(`onEvent threw: ${e}`);
    }
  }

  private later(ms: number, fn: () => void): ReturnType<typeof setTimeout> {
    const t = setTimeout(() => {
      this.timers.delete(t);
      fn();
    }, ms);
    this.timers.add(t);
    return t;
  }

  private agentArgs(): string[] {
    const i = this.o.installation;
    const a = ["--vad", i.vad, "-m", i.asr, "--spk-model", i.speaker];
    if (this.o.voiceprintPath) a.push("--speaker-db", this.o.voiceprintPath);
    if (this.o.speakerThreshold !== undefined) a.push("--speaker-threshold", String(this.o.speakerThreshold));
    return [...a, ...(this.o.args ?? [])];
  }

  private async launch(): Promise<void> {
    const gen = ++this.gen;
    this.set({ phase: "starting", pid: undefined, listening: false, capture: undefined });
    const file = this.o.command?.file ?? this.o.installation.exe;
    const args = [...(this.o.command?.args ?? []), ...this.agentArgs()];
    let agent: AgentProcess;
    const startedAt = Date.now();
    try {
      agent = await startAgentProcess({
        file,
        args,
        readyTimeoutMs: this.o.readyTimeoutMs,
        onStderr: (line) => this.log.info(line),
      });
    } catch (e) {
      if (gen !== this.gen) return;
      const code = (e as { exitCode?: number | null }).exitCode ?? null;
      this.onExit(gen, code, startedAt, `start failed: ${(e as Error).message}`);
      return;
    }
    if (gen !== this.gen || !this.wanted) {
      await agent.shutdown();
      return;
    }
    this.agent = agent;
    agent.exited.then((code) => this.onExit(gen, code, startedAt, `exited with code ${code}`));
    this.set({ phase: "running", pid: agent.ready.pid, version: agent.ready.version, error: undefined });
    this.log.info(`running (pid ${agent.ready.pid}, version ${agent.ready.version})`);
    this.abort = new AbortController();
    this.subscribe(gen, agent, this.abort.signal, false);
    this.link = { agent, gen, busy: false, maybeOn: false, retry: null };
    await this.reconcileListening(this.link);
  }

  private onExit(gen: number, code: number | null, startedAt: number, why: string) {
    if (gen !== this.gen || !this.wanted) return; // stop() or a newer process
    this.agent = null;
    this.cancelListenRetry();
    this.link = null;
    this.abort?.abort();
    this.abort = null;
    const r = this.o.restart ?? {};
    if (Date.now() - startedAt >= (r.stableAfterMs ?? 60000)) this.failures = 0;
    this.failures++;
    const message = `voice agent ${why}`;
    if (code === EXIT_BAD_ARGS || this.failures >= (r.maxFailures ?? 5)) {
      this.log.error(`${message}; giving up after ${this.failures} attempt(s)`);
      this.wanted = false;
      this.set({ phase: "failed", pid: undefined, listening: false, capture: undefined, error: { message } });
      return;
    }
    const delay = Math.min((r.initialDelayMs ?? 2000) * 2 ** (this.failures - 1), r.maxDelayMs ?? 60000);
    this.log.warn(`${message}; restarting in ${delay} ms`);
    this.set({ phase: "restarting", pid: undefined, listening: false, capture: undefined, error: { message },
               restarts: this.s.restarts + 1 });
    this.later(delay, () => {
      if (gen === this.gen && this.wanted) void this.relaunch();
    });
  }

  private cancelListenRetry() {
    const l = this.link;
    if (!l?.retry) return;
    clearTimeout(l.retry);
    this.timers.delete(l.retry);
    l.retry = null;
  }

  // Brings the agent's listening in line with listenWanted. One request at a time
  // per agent; when it returns, the wish is checked again, so the last setListening()
  // wins however fast it is toggled.
  private async reconcileListening(l: ListenLink): Promise<void> {
    const current = () => this.link === l && l.gen === this.gen; // not stopped, not crashed
    if (!current() || l.busy) return; // the request in flight reconciles when it returns
    let again = true;
    l.busy = true;
    try {
      if (this.listenWanted) {
        if (this.s.listening || l.retry) return; // on, or a retry is already scheduled
        l.maybeOn = true;
        again = await this.sendListenStart(l, current);
      } else {
        // even a listen/start that failed may have raced with the device: always stop after one
        if (!l.maybeOn && !this.s.listening) return;
        again = await this.sendListenStop(l, current);
      }
    } finally {
      l.busy = false;
    }
    if (again && current()) await this.reconcileListening(l);
  }

  // listen/start; false when it failed (a retry is scheduled if still wanted).
  private async sendListenStart(l: ListenLink, current: () => boolean): Promise<boolean> {
    const q = this.o.device !== undefined ? `?device=${this.o.device}` : "";
    try {
      const st = await l.agent.call<{ listening?: boolean }>("POST", "/v1/listen/start" + q);
      if (current()) this.set({ listening: st?.listening ?? true, error: undefined });
      return true;
    } catch (e) {
      if (!current()) return false;
      const code = e instanceof AgentHttpError ? e.code : undefined;
      const message = e instanceof AgentHttpError ? String(e.body?.error ?? e.message) : (e as Error).message;
      const retryMs = this.o.listenRetryMs ?? 5000;
      this.set({ listening: false });
      if (!this.listenWanted) return true; // turned off meanwhile: reconcile sends listen/stop
      if (this.s.error?.code !== code || this.s.error?.message !== message)
        this.log.warn(`listen/start failed (${code ?? "error"}): ${message}; retrying every ${retryMs} ms`);
      this.set({ error: { code, message } });
      l.retry = this.later(retryMs, () => {
        l.retry = null;
        void this.reconcileListening(l);
      });
      return false;
    }
  }

  // listen/stop; on failure only logs and takes `listening` from GET /v1/status.
  private async sendListenStop(l: ListenLink, current: () => boolean): Promise<boolean> {
    try {
      const st = await l.agent.call<{ listening?: boolean }>("POST", "/v1/listen/stop");
      if (current()) {
        l.maybeOn = false;
        this.set({ listening: st?.listening ?? false });
      }
      return true;
    } catch (e) {
      this.log.warn(`listen/stop failed: ${(e as Error).message}`);
      if (!current()) return false;
      try {
        const st = await l.agent.call<{ listening?: boolean }>("GET", "/v1/status");
        if (current()) {
          if (st?.listening === false) l.maybeOn = false;
          this.set({ listening: st?.listening ?? this.s.listening });
        }
      } catch {}
      return false; // no tight loop against a failing agent; the next setListening() retries
    }
  }

  // Forwards the agent's events; if the stream drops while the process lives,
  // reconnects and emits a "snapshot" event with GET /v1/status.
  private subscribe(gen: number, agent: AgentProcess, signal: AbortSignal, resync: boolean) {
    void (async () => {
      try {
        const stream = await agent.openEvents(signal);
        if (resync) {
          // connected again: the events missed meanwhile are summed up by the status
          const status = await agent.call("GET", "/v1/status");
          this.emit({ id: 0, type: "snapshot", data: status });
        }
        for await (const ev of stream) this.emit(ev);
      } catch (e) {
        if (signal.aborted) return;
        this.log.warn(`event stream error: ${(e as Error).message}`);
      }
      if (signal.aborted || gen !== this.gen) return;
      this.later(1000, () => {
        if (gen === this.gen && !signal.aborted && agent.child.exitCode === null) this.subscribe(gen, agent, signal, true);
      });
    })();
  }
}
