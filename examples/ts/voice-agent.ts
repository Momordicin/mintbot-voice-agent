// Spawning and talking to mintbot-voice-agent from a TypeScript (Node >= 22) core,
// on the client package (client/); a real core uses its VoiceAgentSupervisor.
//
//   node --experimental-strip-types examples/ts/voice-agent.ts --exe build/bin/mintbot-voice-agent.exe \
//        --vad models/fsmn-vad.gguf --model models/sensevoice-small-q8_0.gguf [--source wav:<file>] \
//        [--quit-on-ended] [--exit-after-ready-ms N]
import type { ChildProcess } from "node:child_process";
import { pathToFileURL } from "node:url";

import { startAgentProcess, type AgentEvent } from "../../client/src/index.ts";

export type { AgentEvent };

export interface VoiceAgent {
  child: ChildProcess;
  port: number;
  call(method: "GET" | "POST", path: string): Promise<any>;
  events(signal?: AbortSignal): AsyncGenerator<AgentEvent>;
  shutdown(): Promise<void>;
}

export async function startVoiceAgent(opts: { exe: string; vad: string; model: string; args?: string[] }): Promise<VoiceAgent> {
  const a = await startAgentProcess({
    file: opts.exe,
    args: ["--vad", opts.vad, "-m", opts.model, ...(opts.args ?? [])],
    onStderr: (line) => process.stderr.write(line + "\n"),
  });
  return {
    child: a.child,
    port: a.ready.port,
    call: (method, path) => a.call(method, path),
    events: (signal) => a.events(signal),
    shutdown: () => a.shutdown(),
  };
}

async function main() {
  const arg = (name: string) => {
    const i = process.argv.indexOf(name);
    return i >= 0 ? process.argv[i + 1] : undefined;
  };
  const source = arg("--source");
  const agent = await startVoiceAgent({
    exe: arg("--exe")!,
    vad: arg("--vad")!,
    model: arg("--model")!,
    args: source ? ["--source", source] : [],
  });
  console.error(`[core] agent ready on port ${agent.port}, pid ${agent.child.pid}`);
  agent.child.on("exit", (code) => console.error(`[core] agent exited (${code})`));

  const crashAfter = arg("--exit-after-ready-ms");
  if (crashAfter) {
    // Simulates the core dying without any cleanup; the agent must exit on its own.
    setTimeout(() => process.exit(3), Number(crashAfter));
  }

  const abort = new AbortController();
  const listening = (async () => {
    for await (const ev of agent.events(abort.signal)) {
      console.log(JSON.stringify({ type: ev.type, ...ev.data }));
      if (ev.type === "device" && ev.data.state === "ended" && process.argv.includes("--quit-on-ended")) {
        // wait until the last utterance has been transcribed, then stop
        for (;;) {
          const st = await agent.call("GET", "/v1/status");
          if (!st.in_speech && st.asr_queue === 0 && !st.listening) break;
          await new Promise((r) => setTimeout(r, 200));
        }
        await new Promise((r) => setTimeout(r, 300));
        abort.abort();
        await agent.shutdown();
        return;
      }
    }
  })().catch((e) => {
    if (e.name !== "AbortError") throw e;
  });

  console.error("[core] devices:", JSON.stringify((await agent.call("GET", "/v1/devices")).devices));
  await agent.call("POST", "/v1/listen/start");
  await listening;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
