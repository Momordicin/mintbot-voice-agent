// Installing a release and running it under the supervisor, with the client package (client/).
//
//   node --experimental-strip-types examples/ts/install-voice-agent.ts --manifest <path|url> --root <dir> [--smoke <wav>]
import { pathToFileURL } from "node:url";

import { API_VERSION, installVoiceAgent, VoiceAgentSupervisor } from "../../client/src/index.ts";

async function main() {
  const arg = (name: string) => {
    const i = process.argv.indexOf(name);
    return i >= 0 ? process.argv[i + 1] : undefined;
  };
  const t0 = Date.now();
  const inst = await installVoiceAgent({
    manifest: arg("--manifest")!,
    root: arg("--root")!,
    apiVersion: Number(arg("--api-version") ?? API_VERSION),
    onProgress: (m) => console.error(`[install] ${m}`),
  });
  console.error(`[install] done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  console.log(JSON.stringify(inst, null, 1));

  const smoke = arg("--smoke"); // replay a wav through the installed agent
  if (!smoke) return;
  let ended = false;
  const sup = new VoiceAgentSupervisor({
    installation: inst,
    args: ["--source", "wav:" + smoke],
    onEvent: (ev) => {
      if (ev.type === "final") console.log(`[smoke] final ${ev.data.text}`);
      if (ev.type === "device" && ev.data.state === "ended") ended = true;
    },
    onState: (s) => console.error(`[smoke] ${s.phase}${s.listening ? " listening" : ""}${s.error ? " " + s.error.message : ""}`),
  });
  await sup.start();
  while (!ended) await new Promise((r) => setTimeout(r, 100));
  for (;;) {
    const st = await sup.call("GET", "/v1/status");
    if (st.asr_queue === 0 && st.capture === "stopped") break;
    await new Promise((r) => setTimeout(r, 200));
  }
  await sup.stop();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
