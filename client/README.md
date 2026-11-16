# @mintbot/voice-agent-client

Installs a mintbot-voice-agent release into an application's
directory and keeps the agent running from a Node.js (≥ 22) core on Windows x64:
download with progress, verification, cancellation and repair
(`installVoiceAgent`, `readInstallation`), and a supervisor that starts the
agent, restarts it after a crash, follows its events and switches the
microphone on and off (`VoiceAgentSupervisor`); and the GPT-SoVITS
speech-synthesis component (`createTtsManager`). One runtime dependency: `7zip-bin`.

```ts
import { API_VERSION, installVoiceAgent, readInstallation, VoiceAgentSupervisor } from "@mintbot/voice-agent-client";

const inst = (await readInstallation(root, { apiVersion: API_VERSION }))
          ?? (await installVoiceAgent({ manifest, root, apiVersion: API_VERSION, onStep, signal }));
const voice = new VoiceAgentSupervisor({ installation: inst, voiceprintPath, onEvent, onState });
await voice.start();
voice.setListening(false); // releases the microphone until setListening(true)
const id = await voice.play(wav); // a PCM 16-bit mono WAV, after the clips before it; `playback` events by id
await voice.stopPlayback();       // stops it and drops the queued clips
await voice.stop();
```

```ts
import { createTtsManager } from "@mintbot/voice-agent-client";

const tts = createTtsManager({ root, manifest, onState });
tts.install({ variant: "v2pro", root: ttsRoot }); // progress in tts.status()
await tts.useExisting(dir); // a GPT-SoVITS folder the user already has
await tts.checkUpdate({ signal });
await tts.startService({ gptWeights, sovitsWeights }); // outcome in tts.status().service
const wav = await tts.synthesize({ text, textLang: "zh", refAudio, refText, promptLang: "zh" });
await tts.dispose();
```

Options, progress phases, error codes (`InstallErrorCode`, `TtsErrorCode`, `PlaybackErrorCode`) and event payloads
(`FinalEvent`, `CommandEvent`, `SpeechOtherEvent`, `SpeakerStatus`, `HotwordsStatus`, `PlaybackEvent`, ...) are documented in the
comments of the TypeScript sources shipped in `src/`; with the declaration
maps, "go to definition" in an editor opens them.

License: MIT (`LICENSE`).
