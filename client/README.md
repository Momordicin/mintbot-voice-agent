# @mintbot/voice-agent-client

Installs a mintbot-voice-agent release into an application's
directory and keeps the agent running from a Node.js (≥ 22) core on Windows x64:
download with progress, verification, cancellation and repair
(`installVoiceAgent`, `readInstallation`), and a supervisor that starts the
agent, restarts it after a crash, follows its events and switches the
microphone on and off (`VoiceAgentSupervisor`). No runtime dependencies.

```ts
import { API_VERSION, installVoiceAgent, readInstallation, VoiceAgentSupervisor } from "@mintbot/voice-agent-client";

const inst = (await readInstallation(root, { apiVersion: API_VERSION }))
          ?? (await installVoiceAgent({ manifest, root, apiVersion: API_VERSION, onStep, signal }));
const voice = new VoiceAgentSupervisor({ installation: inst, voiceprintPath, onEvent, onState });
await voice.start();
voice.setListening(false); // releases the microphone until setListening(true)
await voice.stop();
```

Options, progress phases, error codes (`InstallErrorCode`) and event payloads
(`FinalEvent`, `SpeechOtherEvent`, `SpeakerStatus`, ...) are documented in the
comments of the TypeScript sources shipped in `src/`; with the declaration
maps, "go to definition" in an editor opens them.

License: MIT (`LICENSE`).
