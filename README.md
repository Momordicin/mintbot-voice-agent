# mintbot-voice-agent

An always-on, local voice input plugin for desktop apps on Windows. It listens
to the microphone, keeps only the voice of the enrolled owner, and turns what
they say into text, entirely on the user's machine: no cloud service, no
PyTorch or ONNX at runtime.

It is made of two parts:

- **`mintbot-voice-agent.exe`**, a background process: noise suppression and
  gain control (WebRTC), voice activity detection (FSMN-VAD), owner voiceprint
  check (CAM++) and speech recognition (SenseVoiceSmall: Chinese, English,
  Cantonese, Japanese, Korean, with emotion and audio-event tags), all on ggml.
  Results arrive about 0.6 s after the speaker stops. The app talks to it over
  loopback HTTP (REST commands, Server-Sent Events).
- **`@mintbot/voice-agent-client`**, a Node.js package for the app: it
  downloads and installs a release (about 280 MB, models included), starts the
  process, restarts it if it crashes, forwards its events and turns the
  microphone on and off.

It was built as the voice input of [MintBot](https://github.com/Momordicin/MintBot).

## Requirements

- Windows 10 1803 or later, x64
- Node.js 22 or later in the host app
- About 600 MB free disk space while installing (about 300 MB afterwards)

## Quick start

Add the client from a release (replace `<owner>/<repo>` and the versions with
those of the release you use):

```powershell
pnpm add https://github.com/<owner>/<repo>/releases/download/v0.1.1/mintbot-voice-agent-client-0.3.0.tgz
```

Install the agent once, then keep it running:

```ts
import path from "node:path";
import { API_VERSION, installVoiceAgent, readInstallation, VoiceAgentSupervisor } from "@mintbot/voice-agent-client";

const manifest = "https://github.com/<owner>/<repo>/releases/download/v0.1.1/manifest.json";
const root = path.join(process.env.LOCALAPPDATA!, "MyApp", "voice-agent");      // where the agent is installed
const voiceprint = path.join(process.env.APPDATA!, "MyApp", "voiceprint.json"); // user data: keep it outside root

const installation =
  (await readInstallation(root, { apiVersion: API_VERSION })) ??
  (await installVoiceAgent({ manifest, root, apiVersion: API_VERSION,
                             onStep: (p) => console.log(p.package, p.phase, p.bytes ?? "") }));

const voice = new VoiceAgentSupervisor({
  installation,
  voiceprintPath: voiceprint,
  onEvent: (ev) => {
    if (ev.type === "final") console.log("heard:", ev.data.text);
  },
  onState: (s) => console.log("voice agent:", s.phase, s.listening ? "listening" : "", s.error?.code ?? ""),
});
await voice.start();      // starts listening right away

voice.setListening(false); // releases the microphone (stays off across restarts)
voice.setListening(true);

await voice.stop();       // on app exit; the agent also exits by itself if the app dies
```

Until an owner voiceprint exists, everything that is said is transcribed. To
enroll, call `voice.call("POST", "/v1/speaker/enroll/start")`, let the user
speak five or more normal sentences (about 30 seconds), and once an
`enroll_progress` event reports `ready: true` call
`voice.call("POST", "/v1/speaker/enroll/commit")`. From then on, only the
owner's speech is transcribed; `voice.call("DELETE", "/v1/speaker")` removes the
voiceprint again.

The options, events and error codes are documented in the comments of the
client's TypeScript sources (`client/src/`, also shipped in the package; "go to
definition" in an editor opens them).

## Privacy

Audio is processed in memory and never written to disk. The text results go
only to the host app, and the voiceprint (biometric data) is stored only in the
file the app chooses.

## Building from source

With Visual Studio 2022 (C++), CMake, Ninja, Python 3.12 and Node.js 22:

```powershell
pip install -r tools\requirements-release.txt --extra-index-url https://download.pytorch.org/whl/cpu
python tools\build_release.py --out dist
```

This builds the program, re-creates the models from the official ModelScope
weights (checked byte for byte against `tools\models.lock.json`), packages a
release into `dist\<version>\` and tests it. Pushing a tag `v<version>` runs the
same on GitHub Actions and publishes the release.

## License

MIT, see [LICENSE](LICENSE).

The program includes or is derived from third-party code under MIT, BSD-3-Clause
and Apache-2.0 licenses (FunASR, SenseVoice, 3D-Speaker, ggml, WebRTC
AudioProcessing, abseil, miniaudio, cpp-httplib); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model weights in the
release come from Tongyi Lab / Alibaba Group under the FunASR Model Open Source
License Agreement v1.1 and Apache-2.0; their licenses and attribution are
included in the model package.
