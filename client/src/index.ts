// @mintbot/voice-agent-client: install, start, supervise and talk to mintbot-voice-agent.
//
//   const inst = await readInstallation(root, { apiVersion: API_VERSION })
//             ?? await installVoiceAgent({ manifest, root, apiVersion: API_VERSION });
//   const voice = new VoiceAgentSupervisor({
//     installation: inst, voiceprintPath, speakerThreshold: 0.5,
//     onEvent: (ev) => { if (ev.type === "final") record(ev.data.text); },
//     onState: (s) => { /* phase, listenWanted, listening, error.code === "mic_access_denied", ... */ },
//   });
//   await voice.start();
//   voice.setListening(false); // the microphone is released and stays off across restarts until setListening(true)
//   const id = await voice.play(wav); // PCM 16-bit mono WAV, after the clips queued before it; `playback` events by id
//   await voice.stopPlayback();
//   ...
//   await voice.stop();
export { API_VERSION, AgentHttpError } from "./protocol.ts";
export type {
  AgentEvent,
  CaptureErrorCode,
  CommandEvent,
  DeviceEvent,
  DeviceIdentity,
  EnrollFailedEvent,
  EnrollProgressEvent,
  FinalEvent,
  HotwordReplacement,
  HotwordsReloadError,
  HotwordsReloadReply,
  HotwordsStatus,
  PlaybackEvent,
  PlaybackStatus,
  ReadyLine,
  SpeakerFields,
  SpeakerStatus,
  SpeechOtherEvent,
  TimingFields,
} from "./protocol.ts";
export { startAgentProcess } from "./agent.ts";
export type { AgentProcess, AgentProcessOptions } from "./agent.ts";
export { installVoiceAgent, readInstallation, staleVersions, VoiceInstallError } from "./install.ts";
export type {
  InstallErrorCode,
  Installation,
  InstallOptions,
  InstallProgress,
  Manifest,
  PackageInfo,
  ReadInstallationOptions,
} from "./install.ts";
export { PlaybackError, VoiceAgentSupervisor } from "./supervisor.ts";
export type { Logger, PlaybackErrorCode, SupervisorOptions, SupervisorPhase, SupervisorState } from "./supervisor.ts";
export { createTtsManager, TtsError } from "./tts.ts";
export type {
  TtsErrorCode,
  TtsInstallStatus,
  TtsInstallTarget,
  TtsLastDownload,
  TtsManager,
  TtsManagerOptions,
  TtsModel,
  TtsPackageInfo,
  TtsServiceStatus,
  TtsStatus,
  TtsSynthesisRequest,
  TtsVariant,
} from "./tts.ts";
