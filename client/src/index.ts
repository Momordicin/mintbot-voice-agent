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
//   ...
//   await voice.stop();
export { API_VERSION, AgentHttpError } from "./protocol.ts";
export type {
  AgentEvent,
  CaptureErrorCode,
  EnrollProgressEvent,
  FinalEvent,
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
export { VoiceAgentSupervisor } from "./supervisor.ts";
export type { Logger, SupervisorOptions, SupervisorPhase, SupervisorState } from "./supervisor.ts";
