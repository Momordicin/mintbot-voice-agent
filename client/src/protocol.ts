// Wire types of the mintbot-voice-agent REST / SSE API (app/voice_agent.cpp lists the endpoints).

// The API level this client speaks; the agent reports its own in the ready line,
// /v1/health and `--version`.
export const API_VERSION = 1;

export interface AgentEvent {
  id: number;
  // status | speech_start | speech_end | final | speech_other | device | capture | error |
  // enroll_progress | speaker, plus "snapshot" from the supervisor (see VoiceAgentSupervisor)
  type: string;
  data: any;
}

export interface ReadyLine {
  type: "ready";
  port: number;
  pid: number;
  version: string;
  api_version: number;
}

// Speaker fields of `final` and `speech_other` (present when the agent runs with --spk-model).
export interface SpeakerFields {
  speaker_enrolled?: boolean; // a voiceprint exists; false = everything is transcribed
  speaker_score?: number; // cosine to the owner voiceprint (0 when not enrolled)
  is_owner?: boolean; // speaker_score >= threshold (true when not enrolled)
  speaker_uncertain?: boolean; // shorter than --speaker-min-ms
  speaker_ms?: number; // time spent scoring when the utterance ended
  speaker_windows?: number; // 1.5 s windows scored
  speaker_reused?: number; // of which computed while the speech was going on
}

// Timing fields of `final` and `speech_other`.
export interface TimingFields {
  latency_ms: number; // segment closed -> event published
  queue_ms: number;
  detect_lag_ms: number; // segment end -> segment closed by the VAD
  total_ms: number; // segment end -> event published
  early: boolean; // the result was computed before the segment closed
}

// `final` event: one transcribed utterance (the owner's once a voiceprint exists).
export interface FinalEvent extends SpeakerFields, TimingFields {
  start_ms: number; // stream time
  end_ms: number;
  end_unix_ms: number;
  text: string;
  language: string; // e.g. "zh"
  emotion: string; // e.g. "NEUTRAL"
  event: string; // e.g. "Speech"
  raw: string; // with <|...|> tags
  asr_ms: number;
  // spoken during a running enrollment (it went into the voiceprint); only with --spk-model
  enrolling?: boolean;
}

// `speech_other` event: an utterance that was not transcribed (not the owner).
export interface SpeechOtherEvent extends SpeakerFields, TimingFields {
  start_ms: number;
  end_ms: number;
  end_unix_ms: number;
}

// `enroll_progress` event, and the answer of POST /v1/speaker/enroll/start
// (which also carries `active`).
export interface EnrollProgressEvent {
  active?: boolean;
  utterances: number;
  speech_ms: number;
  ready: boolean; // enough material for POST /v1/speaker/enroll/commit
}

// GET /v1/speaker, the answers of enroll/commit, enroll/cancel and DELETE /v1/speaker,
// and the `speaker` event (after commit or forget). 404 without --spk-model.
export interface SpeakerStatus {
  enrolled: boolean;
  utterances: number; // in the voiceprint
  speech_ms: number;
  threshold: number;
  min_reliable_ms: number;
  gate_uncertain: boolean;
  uncertain_floor: number;
  enrolling: boolean;
  enroll_utterances: number; // collected by the running enrollment
  enroll_speech_ms: number;
  enroll_ready: boolean;
  persistent: boolean; // --speaker-db given: the voiceprint is kept in a file
}

// Error codes of POST /v1/listen/start and of `capture` events.
export type CaptureErrorCode = "mic_access_denied" | "capture_failed" | "";

export class AgentHttpError extends Error {
  readonly status: number;
  readonly body: any;
  constructor(method: string, path: string, status: number, body: any) {
    super(`${method} ${path}: ${status} ${typeof body === "string" ? body : JSON.stringify(body)}`);
    this.name = "AgentHttpError";
    this.status = status;
    this.body = body;
  }
  // `code` of the error body, e.g. "mic_access_denied"
  get code(): string | undefined {
    return typeof this.body?.code === "string" ? this.body.code : undefined;
  }
}
