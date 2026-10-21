// Always-on voice agent engine: capture -> VAD -> speaker gate -> ASR, published as JSON events.
#pragma once

#include "agent/event_hub.h"
#include "pipeline/pipeline.h"
#include "spk/speaker_gate.h"

#include <memory>
#include <string>

namespace mintbot::agent {

struct AgentOptions {
    pipeline::PipelineOptions pipe;
    std::string source = "mic";     // "mic" or "wav:<path>"
    double replay_speed = 1.0;
    int max_queue_ms = 10000;       // the oldest audio is dropped beyond this backlog
    int watchdog_ms = 3000;         // restart the source after this long without audio
    int restart_backoff_ms = 2000;
    std::string record_path;        // raw (pre-APM) input as a 16 kHz WAV
    std::string speaker_model;      // empty: no speaker gate
    spk::SpeakerGateOptions speaker;
    int early_asr_ms = 200;         // run scoring + ASR after this much silence; 0 = off
    int vad_block_ms = 50;
    bool speaker_observe = false;   // testing: transcribe every utterance, only report the decision
};

class Agent {
public:
    explicit Agent(EventHub& hub);
    ~Agent();
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    bool load(const std::string& vad_gguf, const std::string& asr_gguf, const AgentOptions& opts, std::string* error);

    bool start_listening(int device, std::string* error);  // device -1: system default
    void stop_listening();  // finishes the open utterance

    std::string status_json();
    spk::SpeakerGate* speaker();  // nullptr without a speaker gate
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::agent
