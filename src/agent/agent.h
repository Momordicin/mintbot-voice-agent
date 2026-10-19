// Always-on voice agent engine.
//
// Threads: the audio source's callback thread only queues PCM; a front thread
// runs FSMN-VAD and the segmenter over whatever has been queued (the VAD is
// chunk-size invariant) and restarts the source when it stops or goes silent;
// an ASR thread owns SenseVoice. Every state change is published to the
// EventHub as a JSON event.
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
    double replay_speed = 1.0;      // wav source only
    int max_queue_ms = 10000;       // capture backlog cap; the oldest audio is dropped beyond it
    int watchdog_ms = 3000;         // restart the source after this long without audio
    int restart_backoff_ms = 2000;  // delay between restart attempts
    std::string record_path;        // if set, raw (pre-APM) input is written here as a 16 kHz WAV
    std::string speaker_model;      // CAM++ GGUF; empty disables the speaker gate
    spk::SpeakerGateOptions speaker;
    // Start speaker scoring + ASR once this much silence follows speech, on the
    // exact samples the segment will have if the silence goes on; the result is
    // published when the VAD closes the segment there. 0 disables it.
    int early_asr_ms = 200;
    // Audio is given to the VAD in blocks of this length instead of every 10 ms
    // capture period; its output does not depend on the block size.
    int vad_block_ms = 50;
};

class Agent {
public:
    explicit Agent(EventHub& hub);
    ~Agent();
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    bool load(const std::string& vad_gguf, const std::string& asr_gguf, const AgentOptions& opts, std::string* error);

    // device: capture device index, -1 for the system default. Ignored for wav sources.
    bool start_listening(int device, std::string* error);
    // Stops the source and finishes the open utterance, if any.
    void stop_listening();

    std::string status_json();
    // nullptr when the speaker gate is disabled
    spk::SpeakerGate* speaker();
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::agent
