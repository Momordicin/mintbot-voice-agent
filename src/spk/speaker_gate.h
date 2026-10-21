// Owner-speaker gate on top of CAM++: enrollment, voiceprint file, scoring.
//
// Scoring follows FunASR's streaming speaker assignment (funasr/bin/realtime_ws.py):
// sv_chunk windows -> L2-normalised embeddings -> cosine against the centre.
// Thread-safety: analyze() runs on one worker thread (it owns the network);
// the enrollment / status calls may come from any thread.
#pragma once

#include "spk/campplus.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mintbot::spk {

struct SpeakerGateOptions {
    std::string db_path;          // voiceprint file chosen by the core; empty = in memory only
    // cosine; calibrated on the first owner's recordings (owner >= 0.50, others <= 0.42),
    // the core passes each user's own value
    float threshold = 0.5f;
    int min_reliable_ms = 1000;   // shorter utterances are flagged uncertain
    bool gate_uncertain = false;  // false: uncertain (short) utterances are transcribed anyway ...
    float uncertain_floor = 0.3f; // ... unless they score below this (owner's short ones >= 0.59, others' 0.08-0.15)
    int min_enroll_utterances = 3;
    int min_enroll_ms = 10000;    // speech needed before commit
    int n_threads = 4;            // measured: 1 -> 598 ms, 2 -> 370, 4 -> 275, 8 -> 266 for 7 windows
};

struct SpeakerDecision {
    bool enrolled = false;   // no voiceprint -> gate open
    float score = 0.0f;      // cosine to the owner centre
    float threshold = 0.0f;  // the threshold applied (lowered on a device other than the enrollment's)
    bool uncertain = false;  // too short to judge reliably
    bool is_owner = true;    // score >= threshold (always true when not enrolled)
    bool transcribe = true;  // what the agent should do with the utterance
    bool enrolling = false;  // the utterance was added to a running enrollment
    double embed_ms = 0;     // time spent here; windows taken from the cache cost nothing
    int windows = 0;         // sv_chunk windows of the utterance
    int reused = 0;          // of which computed ahead of time while the speech was still going on
};

struct EnrollState {
    bool active = false;
    int utterances = 0;
    int64_t speech_ms = 0;
    bool ready = false;  // enough material to commit
};

class SpeakerGate {
public:
    bool load(const std::string& campplus_gguf, const SpeakerGateOptions& opts, std::string* error);

    // `pre`: full windows of this utterance already embedded with embed_window().
    SpeakerDecision analyze(const float* pcm, size_t n, const WindowCache* pre = nullptr);

    // Whether analyze() will embed at all (enrolled or enrolling); decides if
    // computing windows ahead of time is worth it.
    bool wants_windows();
    // A decision made earlier may be reused only while neither of these changed:
    // no enrollment running, and the same voiceprint (version bumps on commit / forget).
    bool enrolling();
    uint64_t version();
    // Raw embedding of one full kSvWindow window, for the cache passed to analyze().
    bool embed_window(const float* pcm, std::vector<float>& emb) { return cam_.embed_window(pcm, kSvWindow, emb); }
    void set_abort_callback(ggml_abort_callback cb, void* data) { cam_.set_abort_callback(cb, data); }

    // The capture device's name (DeviceIdentity::display_name), set when capture
    // starts and when the device changes; empty for sources without a device. On
    // a device other than the one the voiceprint was enrolled on, the threshold is
    // halved (not below 0.2) and the short-utterance floor is the mean of the
    // usual floor and that threshold. Returns false if this stopped a running
    // enrollment: an enrollment must come from one device.
    bool set_device(const std::string& name);

    EnrollState enroll_begin();
    EnrollState enroll_state();
    bool enroll_commit(std::string* error);
    void enroll_cancel();
    bool forget(std::string* error);  // deletes the voiceprint file

    std::string status_json();

private:
    bool save_locked(std::string* error);
    // the threshold and short-utterance floor for the current device
    void limits_locked(float& threshold, float& floor) const;

    CamPlus cam_;
    SpeakerGateOptions opts_;
    std::mutex m_;
    std::vector<float> center_;  // empty = not enrolled
    int center_utterances_ = 0;
    int64_t center_ms_ = 0;
    std::string center_device_;  // device the voiceprint was enrolled on; empty: unknown
    std::string device_;         // current capture device
    bool enrolling_ = false;
    uint64_t version_ = 0;
    std::vector<std::vector<float>> enroll_windows_;
    int enroll_utts_ = 0;
    int64_t enroll_ms_ = 0;
    std::string enroll_device_;
    bool enroll_device_changed_ = false;  // the last enrollment was stopped by a device change
};

}  // namespace mintbot::spk
