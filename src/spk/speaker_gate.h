// Owner-speaker gate on top of CAM++: enrolls the owner's voiceprint, keeps it in a
// file, and scores each utterance against it to decide whether it is transcribed.
// Scoring follows FunASR's streaming speaker assignment (funasr/bin/realtime_ws.py).
#pragma once

#include "spk/campplus.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mintbot::spk {

struct SpeakerGateOptions {
    std::string db_path;
    float threshold = 0.5f;
    int min_reliable_ms = 1000;
    bool gate_uncertain = false;
    float uncertain_floor = 0.3f;
    float cross_base = 0.153f;
    float cross_per_ln_s = 0.023f;
    float cross_min = 0.1f;
    int min_enroll_utterances = 3;
    int min_enroll_ms = 10000;
    int n_threads = 4;
};

struct SpeakerDecision {
    bool enrolled = false;
    float score = 0.0f;
    float threshold = 0.0f;
    bool same_device = true;
    bool uncertain = false;
    bool is_owner = true;
    bool transcribe = true;
    bool enrolling = false;
    double embed_ms = 0;
    int windows = 0;
    int reused = 0;
};

struct EnrollState {
    bool active = false;
    int utterances = 0;
    int64_t speech_ms = 0;
    bool ready = false;
};

class SpeakerGate {
public:
    bool load(const std::string& campplus_gguf, const SpeakerGateOptions& opts, std::string* error);

    SpeakerDecision analyze(const float* pcm, size_t n, const WindowCache* pre = nullptr);

    bool wants_windows();
    bool enrolling();
    uint64_t version();
    bool embed_window(const float* pcm, std::vector<float>& emb) { return cam_.embed_window(pcm, kSvWindow, emb); }
    void set_abort_callback(ggml_abort_callback cb, void* data) { cam_.set_abort_callback(cb, data); }

    bool set_device(const std::string& name);

    EnrollState enroll_begin();
    EnrollState enroll_state();
    bool enroll_commit(std::string* error);
    void enroll_cancel();
    bool forget(std::string* error);

    std::string status_json();

private:
    bool save_locked(std::string* error);
    bool same_device_locked() const;
    void limits_locked(int64_t ms, float& threshold, float& floor) const;

    CamPlus cam_;
    SpeakerGateOptions opts_;
    std::mutex m_;
    std::vector<float> center_;
    int center_utterances_ = 0;
    int64_t center_ms_ = 0;
    std::string center_device_;
    std::string device_;
    bool enrolling_ = false;
    uint64_t version_ = 0;
    std::vector<std::vector<float>> enroll_windows_;
    int enroll_utts_ = 0;
    int64_t enroll_ms_ = 0;
    std::string enroll_device_;
    bool enroll_device_changed_ = false;
};

}  // namespace mintbot::spk
