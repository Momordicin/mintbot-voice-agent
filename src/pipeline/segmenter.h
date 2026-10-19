// Cuts VAD-delimited utterances out of a bounded audio buffer.
//
// Follows FunASR's runtime/onnxruntime/src/audio.cpp Audio::Split(VadModel*,
// int, bool, ASR_TYPE) in ASR_OFFLINE mode (MIT): positions are absolute
// sample indices with an `offset` for what has been erased, every event of a
// call is handled before anything is erased, and the buffer keeps 2 s of
// history when idle or 2 s before the open segment's start while in speech.
#pragma once

#include "vad/fsmn_vad_stream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mintbot::pipeline {

struct Utterance {
    int64_t start_sample = 0;  // absolute, as open_start() reported it while the segment was open
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    std::vector<float> pcm;  // 16 kHz mono
};

class Segmenter {
public:
    explicit Segmenter(int keep_ms = 2000) : keep_(int64_t(keep_ms) * 16) {}

    void push_audio(const float* pcm, size_t n);
    // Apply VAD events (absolute ms) for audio already pushed; returns the
    // utterances they close, then trims the buffer.
    std::vector<Utterance> on_events(const std::vector<vad::VadEvent>& events);

    size_t buffered_samples() const { return buf_.size(); }
    // Absolute sample index where the open segment starts, -1 when none is open.
    int64_t open_start() const { return speech_start_; }
    // Absolute sample index one past the last sample pushed.
    int64_t end_sample() const { return offset_ + (int64_t)buf_.size(); }
    // Copies n samples starting at absolute index `from`; false if they are not all buffered.
    bool copy(int64_t from, size_t n, float* out) const;
    // End events that arrived without a start (upstream logs these and uses 0).
    int64_t anomalies() const { return anomalies_; }

private:
    void trim();

    std::vector<float> buf_;
    int64_t offset_ = 0;          // absolute sample index of buf_[0]
    int64_t speech_start_ = -1;   // absolute sample index, -1 when no open segment
    int64_t keep_;
    int64_t anomalies_ = 0;
};

}  // namespace mintbot::pipeline
