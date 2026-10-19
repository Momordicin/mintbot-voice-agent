#include "pipeline/segmenter.h"

#include <algorithm>

namespace mintbot::pipeline {

void Segmenter::push_audio(const float* pcm, size_t n) { buf_.insert(buf_.end(), pcm, pcm + n); }

std::vector<Utterance> Segmenter::on_events(const std::vector<vad::VadEvent>& events) {
    std::vector<Utterance> out;
    const int64_t buf_end = offset_ + (int64_t)buf_.size();
    for (const auto& e : events) {
        int64_t pos = std::clamp<int64_t>(e.ms * 16, offset_, buf_end);
        if (e.type == vad::VadEvent::SpeechStart) {
            speech_start_ = pos;
            continue;
        }
        int64_t start = speech_start_;
        if (start < 0) {  // upstream: "Vad start is null while vad end is available. Set vad start 0"
            ++anomalies_;
            start = offset_;
        }
        Utterance u;
        u.start_sample = start;
        u.start_ms = start / 16;
        u.end_ms = e.ms;
        u.pcm.assign(buf_.begin() + (start - offset_), buf_.begin() + (pos - offset_));
        out.push_back(std::move(u));
        speech_start_ = -1;
    }
    trim();
    return out;
}

bool Segmenter::copy(int64_t from, size_t n, float* out) const {
    if (from < offset_ || from + (int64_t)n > end_sample()) return false;
    std::copy(buf_.begin() + (from - offset_), buf_.begin() + (from - offset_ + (int64_t)n), out);
    return true;
}

void Segmenter::trim() {
    int64_t erase = 0;
    if (speech_start_ < 0)
        erase = (int64_t)buf_.size() - keep_;
    else
        erase = speech_start_ - offset_ - keep_;
    if (erase <= 0) return;
    buf_.erase(buf_.begin(), buf_.begin() + erase);
    offset_ += erase;
}

}  // namespace mintbot::pipeline
