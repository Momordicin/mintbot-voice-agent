#include "pipeline/pipeline.h"

#include <chrono>
#include <cstdio>

namespace mintbot::pipeline {

bool VoicePipeline::load(const std::string& vad_gguf, const std::string& asr_gguf, const PipelineOptions& opts) {
    seg_ = Segmenter(opts.keep_ms);
    std::string err;
    if (!apm_.init(opts.apm, &err)) { fprintf(stderr, "pipeline: %s\n", err.c_str()); return false; }
    return vad_.load(vad_gguf, opts.vad) && asr_.load(asr_gguf, opts.asr);
}

std::vector<Result> VoicePipeline::feed(const float* pcm, size_t n) {
    buf_.clear();
    apm_.process(pcm, n, buf_);
    if (buf_.empty()) return {};
    seg_.push_audio(buf_.data(), buf_.size());
    return transcribe(seg_.on_events(vad_.feed(buf_.data(), buf_.size())));
}

std::vector<Result> VoicePipeline::flush() { return transcribe(seg_.on_events(vad_.flush())); }

std::vector<Result> VoicePipeline::transcribe(std::vector<Utterance> utts) {
    std::vector<Result> out;
    for (auto& u : utts) {
        auto t0 = std::chrono::steady_clock::now();
        Result r;
        r.start_ms = u.start_ms;
        r.end_ms = u.end_ms;
        r.transcript = asr_.transcribe(u.pcm.data(), u.pcm.size());
        r.asr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace mintbot::pipeline
