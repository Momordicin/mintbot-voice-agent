// [APM ->] VAD -> segmenter -> SenseVoice, driven synchronously by feed().
#pragma once

#include "asr/sensevoice.h"
#include "dsp/apm.h"
#include "pipeline/segmenter.h"
#include "vad/fsmn_vad_stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mintbot::pipeline {

struct PipelineOptions {
    dsp::ApmOptions apm;  // off unless enabled
    vad::FsmnVadOptions vad;
    asr::SenseVoiceOptions asr;
    int keep_ms = 2000;
};

struct Result {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    asr::Transcript transcript;
    double asr_ms = 0;  // time spent in SenseVoice for this utterance
};

class VoicePipeline {
public:
    bool load(const std::string& vad_gguf, const std::string& asr_gguf, const PipelineOptions& opts = {});

    // 16 kHz mono float PCM in [-1, 1], any size.
    std::vector<Result> feed(const float* pcm, size_t n);
    std::vector<Result> flush();

    const Segmenter& segmenter() const { return seg_; }

private:
    std::vector<Result> transcribe(std::vector<Utterance> utts);

    dsp::AudioPreprocessor apm_;
    std::vector<float> buf_;
    vad::FsmnVadStream vad_;
    Segmenter seg_;
    asr::SenseVoice asr_;
};

}  // namespace mintbot::pipeline
