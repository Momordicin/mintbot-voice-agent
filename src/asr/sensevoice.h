// SenseVoiceSmall (SAN-M encoder + CTC head) on ggml.
//
// Ported from the SenseVoice repository's validated ggml runtime
// (runtime/llama.cpp/funasr-sensevoice/funasr-sensevoice.cpp and
// sensevoice-server.cpp, MIT, Copyright (c) 2025 FunASR): same fbank / LFR
// front end, encoder graph and greedy CTC decode. Language and ITN are chosen
// per instance through the four query tokens instead of being fixed at export.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <memory>
#include <string>

namespace mintbot::asr {

struct SenseVoiceOptions {
    int         n_threads = 4;
    std::string language  = "auto";  // auto, zh, en, yue, ja, ko, nospeech
    bool        itn       = true;    // punctuation, casing and digits
};

struct Transcript {
    std::string text;      // tags removed
    std::string language;  // e.g. "zh"; empty if the model emitted none
    std::string emotion;   // e.g. "NEUTRAL", "HAPPY"
    std::string event;     // e.g. "Speech", "BGM", "Laughter"
    std::string raw;       // decoded text with <|...|> tags kept
};

class SenseVoice {
public:
    SenseVoice();
    ~SenseVoice();
    SenseVoice(const SenseVoice&) = delete;
    SenseVoice& operator=(const SenseVoice&) = delete;

    bool load(const std::string& gguf_path, const SenseVoiceOptions& opts = {});

    // 16 kHz mono float PCM in [-1, 1]; one utterance (the model is trained on
    // segments up to ~30 s). Not thread-safe: use one instance per thread.
    Transcript transcribe(const float* pcm, size_t n);

    // Polled during transcribe(); returning true abandons it (the result is then empty).
    void set_abort_callback(ggml_abort_callback cb, void* data);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::asr
