// Capture-side audio preprocessing with WebRTC AudioProcessing:
// high-pass filter -> noise suppression -> AGC2 (adaptive digital), at 16 kHz mono.
// Echo cancellation is not enabled yet (it needs the render reference).
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mintbot::dsp {

struct ApmOptions {
    bool enabled   = false;
    bool high_pass = true;
    int  ns_level  = 2;     // 0 off, 1 low, 2 moderate, 3 high, 4 very high
    bool agc2      = true;  // adaptive digital gain (has its own speech detector and noise-floor cap)
};

class AudioPreprocessor {
public:
    AudioPreprocessor();
    ~AudioPreprocessor();
    AudioPreprocessor(AudioPreprocessor&&) noexcept;
    AudioPreprocessor& operator=(AudioPreprocessor&&) noexcept;

    bool init(const ApmOptions& opts, std::string* error = nullptr);

    // Processes 16 kHz mono float PCM in [-1, 1] of any length. APM works on
    // 10 ms frames, so up to 159 samples are held back until the next call;
    // `out` receives only complete processed frames. Passthrough when disabled.
    void process(const float* in, size_t n, std::vector<float>& out);

    bool active() const;
    static bool available();  // false when built with MINTBOT_WITH_APM=OFF

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::dsp
