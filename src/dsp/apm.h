// Capture-side audio preprocessing with WebRTC AudioProcessing:
// [echo cancellation (AEC3)] -> high-pass filter -> noise suppression -> AGC2
// (adaptive digital), at 16 kHz mono. AEC3 is off unless asked for; it needs the
// render reference (what the speakers play) through process_reverse().
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mintbot::dsp {

struct ApmOptions {
    bool enabled   = false;
    bool high_pass = true;
    int  ns_level  = 2;     // 0 off, 1 low, 2 moderate, 3 high, 4 very high
    bool agc2      = true;  // adaptive digital gain (has its own speech detector and noise-floor cap)
    bool aec       = false; // AEC3; feed the render reference with process_reverse()
    bool aec_linear_output = false;  // also keep AEC3's linear-filter output (before its suppressor)
};

// AEC3 statistics, empty fields until it has an estimate.
struct AecStats {
    std::optional<double> echo_return_loss;              // dB
    std::optional<double> echo_return_loss_enhancement;  // dB
    std::optional<double> divergent_filter_fraction;
    std::optional<double> residual_echo_likelihood;
    std::optional<int32_t> delay_ms;                     // AEC3's own render-to-capture delay estimate
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
    // With aec_linear_output, `linear` (if given) receives AEC3's linear-filter
    // output for the same frames.
    void process(const float* in, size_t n, std::vector<float>& out, std::vector<float>* linear = nullptr);

    // AEC3 render reference: 16 kHz mono float PCM of what the speakers play, in
    // order, any length (10 ms frames are passed on as they complete). Give it
    // the render audio before the capture audio that contains its echo.
    void process_reverse(const float* in, size_t n);
    // Optional hint for AEC3's initial alignment (it estimates the delay itself).
    void set_stream_delay_ms(int ms);
    AecStats aec_stats();

    bool active() const;
    static bool available();  // false when built with MINTBOT_WITH_APM=OFF

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::dsp
