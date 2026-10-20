#include "dsp/apm.h"

#ifdef MINTBOT_HAVE_APM
#include "api/audio/audio_processing.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "api/scoped_refptr.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"

#include <array>
#endif

namespace mintbot::dsp {

constexpr size_t kFrame = 160;  // 10 ms at 16 kHz

#ifdef MINTBOT_HAVE_APM
// APM only allocates the buffer for the linear AEC output; AEC3 itself must be
// configured to produce it, which the built-in AEC3 setup does not do, and this
// WebRTC package has no EchoCanceller3Factory.
class LinearAec3Factory : public webrtc::EchoControlFactory {
public:
    std::unique_ptr<webrtc::EchoControl> Create(int rate, int render_channels, int capture_channels) override {
        webrtc::EchoCanceller3Config c;
        c.filter.export_linear_aec_output = true;
        return std::make_unique<webrtc::EchoCanceller3>(c, std::nullopt, rate, (size_t)render_channels,
                                                        (size_t)capture_channels);
    }
};
#endif

struct AudioPreprocessor::Impl {
    ApmOptions opts;
    std::vector<float> pending;
    std::vector<float> pending_reverse;
    int delay_ms = -1;
#ifdef MINTBOT_HAVE_APM
    rtc::scoped_refptr<webrtc::AudioProcessing> apm;
#endif
};

AudioPreprocessor::AudioPreprocessor() : p_(std::make_unique<Impl>()) {}
AudioPreprocessor::~AudioPreprocessor() = default;
AudioPreprocessor::AudioPreprocessor(AudioPreprocessor&&) noexcept = default;
AudioPreprocessor& AudioPreprocessor::operator=(AudioPreprocessor&&) noexcept = default;

bool AudioPreprocessor::available() {
#ifdef MINTBOT_HAVE_APM
    return true;
#else
    return false;
#endif
}

bool AudioPreprocessor::init(const ApmOptions& opts, std::string* error) {
    p_->opts = opts;
    p_->pending.clear();
    p_->pending_reverse.clear();
    p_->delay_ms = -1;
    if (!opts.enabled) return true;
#ifdef MINTBOT_HAVE_APM
    if (opts.ns_level < 0 || opts.ns_level > 4) {
        if (error) *error = "ns_level must be 0..4";
        return false;
    }
    webrtc::AudioProcessingBuilder builder;
    if (opts.aec && opts.aec_linear_output) builder.SetEchoControlFactory(std::make_unique<LinearAec3Factory>());
    p_->apm = builder.Create();
    if (!p_->apm) {
        if (error) *error = "AudioProcessingBuilder().Create() failed";
        return false;
    }
    webrtc::AudioProcessing::Config c;
    c.high_pass_filter.enabled = opts.high_pass;
    c.noise_suppression.enabled = opts.ns_level > 0;
    if (opts.ns_level > 0)
        c.noise_suppression.level = (webrtc::AudioProcessing::Config::NoiseSuppression::Level)(opts.ns_level - 1);
    c.gain_controller2.enabled = opts.agc2;
    c.gain_controller2.adaptive_digital.enabled = opts.agc2;
    c.echo_canceller.enabled = opts.aec;
    c.echo_canceller.mobile_mode = false;  // AEC3, not AECM
    c.echo_canceller.export_linear_aec_output = opts.aec && opts.aec_linear_output;
    p_->apm->ApplyConfig(c);
    return true;
#else
    if (error) *error = "built without WebRTC APM (MINTBOT_WITH_APM=OFF)";
    return false;
#endif
}

bool AudioPreprocessor::active() const {
#ifdef MINTBOT_HAVE_APM
    return p_->opts.enabled && p_->apm;
#else
    return false;
#endif
}

void AudioPreprocessor::process(const float* in, size_t n, std::vector<float>& out, std::vector<float>* linear) {
    if (!active()) {
        out.insert(out.end(), in, in + n);
        if (linear) linear->insert(linear->end(), in, in + n);
        return;
    }
#ifdef MINTBOT_HAVE_APM
    auto& q = p_->pending;
    q.insert(q.end(), in, in + n);
    const webrtc::StreamConfig sc(16000, 1);
    const bool want_linear = linear && p_->opts.aec && p_->opts.aec_linear_output;
    size_t done = 0;
    float frame[kFrame];
    for (; done + kFrame <= q.size(); done += kFrame) {
        const float* src[1] = {q.data() + done};
        float* dst[1] = {frame};
        if (p_->opts.aec && p_->delay_ms >= 0) p_->apm->set_stream_delay_ms(p_->delay_ms);
        p_->apm->ProcessStream(src, sc, sc, dst);
        out.insert(out.end(), frame, frame + kFrame);
        if (want_linear) {
            std::array<float, kFrame> lin{};
            if (p_->apm->GetLinearAecOutput(rtc::ArrayView<std::array<float, kFrame>>(&lin, 1)))
                linear->insert(linear->end(), lin.begin(), lin.end());
            else
                linear->insert(linear->end(), kFrame, 0.0f);  // keeps it aligned with `out`
        }
    }
    q.erase(q.begin(), q.begin() + done);
#endif
}

void AudioPreprocessor::process_reverse(const float* in, size_t n) {
#ifdef MINTBOT_HAVE_APM
    if (!active() || !p_->opts.aec) return;
    auto& q = p_->pending_reverse;
    q.insert(q.end(), in, in + n);
    const webrtc::StreamConfig sc(16000, 1);
    size_t done = 0;
    for (; done + kFrame <= q.size(); done += kFrame) {
        const float* src[1] = {q.data() + done};
        p_->apm->AnalyzeReverseStream(src, sc);  // analysis only: the render audio itself is not changed
    }
    q.erase(q.begin(), q.begin() + done);
#else
    (void)in;
    (void)n;
#endif
}

void AudioPreprocessor::set_stream_delay_ms(int ms) { p_->delay_ms = ms; }

AecStats AudioPreprocessor::aec_stats() {
    AecStats s;
#ifdef MINTBOT_HAVE_APM
    if (!active() || !p_->opts.aec) return s;
    const auto st = p_->apm->GetStatistics();
    s.echo_return_loss = st.echo_return_loss;
    s.echo_return_loss_enhancement = st.echo_return_loss_enhancement;
    s.divergent_filter_fraction = st.divergent_filter_fraction;
    s.residual_echo_likelihood = st.residual_echo_likelihood;
    s.delay_ms = st.delay_ms;
#endif
    return s;
}

}  // namespace mintbot::dsp
