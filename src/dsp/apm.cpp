#include "dsp/apm.h"

#ifdef MINTBOT_HAVE_APM
#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"
#endif

namespace mintbot::dsp {

constexpr size_t kFrame = 160;  // 10 ms at 16 kHz

struct AudioPreprocessor::Impl {
    ApmOptions opts;
    std::vector<float> pending;
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
    if (!opts.enabled) return true;
#ifdef MINTBOT_HAVE_APM
    if (opts.ns_level < 0 || opts.ns_level > 4) {
        if (error) *error = "ns_level must be 0..4";
        return false;
    }
    p_->apm = webrtc::AudioProcessingBuilder().Create();
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

void AudioPreprocessor::process(const float* in, size_t n, std::vector<float>& out) {
    if (!active()) {
        out.insert(out.end(), in, in + n);
        return;
    }
#ifdef MINTBOT_HAVE_APM
    auto& q = p_->pending;
    q.insert(q.end(), in, in + n);
    const webrtc::StreamConfig sc(16000, 1);
    size_t done = 0;
    float frame[kFrame];
    for (; done + kFrame <= q.size(); done += kFrame) {
        const float* src[1] = {q.data() + done};
        float* dst[1] = {frame};
        p_->apm->ProcessStream(src, sc, sc, dst);
        out.insert(out.end(), frame, frame + kFrame);
    }
    q.erase(q.begin(), q.begin() + done);
#endif
}

}  // namespace mintbot::dsp
