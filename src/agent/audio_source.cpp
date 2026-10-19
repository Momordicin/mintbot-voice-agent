#include "agent/audio_source.h"

#include "common/wav_io.h"

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace mintbot::agent {

std::vector<CaptureDevice> list_capture_devices(std::string* error) {
    std::vector<CaptureDevice> out;
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        if (error) *error = "ma_context_init failed";
        return out;
    }
    ma_device_info* infos = nullptr;
    ma_uint32 n = 0;
    if (ma_context_get_devices(&ctx, nullptr, nullptr, &infos, &n) == MA_SUCCESS) {
        for (ma_uint32 i = 0; i < n; i++) out.push_back({(int)i, infos[i].name, infos[i].isDefault != 0});
    } else if (error) {
        *error = "ma_context_get_devices failed";
    }
    ma_context_uninit(&ctx);
    return out;
}

const char* capture_error_code(const std::string& error) {
    return error.find("(" + std::to_string((int)MA_ACCESS_DENIED) + ")") != std::string::npos ? "mic_access_denied"
                                                                                               : "capture_failed";
}

namespace {

class CaptureSource : public AudioSource {
public:
    explicit CaptureSource(int index) : index_(index) {}
    ~CaptureSource() override { stop(); }

    bool start(PcmFn on_pcm, StateFn on_state, std::string* error) override {
        on_pcm_ = std::move(on_pcm);
        on_state_ = std::move(on_state);
        if (ma_context_init(nullptr, 0, nullptr, &ctx_) != MA_SUCCESS) return fail(error, "ma_context_init failed");
        have_ctx_ = true;

        ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
        cfg.capture.format = ma_format_f32;
        cfg.capture.channels = 1;
        cfg.sampleRate = 16000;  // miniaudio converts from the device's native format
        cfg.periodSizeInMilliseconds = 10;
        cfg.dataCallback = &CaptureSource::on_data;
        cfg.notificationCallback = &CaptureSource::on_notification;
        cfg.pUserData = this;
        if (index_ >= 0) {
            ma_device_info* infos = nullptr;
            ma_uint32 n = 0;
            if (ma_context_get_devices(&ctx_, nullptr, nullptr, &infos, &n) != MA_SUCCESS || (ma_uint32)index_ >= n)
                return fail(error, "capture device " + std::to_string(index_) + " not found");
            dev_id_ = infos[index_].id;
            cfg.capture.pDeviceID = &dev_id_;
        }
        if (ma_result r = ma_device_init(&ctx_, &cfg, &dev_); r != MA_SUCCESS) return fail(error, "ma_device_init", r);
        have_dev_ = true;
        char nm[256] = {0};
        ma_device_get_name(&dev_, ma_device_type_capture, nm, sizeof nm, nullptr);
        name_ = nm;
        if (ma_result r = ma_device_start(&dev_); r != MA_SUCCESS) return fail(error, "ma_device_start", r);
        return true;
    }

    void stop() override {
        if (have_dev_) { ma_device_uninit(&dev_); have_dev_ = false; }
        if (have_ctx_) { ma_context_uninit(&ctx_); have_ctx_ = false; }
    }

    std::string name() const override { return name_.empty() ? std::string("capture") : name_; }

private:
    bool fail(std::string* error, const std::string& msg) {
        if (error) *error = msg;
        stop();
        return false;
    }
    // "<step> failed: <miniaudio description> (<code>)"; access denied usually means
    // Windows Settings > Privacy > Microphone blocks desktop apps.
    bool fail(std::string* error, const char* step, ma_result r) {
        std::string msg = std::string(step) + " failed: " + ma_result_description(r) + " (" + std::to_string((int)r) + ")";
        if (r == MA_ACCESS_DENIED) msg += "; microphone access may be blocked in Windows privacy settings";
        return fail(error, msg);
    }

    static void on_data(ma_device* d, void*, const void* input, ma_uint32 frames) {
        auto* self = (CaptureSource*)d->pUserData;
        if (input && frames) self->on_pcm_((const float*)input, frames);
    }

    static void on_notification(const ma_device_notification* n) {
        auto* self = (CaptureSource*)n->pDevice->pUserData;
        switch (n->type) {
            case ma_device_notification_type_started: self->on_state_("started", self->name()); break;
            case ma_device_notification_type_stopped: self->on_state_("stopped", self->name()); break;
            case ma_device_notification_type_rerouted: self->on_state_("rerouted", self->name()); break;
            case ma_device_notification_type_interruption_began: self->on_state_("interrupted", self->name()); break;
            case ma_device_notification_type_interruption_ended: self->on_state_("resumed", self->name()); break;
            default: break;
        }
    }

    int index_;
    ma_context ctx_{};
    ma_device dev_{};
    ma_device_id dev_id_{};
    bool have_ctx_ = false, have_dev_ = false;
    std::string name_;
    PcmFn on_pcm_;
    StateFn on_state_;
};

class WavSource : public AudioSource {
public:
    WavSource(std::string path, double speed) : path_(std::move(path)), speed_(speed > 0 ? speed : 1.0) {}
    ~WavSource() override { stop(); }

    bool start(PcmFn on_pcm, StateFn on_state, std::string* error) override {
        std::vector<float> pcm;
        if (!read_wav16k(path_.c_str(), pcm)) {
            if (error) *error = "need a 16 kHz mono PCM16 wav: " + path_;
            return false;
        }
        run_ = true;
        th_ = std::thread([this, pcm = std::move(pcm), on_pcm = std::move(on_pcm), on_state = std::move(on_state)] {
            on_state("started", name());
            const size_t step = 160;  // 10 ms, like a capture period
            auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < pcm.size() && run_; i += step) {
                on_pcm(pcm.data() + i, std::min(step, pcm.size() - i));
                auto due = t0 + std::chrono::microseconds((int64_t)((i + step) / 16.0 * 1000 / speed_));
                std::this_thread::sleep_until(due);
            }
            if (run_) on_state("ended", name());
        });
        return true;
    }

    void stop() override {
        run_ = false;
        if (th_.joinable()) th_.join();
    }

    std::string name() const override { return "wav:" + path_; }

private:
    std::string path_;
    double speed_;
    std::atomic<bool> run_{false};
    std::thread th_;
};

}  // namespace

std::unique_ptr<AudioSource> make_capture_source(int device_index) { return std::make_unique<CaptureSource>(device_index); }
std::unique_ptr<AudioSource> make_wav_source(const std::string& path, double speed) {
    return std::make_unique<WavSource>(path, speed);
}

}  // namespace mintbot::agent
