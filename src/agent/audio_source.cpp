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

#ifdef _WIN32
#undef CLSCTX_ALL  // miniaudio's own definition, the same value as the SDK's
#include <windows.h>
#include <objbase.h>
#include <mmdeviceapi.h>
#include <propsys.h>
#include <devquery.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
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

#ifdef _WIN32
// Endpoint and device-container properties. Defined here rather than through
// <initguid.h>, which would also define every GUID of the headers above.
const PROPERTYKEY kEndpointName = {{0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};
const PROPERTYKEY kContainerId = {{0x8c7ed206, 0x3f8a, 0x4827, {0xb3, 0xab, 0xae, 0x9e, 0x1f, 0xae, 0xfc, 0x6c}}, 2};
const DEVPROPKEY kItemNameDisplay = {{0xb725f130, 0x47ef, 0x101a, {0xa5, 0xf1, 0x02, 0x60, 0x8c, 0x9e, 0xeb, 0xac}}, 10};
// the container of everything built into the computer; its name is the computer's
const GUID kThisComputer = {0x00000000, 0x0000, 0x0000, {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

std::string utf8(const wchar_t* w) {
    if (!w || !*w) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string guid_str(const GUID& g) {
    wchar_t b[40];
    return StringFromGUID2(g, b, 40) ? utf8(b) : std::string();
}

// Name of a physical device (device container); empty unless it is a plain string.
// DevGetObjectProperties (Windows 10+) is loaded at run time, so nothing changes at link time.
std::string container_name(const GUID& container) {
    using GetFn = HRESULT(WINAPI*)(DEV_OBJECT_TYPE, PCWSTR, ULONG, ULONG, const DEVPROPCOMPKEY*, PULONG, const DEVPROPERTY**);
    using FreeFn = void(WINAPI*)(ULONG, const DEVPROPERTY*);
    static HMODULE mod = LoadLibraryExW(L"cfgmgr32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    static auto get = mod ? (GetFn)GetProcAddress(mod, "DevGetObjectProperties") : nullptr;
    static auto release = mod ? (FreeFn)GetProcAddress(mod, "DevFreeObjectProperties") : nullptr;
    if (!get || !release) return {};
    wchar_t id[40];
    if (!StringFromGUID2(container, id, 40)) return {};
    DEVPROPCOMPKEY key{};
    key.Key = kItemNameDisplay;
    key.Store = DEVPROP_STORE_SYSTEM;
    ULONG n = 0;
    const DEVPROPERTY* props = nullptr;
    if (FAILED(get(DevObjectTypeDeviceContainer, id, DevQueryFlagNone, 1, &key, &n, &props))) return {};
    std::string name;
    // DEVPROP_TYPE_STRING_INDIRECT (e.g. monitors' audio) is a resource reference, not a name
    if (n == 1 && props[0].Type == DEVPROP_TYPE_STRING && props[0].Buffer) name = utf8((const wchar_t*)props[0].Buffer);
    release(n, props);
    return name;
}

// COM must be initialised on the calling thread.
DeviceIdentity query_endpoint(const std::wstring& endpoint_id) {
    DeviceIdentity d;
    d.device_id = utf8(endpoint_id.c_str());
    IMMDeviceEnumerator* en = nullptr;
    IMMDevice* dev = nullptr;
    IPropertyStore* ps = nullptr;
    GUID container{};
    bool have_container = false;
    if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                   (void**)&en)) &&
        SUCCEEDED(en->GetDevice(endpoint_id.c_str(), &dev)) && SUCCEEDED(dev->OpenPropertyStore(STGM_READ, &ps))) {
        PROPVARIANT v;
        PropVariantInit(&v);
        if (SUCCEEDED(ps->GetValue(kEndpointName, &v)) && v.vt == VT_LPWSTR) d.name = utf8(v.pwszVal);
        PropVariantClear(&v);
        if (SUCCEEDED(ps->GetValue(kContainerId, &v)) && v.vt == VT_CLSID && v.puuid) {
            container = *v.puuid;
            have_container = true;
            d.container_id = guid_str(container);
        }
        PropVariantClear(&v);
    }
    if (ps) ps->Release();
    if (dev) dev->Release();
    if (en) en->Release();
    std::string physical = have_container && !IsEqualGUID(container, kThisComputer) ? container_name(container) : "";
    d.display_name = physical.empty() ? d.name : physical;
    return d;
}

// Works on any thread: COM is initialised for the call if it is not yet, and an
// apartment the thread already has is kept.
DeviceIdentity lookup_endpoint(const std::wstring& endpoint_id) {
    HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);  // RPC_E_CHANGED_MODE: already an STA, usable
    DeviceIdentity d;
    if (SUCCEEDED(co) || co == RPC_E_CHANGED_MODE) d = query_endpoint(endpoint_id);
    else d.device_id = utf8(endpoint_id.c_str());
    if (SUCCEEDED(co)) CoUninitialize();
    return d;
}
#endif

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
        refresh_identity();
        if (ma_result r = ma_device_start(&dev_); r != MA_SUCCESS) return fail(error, "ma_device_start", r);
        return true;
    }

    void stop() override {
        if (have_dev_) { ma_device_uninit(&dev_); have_dev_ = false; }
        if (have_ctx_) { ma_context_uninit(&ctx_); have_ctx_ = false; }
    }

    std::string name() const override {
        std::string n = identity().name;
        return n.empty() ? std::string("capture") : n;
    }

    DeviceIdentity identity() const override {
        std::lock_guard<std::mutex> lk(id_m_);
        return identity_;
    }

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

    // Called after opening and after each reroute, before the device is announced.
    // The endpoint is the one miniaudio actually opened (ma_device_get_info would
    // give the current default). miniaudio's name is the fallback: for the default
    // device it is the current default, the device in use at these two moments.
    void refresh_identity() {
        DeviceIdentity d;
#ifdef _WIN32
        if (dev_.pContext->backend == ma_backend_wasapi) d = lookup_endpoint(dev_.capture.id.wasapi);
#endif
        if (d.name.empty()) {
            char nm[256] = {0};
            ma_device_get_name(&dev_, ma_device_type_capture, nm, sizeof nm, nullptr);
            d.name = nm;
        }
        if (d.display_name.empty()) d.display_name = d.name;
        std::lock_guard<std::mutex> lk(id_m_);
        identity_ = d;
    }

    static void on_data(ma_device* d, void*, const void* input, ma_uint32 frames) {
        auto* self = (CaptureSource*)d->pUserData;
        if (input && frames) self->on_pcm_((const float*)input, frames);
    }

    static void on_notification(const ma_device_notification* n) {
        auto* self = (CaptureSource*)n->pDevice->pUserData;
        switch (n->type) {
            case ma_device_notification_type_started: self->on_state_("started", self->identity()); break;
            case ma_device_notification_type_stopped: self->on_state_("stopped", self->identity()); break;
            case ma_device_notification_type_rerouted:
                self->refresh_identity();  // miniaudio has opened the new default device
                self->on_state_("rerouted", self->identity());
                break;
            case ma_device_notification_type_interruption_began: self->on_state_("interrupted", self->identity()); break;
            case ma_device_notification_type_interruption_ended: self->on_state_("resumed", self->identity()); break;
            default: break;
        }
    }


    int index_;
    ma_context ctx_{};
    ma_device dev_{};
    ma_device_id dev_id_{};
    bool have_ctx_ = false, have_dev_ = false;
    PcmFn on_pcm_;
    StateFn on_state_;

    mutable std::mutex id_m_;
    DeviceIdentity identity_;
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
            on_state("started", identity());
            const size_t step = 160;  // 10 ms, like a capture period
            auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < pcm.size() && run_; i += step) {
                on_pcm(pcm.data() + i, std::min(step, pcm.size() - i));
                auto due = t0 + std::chrono::microseconds((int64_t)((i + step) / 16.0 * 1000 / speed_));
                std::this_thread::sleep_until(due);
            }
            if (run_) on_state("ended", identity());
        });
        return true;
    }

    void stop() override {
        run_ = false;
        if (th_.joinable()) th_.join();
    }

    std::string name() const override { return "wav:" + path_; }
    DeviceIdentity identity() const override { return {name(), "", "", ""}; }

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
