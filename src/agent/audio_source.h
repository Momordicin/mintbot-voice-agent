// Audio sources delivering 16 kHz mono float PCM: a capture device through
// miniaudio, or a WAV file replayed at (a multiple of) real time for tests.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mintbot::agent {

struct CaptureDevice {
    int index;
    std::string name;  // UTF-8
    bool is_default;
};

std::vector<CaptureDevice> list_capture_devices(std::string* error = nullptr);

// Machine-readable class of a start() error: "mic_access_denied" (Windows privacy
// settings block the microphone; ms-settings:privacy-microphone) or "capture_failed".
const char* capture_error_code(const std::string& error);

// Which device a source is actually capturing from. Windows (WASAPI) only; the
// fields are empty where nothing is known (WAV replay, a failed query).
struct DeviceIdentity {
    std::string name;          // endpoint name, e.g. "耳机 (AirPods Pro)"; for WAV replay "wav:<path>"
    std::string device_id;     // endpoint ID; changes when the driver is updated or the device reinstalled
    std::string container_id;  // physical device, e.g. "{AC00D6F5-...}": shared by all endpoints of one
                               // headset (playback, and capture over each Bluetooth profile)
    std::string display_name;  // the physical device's name ("Timekeeper的AirPods Pro"), or the
                               // endpoint name for devices built into the computer
};

class AudioSource {
public:
    using PcmFn = std::function<void(const float* pcm, size_t n)>;
    // state: "started", "stopped", "rerouted", "interrupted", "resumed", "ended", "error";
    // `device` is the device at that moment (after a reroute: the new one)
    using StateFn = std::function<void(const std::string& state, const DeviceIdentity& device)>;

    virtual ~AudioSource() = default;
    // Callbacks run on the source's own thread; they must not call stop().
    virtual bool start(PcmFn on_pcm, StateFn on_state, std::string* error) = 0;
    virtual void stop() = 0;
    virtual std::string name() const = 0;
    virtual DeviceIdentity identity() const = 0;  // current device; any thread
};

// device_index: index from list_capture_devices(), or -1 for the system default
// (which follows default-device changes automatically on WASAPI).
std::unique_ptr<AudioSource> make_capture_source(int device_index);
std::unique_ptr<AudioSource> make_wav_source(const std::string& path, double speed);

}  // namespace mintbot::agent
