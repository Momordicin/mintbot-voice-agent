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

class AudioSource {
public:
    using PcmFn = std::function<void(const float* pcm, size_t n)>;
    // state: "started", "stopped", "rerouted", "interrupted", "resumed", "ended", "error"
    using StateFn = std::function<void(const std::string& state, const std::string& detail)>;

    virtual ~AudioSource() = default;
    // Callbacks run on the source's own thread; they must not call stop().
    virtual bool start(PcmFn on_pcm, StateFn on_state, std::string* error) = 0;
    virtual void stop() = 0;
    virtual std::string name() const = 0;
};

// device_index: index from list_capture_devices(), or -1 for the system default
// (which follows default-device changes automatically on WASAPI).
std::unique_ptr<AudioSource> make_capture_source(int device_index);
std::unique_ptr<AudioSource> make_wav_source(const std::string& path, double speed);

}  // namespace mintbot::agent
