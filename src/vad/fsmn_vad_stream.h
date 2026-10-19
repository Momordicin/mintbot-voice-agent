// Streaming FSMN-VAD: FunASR's online VAD pipeline with the network on ggml.
//
// Port of FunASR's C++ runtime (runtime/onnxruntime/src/fsmn-vad-online.cpp,
// MIT): the online fbank/LFR/CMVN caches and the E2EVadModel end-point state
// machine (third_party/funasr/e2e-vad.h) are kept as upstream wrote them; only
// the FSMN forward runs on ggml instead of ONNX Runtime, carrying the same
// per-layer (lorder-1)-frame memory cache across calls.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mintbot::vad {

struct FsmnVadOptions {
    int   max_end_silence_ms    = 800;    // silence that closes a segment (FunASR runtime default)
    int   max_single_segment_ms = 30000;  // longer speech is force-split
    float speech_noise_thres    = 0.6f;   // fsmn-vad config.yaml model_conf.speech_noise_thres
    int   n_threads             = 1;
    // Rebuild all state after this long with no open segment, so an always-on
    // stream never grows the scorer's per-frame history or overflows its
    // sample counters. 0 disables it.
    int64_t idle_reset_ms       = 600000;
};

// A speech boundary in absolute stream time (ms since the first fed sample).
struct VadEvent {
    enum Type { SpeechStart, SpeechEnd } type;
    int64_t ms;
};

class FsmnVadStream {
public:
    FsmnVadStream();
    ~FsmnVadStream();
    FsmnVadStream(const FsmnVadStream&) = delete;
    FsmnVadStream& operator=(const FsmnVadStream&) = delete;

    bool load(const std::string& gguf_path, const FsmnVadOptions& opts = {});

    // Feed 16 kHz mono float PCM in [-1, 1], any length. Returns the boundaries
    // this call completed, in order.
    std::vector<VadEvent> feed(const float* pcm, size_t n);

    // End of input: closes an open segment, then starts a fresh session.
    std::vector<VadEvent> flush();

    bool in_speech() const;

    // Where the open segment will end (absolute ms, as its SpeechEnd event will
    // report it) if the current silence lasts until max_end_silence_ms; -1 when
    // no segment is open or nobody is silent. *silence_ms: silence run so far.
    int64_t pending_end_ms(int64_t* silence_ms) const;

    // Tests: record the per-frame silence probability (output pdf 0).
    void record_probs(bool on);
    const std::vector<float>& recorded_probs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::vad
