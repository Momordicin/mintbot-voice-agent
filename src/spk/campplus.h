// CAM++ speaker embedding (3D-Speaker / FunASR speech_campplus_sv_zh-cn_16k-common)
// on ggml. Front end: Kaldi fbank (80 mel, povey window, as torchaudio) with
// per-window mean normalisation; network: FCM 2-D head, TDNN, three CAM dense
// TDNN blocks, statistics pooling, 192-d dense layer.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mintbot::spk {

constexpr size_t kSvWindow = 24000;  // sv_chunk: 1.5 s windows ...
constexpr size_t kSvShift = 12000;   // ... every 0.75 s

// sv_chunk() windows of an n-sample utterance as [begin, end) offsets, in order.
// Every window but possibly the last is a full window starting at k * kSvShift,
// so it only depends on the first k * kSvShift + kSvWindow samples.
std::vector<std::pair<size_t, size_t>> sv_chunk_windows(size_t n);

// Raw embeddings of full windows computed ahead of time, keyed by start offset.
using WindowCache = std::map<size_t, std::vector<float>>;

class CamPlus {
public:
    CamPlus();
    ~CamPlus();
    CamPlus(const CamPlus&) = delete;
    CamPlus& operator=(const CamPlus&) = delete;

    bool load(const std::string& gguf_path, int n_threads = 2);
    // Polled during embedding; returning true abandons it (the call then fails).
    void set_abort_callback(ggml_abort_callback cb, void* data);
    int dim() const;

    // Raw (unnormalised) embedding of one 16 kHz mono window.
    bool embed_window(const float* pcm, size_t n, std::vector<float>& emb);

    // FunASR sv_chunk(): 1.5 s windows every 0.75 s, the last / a short one
    // zero-padded to 1.5 s; one raw embedding per window. Full windows found
    // in `pre` are taken from it instead of being computed; `reused` counts them.
    bool embed_utterance(const float* pcm, size_t n, std::vector<std::vector<float>>& embs,
                         const WindowCache* pre = nullptr, int* reused = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Mean of the L2-normalised embeddings, L2-normalised again.
std::vector<float> average_normalized(const std::vector<std::vector<float>>& embs);
float cosine(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace mintbot::spk
