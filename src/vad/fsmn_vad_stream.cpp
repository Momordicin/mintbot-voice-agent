// Streaming FSMN-VAD. See fsmn_vad_stream.h for provenance.
//
// Feature extraction (fbank80 / LFR / CMVN) and the FSMN graph are taken from
// the SenseVoice repository's validated ggml runtime (runtime/llama.cpp/
// funasr-common/funasr_vad.h, MIT, Copyright (c) 2025 FunASR). The online
// cache handling (ExtractFeats / OnlineLfrCmvn) follows FunASR's
// runtime/onnxruntime/src/fsmn-vad-online.cpp (MIT) line for line, with the
// deviations marked "deviation:" below.
#include "vad/fsmn_vad_stream.h"

#include "funasr/e2e-vad.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "common/cpu_backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

namespace mintbot::vad {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kFrameLen = 400;    // 25 ms
constexpr int kFrameShift = 160;  // 10 ms
constexpr int kFrameMs = 10;
constexpr int kNfft = 512;
constexpr int kNmel = 80;
constexpr float kPreemph = 0.97f;
constexpr float kLowHz = 20.0f;
constexpr float kHighHz = 8000.0f;

float mel(float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); }

void fft(std::vector<float>& re, std::vector<float>& im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int b = n >> 1;
        for (; j & b; b >>= 1) j ^= b;
        j ^= b;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double a = -2.0 * M_PI / len;
        float wr = cosf(a), wi = sinf(a);
        for (int i = 0; i < n; i += len) {
            float cr = 1, ci = 0;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i + k], ui = im[i + k];
                float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                float nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc;
            }
        }
    }
}

int frame_count(int n_samples) {
    int n = (n_samples - kFrameLen) / kFrameShift + 1;
    return (n >= 1 && n_samples >= kFrameLen) ? n : 0;
}

ggml_tensor* linear(ggml_context* c, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, y, b) : y;
}

}  // namespace

struct FsmnVadStream::Impl {
    FsmnVadOptions opts;

    // weights
    ggml_context* ctx_w = nullptr;
    std::map<std::string, ggml_tensor*> t;
    int idim = 400, pd = 128, nl = 4, lorder = 20, od = 248, lfr_m = 5, lfr_n = 1;
    const float* cmvn_shift = nullptr;
    const float* cmvn_scale = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;

    // fbank tables
    std::vector<std::vector<float>> melbank;
    std::vector<float> window;

    // one VAD session; rebuilt after flush() and after a long idle stretch
    struct Session {
        std::unique_ptr<funasr::E2EVadModel> scorer = std::make_unique<funasr::E2EVadModel>();
        std::vector<float> input_cache;                     // samples after the last whole frame
        std::vector<float> reserve_waveforms;               // waveform aligned with lfr_splice_cache
        std::vector<std::vector<float>> lfr_splice_cache;   // fbank frames awaiting LFR context
        std::vector<std::vector<float>> fsmn_cache;         // per layer: pd x (lorder-1)
        int64_t fed_samples = 0;
        int64_t scored_frames = 0;
        bool open = false;
        int64_t last_end_ms = 0;
    };
    Session s;
    int64_t base_ms = 0;  // absolute time of the current session's first sample

    bool record = false;
    std::vector<float> probs;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (backend) ggml_backend_free(backend);
        if (ctx_w) ggml_free(ctx_w);
    }

    ggml_tensor* g(const std::string& name) {
        auto it = t.find(name);
        return it == t.end() ? nullptr : it->second;
    }

    void new_session() {
        s = Session();
        s.fsmn_cache.assign(nl, std::vector<float>((size_t)pd * (lorder - 1), 0.0f));
    }

    std::vector<float> fbank_frame(const float* x) {
        std::vector<float> fr(kFrameLen), re(kNfft), im(kNfft), out(kNmel);
        double mean = 0;
        for (int i = 0; i < kFrameLen; i++) mean += x[i] * 32768.0f;
        mean /= kFrameLen;
        for (int i = 0; i < kFrameLen; i++) fr[i] = x[i] * 32768.0f - (float)mean;
        for (int i = kFrameLen - 1; i > 0; i--) fr[i] -= kPreemph * fr[i - 1];
        fr[0] -= kPreemph * fr[0];
        for (int i = 0; i < kNfft; i++) { re[i] = i < kFrameLen ? fr[i] * window[i] : 0.0f; im[i] = 0.0f; }
        fft(re, im, kNfft);
        const float floor_e = 1.1920929e-07f;
        for (int m = 0; m < kNmel; m++) {
            float e = 0;
            for (int k = 0; k < kNfft / 2 + 1; k++)
                if (melbank[m][k] > 0) e += melbank[m][k] * (re[k] * re[k] + im[k] * im[k]);
            out[m] = logf(e > floor_e ? e : floor_e);
        }
        return out;
    }

    // FsmnVadOnline::FbankKaldi
    void fbank_kaldi(std::vector<std::vector<float>>& feats, std::vector<float>& waves) {
        waves.insert(waves.begin(), s.input_cache.begin(), s.input_cache.end());
        int n = frame_count((int)waves.size());
        s.input_cache.assign(waves.begin() + (size_t)n * kFrameShift, waves.end());
        if (n == 0) return;
        waves.erase(waves.begin() + (size_t)(n - 1) * kFrameShift + kFrameLen, waves.end());
        for (int i = 0; i < n; i++) feats.push_back(fbank_frame(waves.data() + (size_t)i * kFrameShift));
    }

    // FsmnVadOnline::OnlineLfrCmvn
    int online_lfr_cmvn(std::vector<std::vector<float>>& feats, bool finished) {
        std::vector<std::vector<float>> out;
        int T = (int)feats.size();
        int T_lfr = (int)ceil((T - (lfr_m - 1) / 2) / (float)lfr_n);
        int splice_idx = T_lfr;
        std::vector<float> p;
        for (int i = 0; i < T_lfr; i++) {
            if (lfr_m <= T - i * lfr_n) {
                for (int j = 0; j < lfr_m; j++) p.insert(p.end(), feats[i * lfr_n + j].begin(), feats[i * lfr_n + j].end());
                out.emplace_back(p);
                p.clear();
            } else if (finished) {
                int num_padding = lfr_m - (T - i * lfr_n);
                for (int j = 0; j < (int)feats.size() - i * lfr_n; j++)
                    p.insert(p.end(), feats[i * lfr_n + j].begin(), feats[i * lfr_n + j].end());
                for (int j = 0; j < num_padding; j++) p.insert(p.end(), feats.back().begin(), feats.back().end());
                out.emplace_back(p);
                p.clear();
            } else {
                splice_idx = i;
                break;
            }
        }
        splice_idx = std::min(T - 1, splice_idx * lfr_n);
        s.lfr_splice_cache.assign(feats.begin() + splice_idx, feats.end());
        for (auto& f : out)
            for (int j = 0; j < idim; j++) f[j] = (f[j] + cmvn_shift[j]) * cmvn_scale[j];
        feats = std::move(out);
        return splice_idx;
    }

    // FsmnVadOnline::ExtractFeats
    void extract_feats(std::vector<std::vector<float>>& feats, std::vector<float>& waves, bool finished) {
        fbank_kaldi(feats, waves);
        if (!feats.empty()) {
            if (!s.reserve_waveforms.empty()) waves.insert(waves.begin(), s.reserve_waveforms.begin(), s.reserve_waveforms.end());
            if (s.lfr_splice_cache.empty())
                for (int i = 0; i < (lfr_m - 1) / 2; i++) s.lfr_splice_cache.emplace_back(feats[0]);
            if (feats.size() + s.lfr_splice_cache.size() >= (size_t)lfr_m) {
                feats.insert(feats.begin(), s.lfr_splice_cache.begin(), s.lfr_splice_cache.end());
                int frame_from_waves = ((int)waves.size() - kFrameLen) / kFrameShift + 1;
                int minus_frame = s.reserve_waveforms.empty() ? (lfr_m - 1) / 2 : 0;
                int splice_idx = online_lfr_cmvn(feats, finished);
                int reserve_idx = std::abs(splice_idx - minus_frame);
                s.reserve_waveforms.assign(waves.begin() + (size_t)reserve_idx * kFrameShift,
                                           waves.begin() + (size_t)frame_from_waves * kFrameShift);
                waves.erase(waves.begin() + (size_t)(frame_from_waves - 1) * kFrameShift + kFrameLen, waves.end());
            } else {
                s.reserve_waveforms.assign(waves.begin() + kFrameLen - kFrameShift, waves.end());
                s.lfr_splice_cache.insert(s.lfr_splice_cache.end(), feats.begin(), feats.end());
                // deviation: upstream leaves the raw 80-dim frames in `feats` here and
                // would forward them; its 50 ms server chunks never reach this branch,
                // our 10 ms chunks do. Nothing is ready for the network yet.
                feats.clear();
            }
        } else if (finished) {
            if (!s.reserve_waveforms.empty()) waves = s.reserve_waveforms;
            feats = s.lfr_splice_cache;
            if (!feats.empty()) online_lfr_cmvn(feats, finished);
        }
        // deviation: upstream resets the FSMN cache here when `finished`, so its last
        // chunk runs with zeroed memory. Like FunASR's Python model, we forward the
        // last chunk with the real cache and reset the whole session afterwards.
    }

    bool forward(const std::vector<std::vector<float>>& feats, std::vector<std::vector<float>>& scores) {
        const int n = (int)feats.size();
        const int hist = lorder - 1;
        ggml_init_params ip = {(size_t)8 * 1024 * 1024, nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, idim, n);
        ggml_set_input(x);
        ggml_tensor* h = linear(c, g("encoder.in_linear1.linear.weight"), g("encoder.in_linear1.linear.bias"), x);
        h = ggml_relu(c, linear(c, g("encoder.in_linear2.linear.weight"), g("encoder.in_linear2.linear.bias"), h));
        std::vector<ggml_tensor*> cache_in(nl), cache_out(nl);
        for (int i = 0; i < nl; i++) {
            std::string p = "encoder.fsmn." + std::to_string(i) + ".";
            ggml_tensor* z = ggml_mul_mat(c, g(p + "linear.linear.weight"), h);  // [pd, n]
            cache_in[i] = ggml_new_tensor_2d(c, GGML_TYPE_F32, pd, hist);
            ggml_set_input(cache_in[i]);
            // FSMN memory: the previous (lorder-1) frames precede this chunk instead of
            // zero padding, so chunked output equals a whole-stream forward.
            ggml_tensor* zp = ggml_concat(c, cache_in[i], z, 1);               // [pd, hist + n]
            ggml_tensor* fk = g(p + "fsmn_block.conv_left.weight");            // [pd, lorder]
            ggml_tensor* acc = z;
            for (int j = 0; j < lorder; j++) {
                ggml_tensor* sl = ggml_view_2d(c, zp, pd, n, zp->nb[1], (size_t)j * zp->nb[1]);
                ggml_tensor* wj = ggml_view_1d(c, fk, pd, (size_t)j * fk->nb[1]);
                acc = ggml_add(c, acc, ggml_mul(c, sl, wj));
            }
            cache_out[i] = ggml_cont(c, ggml_view_2d(c, zp, pd, hist, zp->nb[1], (size_t)n * zp->nb[1]));
            ggml_set_output(cache_out[i]);
            h = ggml_relu(c, linear(c, g(p + "affine.linear.weight"), g(p + "affine.linear.bias"), acc));
        }
        h = linear(c, g("encoder.out_linear1.linear.weight"), g("encoder.out_linear1.linear.bias"), h);
        h = linear(c, g("encoder.out_linear2.linear.weight"), g("encoder.out_linear2.linear.bias"), h);
        h = ggml_soft_max(c, h);
        ggml_set_output(h);
        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, h);
        for (auto* co : cache_out) ggml_build_forward_expand(gf, co);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { ggml_free(c); return false; }

        std::vector<float> xin((size_t)idim * n);
        for (int r = 0; r < n; r++) memcpy(&xin[(size_t)r * idim], feats[r].data(), idim * sizeof(float));
        ggml_backend_tensor_set(x, xin.data(), 0, ggml_nbytes(x));
        for (int i = 0; i < nl; i++) ggml_backend_tensor_set(cache_in[i], s.fsmn_cache[i].data(), 0, ggml_nbytes(cache_in[i]));
        mintbot::cpu_backend_set_threads(backend, opts.n_threads);
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            std::vector<float> out((size_t)od * n);
            ggml_backend_tensor_get(h, out.data(), 0, ggml_nbytes(h));
            for (int r = 0; r < n; r++) scores.emplace_back(out.begin() + (size_t)r * od, out.begin() + (size_t)(r + 1) * od);
            for (int i = 0; i < nl; i++) ggml_backend_tensor_get(cache_out[i], s.fsmn_cache[i].data(), 0, ggml_nbytes(cache_out[i]));
        }
        ggml_free(c);
        return ok;
    }

    // FsmnVadOnline::Infer plus conversion to absolute events.
    std::vector<VadEvent> infer(std::vector<float> waves, bool finished) {
        std::vector<VadEvent> ev;
        std::vector<std::vector<float>> feats, scores;
        extract_feats(feats, waves, finished);
        if (!feats.empty() && !forward(feats, scores)) {
            fprintf(stderr, "vad: ggml forward failed\n");
            return ev;
        }
        if (record)
            for (auto& sc : scores) probs.push_back(sc[0]);
        if (!scores.empty() || finished) {
            s.scored_frames += (int64_t)scores.size();
            auto segs = (*s.scorer)(scores, waves, finished, true, opts.max_end_silence_ms,
                                    opts.max_single_segment_ms, opts.speech_noise_thres, kSampleRate);
            for (auto& seg : segs) {
                if (seg[0] != -1) { ev.push_back({VadEvent::SpeechStart, base_ms + seg[0]}); s.open = true; }
                if (seg[1] != -1) { ev.push_back({VadEvent::SpeechEnd, base_ms + seg[1]}); s.open = false; s.last_end_ms = seg[1]; }
            }
        }
        return ev;
    }

    void advance_session() {
        base_ms += s.fed_samples * 1000 / kSampleRate;
        new_session();
    }
};

FsmnVadStream::FsmnVadStream() : p_(std::make_unique<Impl>()) {}
FsmnVadStream::~FsmnVadStream() = default;

bool FsmnVadStream::load(const std::string& path, const FsmnVadOptions& opts) {
    Impl& m = *p_;
    m.opts = opts;
    gguf_init_params ip = {false, &m.ctx_w};
    gguf_context* gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) { fprintf(stderr, "vad: cannot load %s\n", path.c_str()); return false; }
    auto rd = [&](const char* k, int d) { int64_t i = gguf_find_key(gg, k); return i < 0 ? d : (int)gguf_get_val_u32(gg, i); };
    m.idim = rd("vad.input_dim", 400); m.pd = rd("vad.proj_dim", 128); m.nl = rd("vad.fsmn_layers", 4);
    m.lorder = rd("vad.lorder", 20); m.od = rd("vad.output_dim", 248); m.lfr_m = rd("vad.lfr_m", 5); m.lfr_n = rd("vad.lfr_n", 1);
    for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++) {
        const char* nm = gguf_get_tensor_name(gg, i);
        m.t[nm] = ggml_get_tensor(m.ctx_w, nm);
    }
    gguf_free(gg);

    bool ok = m.g("cmvn.shift") && m.g("cmvn.scale") && m.g("encoder.in_linear1.linear.weight") &&
              m.g("encoder.in_linear2.linear.weight") && m.g("encoder.out_linear1.linear.weight") &&
              m.g("encoder.out_linear2.linear.weight");
    for (int i = 0; i < m.nl && ok; i++) {
        std::string p = "encoder.fsmn." + std::to_string(i) + ".";
        ok = m.g(p + "linear.linear.weight") && m.g(p + "fsmn_block.conv_left.weight") && m.g(p + "affine.linear.weight");
    }
    if (!ok || m.idim != m.lfr_m * kNmel) { fprintf(stderr, "vad: unexpected GGUF layout in %s\n", path.c_str()); return false; }
    m.cmvn_shift = (const float*)m.g("cmvn.shift")->data;
    m.cmvn_scale = (const float*)m.g("cmvn.scale")->data;

    m.window.resize(kFrameLen);
    for (int i = 0; i < kFrameLen; i++) m.window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (kFrameLen - 1));  // same expression as funasr_vad.h
    const int nb = kNfft / 2 + 1;
    float bw = (float)kSampleRate / kNfft, ml = mel(kLowHz), mh = mel(kHighHz), dm = (mh - ml) / (kNmel + 1);
    m.melbank.assign(kNmel, std::vector<float>(nb, 0.0f));
    for (int k = 0; k < kNmel; k++) {
        float L = ml + k * dm, C = ml + (k + 1) * dm, R = ml + (k + 2) * dm;
        for (int b = 0; b < nb; b++) {
            float mf = mel(bw * b);
            if (mf > L && mf < R) m.melbank[k][b] = mf <= C ? (mf - L) / (C - L) : (R - mf) / (R - C);
        }
    }

    m.backend = mintbot::cpu_backend_init();
    m.galloc = m.backend ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend)) : nullptr;
    m.new_session();
    m.base_ms = 0;
    return m.backend && m.galloc;
}

std::vector<VadEvent> FsmnVadStream::feed(const float* pcm, size_t n) {
    Impl& m = *p_;
    if (n == 0) return {};
    m.s.fed_samples += (int64_t)n;
    auto ev = m.infer(std::vector<float>(pcm, pcm + n), false);
    int64_t now_ms = m.s.fed_samples * 1000 / kSampleRate;
    if (m.opts.idle_reset_ms > 0 && !m.s.open && now_ms - m.s.last_end_ms >= m.opts.idle_reset_ms) m.advance_session();
    return ev;
}

std::vector<VadEvent> FsmnVadStream::flush() {
    Impl& m = *p_;
    auto ev = m.infer({}, true);
    if (m.s.open) ev.push_back({VadEvent::SpeechEnd, m.base_ms + m.s.scored_frames * kFrameMs});
    m.advance_session();
    return ev;
}

bool FsmnVadStream::in_speech() const { return p_->s.open; }
int64_t FsmnVadStream::pending_end_ms(int64_t* silence_ms) const {
    const Impl& m = *p_;
    int sil = 0;
    const int frame = m.s.open ? m.s.scorer->PendingEndFrame(&sil) : -1;
    if (silence_ms) *silence_ms = frame < 0 ? 0 : (int64_t)sil * kFrameMs;
    // OnVoiceEnd(f) closes the segment at the end of frame f
    return frame < 0 ? -1 : m.base_ms + (int64_t)(frame + 1) * kFrameMs;
}
void FsmnVadStream::record_probs(bool on) { p_->record = on; }
const std::vector<float>& FsmnVadStream::recorded_probs() const { return p_->probs; }

}  // namespace mintbot::vad
