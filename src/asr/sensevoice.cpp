// SenseVoiceSmall on ggml. See sensevoice.h for provenance.
#include "asr/sensevoice.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "common/cpu_backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace mintbot::asr {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kWinLen = 400, kShift = 160, kNfft = 512, kNmel = 80, kLfrM = 7, kLfrN = 6;
constexpr int kFeatDim = kLfrM * kNmel;  // 560
constexpr float kPreemph = 0.97f, kLowHz = 20.0f, kHighHz = 8000.0f, kLnEps = 1e-5f;
constexpr int kGraphSize = 32768;

// Query-token embedding rows (FunASR SenseVoiceSmall: lid_dict / textnorm_dict).
int lid_index(const std::string& lang) {
    static const std::map<std::string, int> lid = {{"auto", 0}, {"zh", 3}, {"en", 4}, {"yue", 7},
                                                   {"ja", 11}, {"ko", 12}, {"nospeech", 13}};
    auto it = lid.find(lang);
    return it == lid.end() ? -1 : it->second;
}

float mel(float f) { return 1127.0f * logf(1.0f + f / 700.0f); }

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

// 80-mel fbank + LFR(7, 6), identical to funasr-sensevoice.cpp compute_fbank().
// SenseVoice feeds the raw log-mel to the encoder: no CMVN at inference.
std::vector<float> compute_fbank(const float* pcm, size_t n, int& T_out) {
    std::vector<float> wav(pcm, pcm + n);
    for (auto& v : wav) v *= 32768.0f;
    std::vector<float> win(kWinLen);
    for (int i = 0; i < kWinLen; i++) win[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (kWinLen - 1));
    const int nbin = kNfft / 2 + 1;
    float bw = (float)kSampleRate / kNfft, ml = mel(kLowHz), mh = mel(kHighHz), dm = (mh - ml) / (kNmel + 1);
    std::vector<std::vector<float>> fb(kNmel, std::vector<float>(nbin, 0.0f));
    for (int m = 0; m < kNmel; m++) {
        float L = ml + m * dm, C = ml + (m + 1) * dm, R = ml + (m + 2) * dm;
        for (int k = 0; k < nbin; k++) {
            float mf = mel(bw * k);
            if (mf > L && mf < R) fb[m][k] = mf <= C ? (mf - L) / (C - L) : (R - mf) / (R - C);
        }
    }
    int N = (int)wav.size(), T = (N - kWinLen) / kShift + 1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(kNmel));
    std::vector<float> re(kNfft), im(kNfft), fr(kWinLen);
    const float floor_e = 1.1920929e-07f;
    for (int t = 0; t < T; t++) {
        const float* s = wav.data() + (size_t)t * kShift;
        double mn = 0;
        for (int i = 0; i < kWinLen; i++) mn += s[i];
        mn /= kWinLen;
        for (int i = 0; i < kWinLen; i++) fr[i] = s[i] - (float)mn;
        for (int i = kWinLen - 1; i > 0; i--) fr[i] -= kPreemph * fr[i - 1];
        fr[0] -= kPreemph * fr[0];
        for (int i = 0; i < kNfft; i++) { re[i] = i < kWinLen ? fr[i] * win[i] : 0.0f; im[i] = 0.0f; }
        fft(re, im, kNfft);
        for (int m = 0; m < kNmel; m++) {
            float e = 0;
            for (int k = 0; k < nbin; k++) if (fb[m][k] > 0) e += fb[m][k] * (re[k] * re[k] + im[k] * im[k]);
            feat[t][m] = logf(e > floor_e ? e : floor_e);
        }
    }
    const int pad = (kLfrM - 1) / 2;
    int Tl = (T + kLfrN - 1) / kLfrN;
    std::vector<std::vector<float>> padded;
    padded.reserve(T + pad + kLfrM);
    for (int i = 0; i < pad; i++) padded.push_back(feat[0]);
    for (int t = 0; t < T; t++) padded.push_back(feat[t]);
    while ((int)padded.size() < (Tl - 1) * kLfrN + kLfrM) padded.push_back(feat[T - 1]);
    std::vector<float> out((size_t)Tl * kFeatDim);
    for (int i = 0; i < Tl; i++)
        for (int j = 0; j < kLfrM; j++)
            memcpy(&out[(size_t)i * kFeatDim + j * kNmel], padded[i * kLfrN + j].data(), kNmel * sizeof(float));
    T_out = Tl;
    return out;
}

void add_posenc(std::vector<float>& x, int T, int depth) {
    double inc = log(10000.0) / (depth / 2.0 - 1.0);
    for (int t = 0; t < T; t++) {
        double pos = t + 1;
        for (int i = 0; i < depth / 2; i++) {
            double st = pos * exp(i * -inc);
            x[(size_t)t * depth + i] += (float)sin(st);
            x[(size_t)t * depth + depth / 2 + i] += (float)cos(st);
        }
    }
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(' ');
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(' ') - a + 1);
}

bool in(const std::string& v, std::initializer_list<const char*> set) {
    for (const char* s : set) if (v == s) return true;
    return false;
}

}  // namespace

struct SenseVoice::Impl {
    SenseVoiceOptions opts;
    int d_model = 512, n_head = 4, num_blocks = 50, tp_blocks = 20, kernel = 11, vocab_size = 25055, blank = 0;
    ggml_context* ctx_w = nullptr;
    std::map<std::string, ggml_tensor*> t;
    std::vector<std::string> vocab;
    std::vector<float> query;  // nq x 560 query-token embedding rows
    int nq = 0;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (backend) ggml_backend_free(backend);
        if (ctx_w) ggml_free(ctx_w);
    }

    ggml_tensor* g(const std::string& name) {
        auto it = t.find(name);
        if (it == t.end()) { fprintf(stderr, "sensevoice: missing tensor %s\n", name.c_str()); abort(); }
        return it->second;
    }

    static ggml_tensor* lin(ggml_context* c, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
        ggml_tensor* y = ggml_mul_mat(c, w, x);
        return b ? ggml_add(c, y, b) : y;
    }
    static ggml_tensor* lnorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, kLnEps), w), b);
    }

    ggml_tensor* sanm_attn(ggml_context* c, const std::string& p, ggml_tensor* x, int T) {
        const int D = d_model, H = n_head, dk = D / H, K = kernel;
        ggml_tensor* qkv = lin(c, g(p + "linear_q_k_v.weight"), g(p + "linear_q_k_v.bias"), x);
        size_t nb1 = qkv->nb[1];
        ggml_tensor* q = ggml_cont(c, ggml_view_2d(c, qkv, D, T, nb1, 0));
        ggml_tensor* k = ggml_cont(c, ggml_view_2d(c, qkv, D, T, nb1, (size_t)D * sizeof(float)));
        ggml_tensor* v = ggml_cont(c, ggml_view_2d(c, qkv, D, T, nb1, (size_t)2 * D * sizeof(float)));
        const int pad = (K - 1) / 2;
        ggml_tensor* fk = g(p + "fsmn_block.weight");
        ggml_tensor* vp = ggml_pad_ext(c, v, 0, 0, pad, pad, 0, 0, 0, 0);
        ggml_tensor* fsmn = v;
        for (int j = 0; j < K; j++) {
            ggml_tensor* sl = ggml_view_2d(c, vp, D, T, vp->nb[1], (size_t)j * vp->nb[1]);
            ggml_tensor* wj = ggml_view_1d(c, fk, D, (size_t)j * fk->nb[1]);
            fsmn = ggml_add(c, fsmn, ggml_mul(c, ggml_cont(c, sl), wj));
        }
        q = ggml_permute(c, ggml_reshape_3d(c, q, dk, H, T), 0, 2, 1, 3);
        k = ggml_permute(c, ggml_reshape_3d(c, k, dk, H, T), 0, 2, 1, 3);
        ggml_tensor* vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, dk, H, T), 1, 2, 0, 3));
        ggml_tensor* kq = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, k, q), 1.0f / sqrtf((float)dk)));
        ggml_tensor* o = ggml_cont_2d(c, ggml_permute(c, ggml_mul_mat(c, vh, kq), 0, 2, 1, 3), D, T);
        return ggml_add(c, lin(c, g(p + "linear_out.weight"), g(p + "linear_out.bias"), o), fsmn);
    }

    ggml_tensor* sanm_layer(ggml_context* c, const std::string& p, ggml_tensor* x, int T, bool residual) {
        ggml_tensor* r = x;
        ggml_tensor* h = lnorm(c, x, g(p + "norm1.weight"), g(p + "norm1.bias"));
        ggml_tensor* sa = sanm_attn(c, p + "self_attn.", h, T);
        x = residual ? ggml_add(c, r, sa) : sa;
        r = x;
        h = lnorm(c, x, g(p + "norm2.weight"), g(p + "norm2.bias"));
        h = ggml_relu(c, lin(c, g(p + "feed_forward.w_1.weight"), g(p + "feed_forward.w_1.bias"), h));
        h = lin(c, g(p + "feed_forward.w_2.weight"), g(p + "feed_forward.w_2.bias"), h);
        return ggml_add(c, r, h);
    }

    // Encoder + CTC head on [T, 560] features -> greedy-CTC token ids.
    bool encode(const std::vector<float>& fb, int T, std::vector<int>& ids) {
        const int N = nq + T;
        std::vector<float> inp((size_t)N * kFeatDim);
        memcpy(inp.data(), query.data(), query.size() * sizeof(float));
        memcpy(&inp[(size_t)nq * kFeatDim], fb.data(), (size_t)T * kFeatDim * sizeof(float));
        float sc = sqrtf((float)d_model);
        for (auto& v : inp) v *= sc;
        add_posenc(inp, N, kFeatDim);

        ggml_init_params ip = {ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false), nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, kFeatDim, N);
        ggml_set_input(x);
        ggml_tensor* h = sanm_layer(c, "encoder.encoders0.0.", x, N, false);
        for (int i = 0; i < num_blocks - 1; i++) h = sanm_layer(c, "encoder.encoders." + std::to_string(i) + ".", h, N, true);
        h = lnorm(c, h, g("encoder.after_norm.weight"), g("encoder.after_norm.bias"));
        for (int i = 0; i < tp_blocks; i++) h = sanm_layer(c, "encoder.tp_encoders." + std::to_string(i) + ".", h, N, true);
        h = lnorm(c, h, g("encoder.tp_norm.weight"), g("encoder.tp_norm.bias"));
        ggml_tensor* logits = lin(c, g("ctc.ctc_lo.weight"), g("ctc.ctc_lo.bias"), h);  // [V, N]
        ggml_set_output(logits);
        ggml_cgraph* gf = ggml_new_graph_custom(c, kGraphSize, false);
        ggml_build_forward_expand(gf, logits);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { ggml_free(c); return false; }
        ggml_backend_tensor_set(x, inp.data(), 0, ggml_nbytes(x));
        mintbot::cpu_backend_set_threads(backend, opts.n_threads);
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            std::vector<float> lg((size_t)vocab_size * N);
            ggml_backend_tensor_get(logits, lg.data(), 0, ggml_nbytes(logits));
            int prev = -1;
            for (int n = 0; n < N; n++) {
                const float* col = &lg[(size_t)n * vocab_size];
                int am = (int)(std::max_element(col, col + vocab_size) - col);
                if (am != prev && am != blank) ids.push_back(am);
                prev = am;
            }
        }
        ggml_free(c);
        return ok;
    }

    Transcript detok(const std::vector<int>& ids) {
        Transcript tr;
        std::string text, raw;
        for (int id : ids) {
            if (id < 0 || id >= (int)vocab.size()) continue;
            const std::string& piece = vocab[id];
            raw += piece;
            if (piece.size() >= 4 && piece.compare(0, 2, "<|") == 0) {
                std::string tag = piece.substr(2, piece.size() - 4);
                if (tr.emotion.empty() && in(tag, {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}))
                    tr.emotion = tag;
                else if (tr.event.empty() && in(tag, {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough",
                                                       "Sing", "Speech_Noise", "GBG", "Event_UNK"}))
                    tr.event = tag;
                else if (tr.language.empty() && !in(tag, {"withitn", "woitn"}))
                    tr.language = tag;
                continue;
            }
            text += piece;
        }
        const std::string sp = "\xe2\x96\x81";  // SentencePiece word boundary
        for (std::string* s : {&text, &raw})
            for (size_t pos; (pos = s->find(sp)) != std::string::npos;) s->replace(pos, sp.size(), " ");
        tr.text = trim(text);
        tr.raw = trim(raw);
        return tr;
    }
};

SenseVoice::SenseVoice() : p_(std::make_unique<Impl>()) {}
SenseVoice::~SenseVoice() = default;

bool SenseVoice::load(const std::string& path, const SenseVoiceOptions& opts) {
    Impl& m = *p_;
    m.opts = opts;
    int lid = lid_index(opts.language);
    if (lid < 0) { fprintf(stderr, "sensevoice: unknown language '%s'\n", opts.language.c_str()); return false; }

    gguf_init_params ip = {false, &m.ctx_w};
    gguf_context* gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) { fprintf(stderr, "sensevoice: cannot load %s\n", path.c_str()); return false; }
    auto rd = [&](const char* k, int d) { int64_t i = gguf_find_key(gg, k); return i < 0 ? d : (int)gguf_get_val_u32(gg, i); };
    m.d_model = rd("sv.output_size", 512); m.n_head = rd("sv.attention_heads", 4);
    m.num_blocks = rd("sv.num_blocks", 50); m.tp_blocks = rd("sv.tp_blocks", 20);
    m.kernel = rd("sv.kernel_size", 11); m.vocab_size = rd("sv.vocab_size", 25055); m.blank = rd("sv.blank_id", 0);
    int64_t ki = gguf_find_key(gg, "sv.vocab");
    if (ki >= 0) {
        size_t nv = gguf_get_arr_n(gg, ki);
        m.vocab.resize(nv);
        for (size_t i = 0; i < nv; i++) { const char* s = gguf_get_arr_str(gg, ki, i); m.vocab[i] = s ? s : ""; }
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++) {
        const char* nm = gguf_get_tensor_name(gg, i);
        m.t[nm] = ggml_get_tensor(m.ctx_w, nm);
    }
    gguf_free(gg);
    if (m.vocab.empty()) { fprintf(stderr, "sensevoice: %s has no sv.vocab\n", path.c_str()); return false; }

    // [language, event, emotion, text-norm] query rows, as SenseVoiceSmall.inference builds them.
    const int qtok[4] = {lid, 1, 2, opts.itn ? 14 : 15};
    m.nq = 4;
    const float* emb = (const float*)m.g("embed.weight")->data;  // [16, 560]
    m.query.resize((size_t)m.nq * kFeatDim);
    for (int i = 0; i < m.nq; i++) memcpy(&m.query[(size_t)i * kFeatDim], emb + (size_t)qtok[i] * kFeatDim, kFeatDim * sizeof(float));

    m.backend = mintbot::cpu_backend_init();
    m.galloc = m.backend ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend)) : nullptr;
    return m.backend && m.galloc;
}

void SenseVoice::set_abort_callback(ggml_abort_callback cb, void* data) {
    if (p_->backend) mintbot::cpu_backend_set_abort(p_->backend, cb, data);
}

Transcript SenseVoice::transcribe(const float* pcm, size_t n) {
    if (n < (size_t)kWinLen) return {};
    int T = 0;
    std::vector<float> fb = compute_fbank(pcm, n, T);
    std::vector<int> ids;
    if (T < 1 || !p_->encode(fb, T, ids)) return {};
    return p_->detok(ids);
}

}  // namespace mintbot::asr
