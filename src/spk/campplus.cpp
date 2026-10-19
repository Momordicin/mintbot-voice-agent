// CAM++ on ggml. Graph mirrors funasr/models/campplus/{model,components}.py.
//
// Derived work: a C++ / ggml re-implementation of the CAM++ network in FunASR
// (https://github.com/modelscope/FunASR, MIT, "Copyright FunASR"), which FunASR
// modified from 3D-Speaker (https://github.com/modelscope/3D-Speaker, Apache
// License 2.0, third_party/licenses/3D-Speaker-LICENSE.txt). Changed from the
// originals: rewritten in C++ on ggml for inference only, BatchNorm folded into
// a per-channel affine, Kaldi fbank front end reimplemented; see
// THIRD_PARTY_NOTICES.md.
//
// 1-D activations use the [channels, time] layout (ne0 = channels) so that
// 1x1 convolutions are plain mat-muls; BatchNorm is an exported per-channel
// affine (tools/export_campplus_gguf.py).
#include "spk/campplus.h"

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

namespace mintbot::spk {
namespace {

constexpr int kRate = 16000, kWin = 400, kShift = 160, kNfft = 512, kNmel = 80;
constexpr float kPreemph = 0.97f, kLowHz = 20.0f, kHighHz = 8000.0f;
constexpr int kGraph = 16384;
constexpr int kSegLen = 100;  // CAMLayer.seg_pooling

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

}  // namespace

struct CamPlus::Impl {
    ggml_context* ctx_w = nullptr;
    std::map<std::string, ggml_tensor*> t;
    int emb = 192, growth = 32, init_ch = 128;
    int layers[3] = {12, 24, 16}, kernel[3] = {3, 3, 3}, dil[3] = {1, 2, 2};
    int n_threads = 2;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<float> window;
    std::vector<std::vector<float>> melbank;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (backend) ggml_backend_free(backend);
        if (ctx_w) ggml_free(ctx_w);
    }

    ggml_tensor* g(const std::string& n) {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "campplus: missing tensor %s\n", n.c_str()); abort(); }
        return it->second;
    }

    // torchaudio.compliance.kaldi.fbank(num_mel_bins=80) defaults, then minus the window mean.
    std::vector<float> fbank(const float* x, size_t n, int& T) {
        T = n >= (size_t)kWin ? (int)((n - kWin) / kShift + 1) : 0;
        std::vector<float> feat((size_t)T * kNmel);
        std::vector<float> fr(kWin), re(kNfft), im(kNfft);
        const float eps = 1.1920929e-07f;
        for (int t = 0; t < T; t++) {
            const float* s = x + (size_t)t * kShift;
            double mn = 0;
            for (int i = 0; i < kWin; i++) mn += s[i];
            mn /= kWin;
            for (int i = 0; i < kWin; i++) fr[i] = s[i] - (float)mn;
            for (int i = kWin - 1; i > 0; i--) fr[i] -= kPreemph * fr[i - 1];
            fr[0] -= kPreemph * fr[0];
            for (int i = 0; i < kNfft; i++) { re[i] = i < kWin ? fr[i] * window[i] : 0.0f; im[i] = 0.0f; }
            fft(re, im, kNfft);
            for (int m = 0; m < kNmel; m++) {
                float e = 0;
                for (int k = 0; k < kNfft / 2 + 1; k++)
                    if (melbank[m][k] > 0) e += melbank[m][k] * (re[k] * re[k] + im[k] * im[k]);
                feat[(size_t)t * kNmel + m] = logf(std::max(e, eps));
            }
        }
        for (int m = 0; m < kNmel && T > 0; m++) {
            double mean = 0;
            for (int t2 = 0; t2 < T; t2++) mean += feat[(size_t)t2 * kNmel + m];
            mean /= T;
            for (int t2 = 0; t2 < T; t2++) feat[(size_t)t2 * kNmel + m] -= (float)mean;
        }
        return feat;
    }

    // ---- graph helpers ----
    ggml_tensor* bn1d(ggml_context* c, ggml_tensor* x, const std::string& p) {  // x [C, T]
        return ggml_add(c, ggml_mul(c, x, g(p + ".scale")), g(p + ".shift"));
    }
    ggml_tensor* bn2d(ggml_context* c, ggml_tensor* x, const std::string& p) {  // x [W, H, C, 1]
        ggml_tensor* s = g(p + ".scale");
        ggml_tensor* b = g(p + ".shift");
        return ggml_add(c, ggml_mul(c, x, ggml_reshape_3d(c, s, 1, 1, s->ne[0])), ggml_reshape_3d(c, b, 1, 1, b->ne[0]));
    }
    ggml_tensor* conv2d(ggml_context* c, ggml_tensor* x, const std::string& w, int stride_h, int pad) {
        return ggml_conv_2d(c, g(w), x, 1, stride_h, pad, pad, 1, 1);
    }
    // 1-D convolution on [Cin, T]; weight (K, Cout, Cin) = ggml [Cin, Cout, K], or [Cin, Cout] for K = 1.
    ggml_tensor* conv1d(ggml_context* c, ggml_tensor* x, const std::string& w, int stride, int pad, int d,
                        const std::string& bias = "") {
        ggml_tensor* W = g(w);
        ggml_tensor* y;
        if (ggml_n_dims(W) <= 2) {
            y = ggml_mul_mat(c, W, x);
        } else {
            const int K = (int)W->ne[2], T = (int)x->ne[1];
            ggml_tensor* xp = pad ? ggml_pad_ext(c, x, 0, 0, pad, pad, 0, 0, 0, 0) : x;
            const int T_full = T + 2 * pad - d * (K - 1);
            y = nullptr;
            for (int k = 0; k < K; k++) {
                ggml_tensor* wk = ggml_view_2d(c, W, W->ne[0], W->ne[1], W->nb[1], (size_t)k * W->nb[2]);
                ggml_tensor* xk = ggml_view_2d(c, xp, xp->ne[0], T_full, xp->nb[1], (size_t)k * d * xp->nb[1]);
                ggml_tensor* term = ggml_mul_mat(c, wk, xk);
                y = y ? ggml_add(c, y, term) : term;
            }
            if (stride > 1) {
                const int T_out = (T_full + stride - 1) / stride;
                y = ggml_cont(c, ggml_view_2d(c, y, y->ne[0], T_out, y->nb[1] * stride, 0));
            }
        }
        if (!bias.empty()) y = ggml_add(c, y, g(bias));
        return y;
    }
    ggml_tensor* time_mean(ggml_context* c, ggml_tensor* x) {  // [C, T] -> [C, 1]
        ggml_tensor* m = ggml_mean(c, ggml_cont(c, ggml_transpose(c, x)));  // [1, C]
        return ggml_reshape_2d(c, m, x->ne[0], 1);
    }

    ggml_tensor* basic_block(ggml_context* c, ggml_tensor* x, const std::string& p, int stride) {
        ggml_tensor* out = ggml_relu(c, bn2d(c, conv2d(c, x, p + ".conv1.weight", stride, 1), p + ".bn1"));
        out = bn2d(c, conv2d(c, out, p + ".conv2.weight", 1, 1), p + ".bn2");
        ggml_tensor* sc = x;
        if (t.count(p + ".shortcut.0.weight")) sc = bn2d(c, conv2d(c, x, p + ".shortcut.0.weight", stride, 0), p + ".shortcut.1");
        return ggml_relu(c, ggml_add(c, out, sc));
    }

    ggml_tensor* cam_layer(ggml_context* c, ggml_tensor* x, const std::string& p, int d) {
        ggml_tensor* y = conv1d(c, x, p + ".linear_local.weight", 1, d, d);
        const int C = (int)x->ne[0], T = (int)x->ne[1];
        ggml_tensor* mean = time_mean(c, x);
        ggml_tensor* context;
        if (T <= kSegLen) {
            context = ggml_scale(c, mean, 2.0f);  // global mean + the single segment's mean
        } else {
            ggml_tensor* seg = nullptr;
            for (int s = 0; s < T; s += kSegLen) {
                const int len = std::min(kSegLen, T - s);
                ggml_tensor* v = ggml_view_2d(c, x, C, len, x->nb[1], (size_t)s * x->nb[1]);
                ggml_tensor* r = ggml_repeat(c, time_mean(c, ggml_cont(c, v)), ggml_new_tensor_2d(c, GGML_TYPE_F32, C, len));
                seg = seg ? ggml_concat(c, seg, r, 1) : r;
            }
            context = ggml_add(c, seg, mean);
        }
        context = ggml_relu(c, conv1d(c, context, p + ".linear1.weight", 1, 0, 1, p + ".linear1.bias"));
        ggml_tensor* m = ggml_sigmoid(c, conv1d(c, context, p + ".linear2.weight", 1, 0, 1, p + ".linear2.bias"));
        return ggml_mul(c, y, m);
    }

    bool forward(const std::vector<float>& feat, int T, std::vector<float>& out) {
        ggml_init_params ip = {ggml_tensor_overhead() * kGraph + ggml_graph_overhead_custom(kGraph, false), nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, T, kNmel, 1, 1);  // [W = time, H = freq, C, N]
        ggml_set_input(x);

        // FCM head
        ggml_tensor* h = ggml_relu(c, bn2d(c, conv2d(c, x, "head.conv1.weight", 1, 1), "head.bn1"));
        h = basic_block(c, h, "head.layer1.0", 2);
        h = basic_block(c, h, "head.layer1.1", 1);
        h = basic_block(c, h, "head.layer2.0", 2);
        h = basic_block(c, h, "head.layer2.1", 1);
        h = ggml_relu(c, bn2d(c, conv2d(c, h, "head.conv2.weight", 2, 1), "head.bn2"));  // [T, F/8, 32, 1]
        h = ggml_reshape_2d(c, h, h->ne[0], h->ne[1] * h->ne[2]);                       // [T, 32 * F/8]
        h = ggml_cont(c, ggml_transpose(c, h));                                         // [C, T]

        // TDNN (k5, stride 2) + three CAM dense blocks with transit layers
        h = ggml_relu(c, bn1d(c, conv1d(c, h, "xvector.tdnn.linear.weight", 2, 2, 1), "xvector.tdnn.nonlinear.batchnorm"));
        for (int b = 0; b < 3; b++) {
            const std::string bp = "xvector.block" + std::to_string(b + 1);
            for (int l = 0; l < layers[b]; l++) {
                const std::string lp = bp + ".tdnnd" + std::to_string(l + 1);
                ggml_tensor* y = ggml_relu(c, bn1d(c, h, lp + ".nonlinear1.batchnorm"));
                y = conv1d(c, y, lp + ".linear1.weight", 1, 0, 1);
                y = ggml_relu(c, bn1d(c, y, lp + ".nonlinear2.batchnorm"));
                y = cam_layer(c, y, lp + ".cam_layer", dil[b] * ((kernel[b] - 1) / 2));
                h = ggml_concat(c, h, y, 0);
            }
            const std::string tp = "xvector.transit" + std::to_string(b + 1);
            h = conv1d(c, ggml_relu(c, bn1d(c, h, tp + ".nonlinear.batchnorm")), tp + ".linear.weight", 1, 0, 1);
        }
        h = ggml_relu(c, bn1d(c, h, "xvector.out_nonlinear.batchnorm"));

        // statistics pooling (mean, unbiased std) + dense
        const int C = (int)h->ne[0], Tn = (int)h->ne[1];
        ggml_tensor* ht = ggml_cont(c, ggml_transpose(c, h));                  // [T, C]
        ggml_tensor* mu = ggml_mean(c, ht);                                     // [1, C]
        ggml_tensor* dv = ggml_sub(c, ht, ggml_repeat(c, mu, ht));
        ggml_tensor* sd = ggml_sqrt(c, ggml_scale(c, ggml_sum_rows(c, ggml_sqr(c, dv)), 1.0f / std::max(1, Tn - 1)));
        ggml_tensor* stats = ggml_concat(c, ggml_reshape_2d(c, mu, C, 1), ggml_reshape_2d(c, sd, C, 1), 0);  // [2C, 1]
        ggml_tensor* e = bn1d(c, conv1d(c, stats, "xvector.dense.linear.weight", 1, 0, 1), "xvector.dense.nonlinear.batchnorm");
        ggml_set_output(e);

        ggml_cgraph* gf = ggml_new_graph_custom(c, kGraph, false);
        ggml_build_forward_expand(gf, e);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { ggml_free(c); return false; }
        std::vector<float> xin((size_t)T * kNmel);  // [t + T * m]
        for (int tt = 0; tt < T; tt++)
            for (int m = 0; m < kNmel; m++) xin[(size_t)m * T + tt] = feat[(size_t)tt * kNmel + m];
        ggml_backend_tensor_set(x, xin.data(), 0, ggml_nbytes(x));
        mintbot::cpu_backend_set_threads(backend, n_threads);
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            out.resize(emb);
            ggml_backend_tensor_get(e, out.data(), 0, ggml_nbytes(e));
        }
        ggml_free(c);
        return ok;
    }
};

CamPlus::CamPlus() : p_(std::make_unique<Impl>()) {}
CamPlus::~CamPlus() = default;
int CamPlus::dim() const { return p_->emb; }

bool CamPlus::load(const std::string& path, int n_threads) {
    Impl& m = *p_;
    m.n_threads = n_threads;
    gguf_init_params ip = {false, &m.ctx_w};
    gguf_context* gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) { fprintf(stderr, "campplus: cannot load %s\n", path.c_str()); return false; }
    auto rd = [&](const char* k, int d) { int64_t i = gguf_find_key(gg, k); return i < 0 ? d : (int)gguf_get_val_u32(gg, i); };
    m.emb = rd("campplus.embedding_size", 192);
    auto arr = [&](const char* k, int* dst) {
        int64_t i = gguf_find_key(gg, k);
        if (i >= 0 && gguf_get_arr_n(gg, i) == 3)
            for (int j = 0; j < 3; j++) dst[j] = ((const int32_t*)gguf_get_arr_data(gg, i))[j];
    };
    arr("campplus.block_layers", m.layers);
    arr("campplus.block_kernel", m.kernel);
    arr("campplus.block_dilation", m.dil);
    for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++) {
        const char* nm = gguf_get_tensor_name(gg, i);
        m.t[nm] = ggml_get_tensor(m.ctx_w, nm);
    }
    gguf_free(gg);
    if (!m.t.count("head.conv1.weight") || !m.t.count("xvector.dense.linear.weight")) {
        fprintf(stderr, "campplus: %s is not a CAM++ GGUF\n", path.c_str());
        return false;
    }
    m.window.resize(kWin);
    for (int i = 0; i < kWin; i++) m.window[i] = (float)pow(0.5 - 0.5 * cos(2.0 * M_PI * i / (kWin - 1)), 0.85);  // povey
    const int nb = kNfft / 2 + 1;
    float bw = (float)kRate / kNfft, ml = mel(kLowHz), mh = mel(kHighHz), dm = (mh - ml) / (kNmel + 1);
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
    return m.backend && m.galloc;
}

void CamPlus::set_abort_callback(ggml_abort_callback cb, void* data) {
    if (p_->backend) mintbot::cpu_backend_set_abort(p_->backend, cb, data);
}

bool CamPlus::embed_window(const float* pcm, size_t n, std::vector<float>& emb) {
    int T = 0;
    auto feat = p_->fbank(pcm, n, T);
    if (T < 8) return false;
    return p_->forward(feat, T, emb);
}

std::vector<std::pair<size_t, size_t>> sv_chunk_windows(size_t n) {
    std::vector<std::pair<size_t, size_t>> w;
    size_t last_end = 0;
    for (size_t st = 0; st < n; st += kSvShift) {
        size_t ed = std::min(st + kSvWindow, n);
        if (ed <= last_end) break;
        last_end = ed;
        w.emplace_back(ed > kSvWindow ? ed - kSvWindow : 0, ed);
    }
    return w;
}

bool CamPlus::embed_utterance(const float* pcm, size_t n, std::vector<std::vector<float>>& embs, const WindowCache* pre,
                              int* reused) {
    static_assert(kSvWindow == (size_t)(1.5 * kRate) && kSvShift == (size_t)(0.75 * kRate), "sv_chunk sizes");
    std::vector<float> buf(kSvWindow);
    for (auto [s0, ed] : sv_chunk_windows(n)) {
        if (pre && ed - s0 == kSvWindow) {
            auto it = pre->find(s0);
            if (it != pre->end()) {
                embs.push_back(it->second);
                if (reused) ++*reused;
                continue;
            }
        }
        std::fill(buf.begin(), buf.end(), 0.0f);
        std::copy(pcm + s0, pcm + ed, buf.begin());
        std::vector<float> e;
        if (!embed_window(buf.data(), buf.size(), e)) return false;
        embs.push_back(std::move(e));
    }
    return !embs.empty();
}

std::vector<float> average_normalized(const std::vector<std::vector<float>>& embs) {
    if (embs.empty()) return {};
    std::vector<float> acc(embs[0].size(), 0.0f);
    for (const auto& e : embs) {
        double nn = 0;
        for (float v : e) nn += (double)v * v;
        float inv = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0.0f;
        for (size_t i = 0; i < e.size(); i++) acc[i] += e[i] * inv;
    }
    double nn = 0;
    for (float v : acc) nn += (double)v * v;
    float inv = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0.0f;
    for (float& v : acc) v *= inv;
    return acc;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); i++) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return (aa > 0 && bb > 0) ? (float)(ab / sqrt(aa * bb)) : 0.0f;
}

}  // namespace mintbot::spk
