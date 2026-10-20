// aec-offline: run a microphone recording and its render reference (both 16 kHz
// mono PCM16 WAV, sample-aligned at the start, e.g. from aec-rec) through the
// WebRTC APM with AEC3, 10 ms at a time (reference frame first, then the capture
// frame), and write the processed audio.
//
//   aec-offline --mic mic.wav --ref ref.wav -o out.wav [--linear lin.wav]
//               [--stats stats.jsonl] [--ref-shift-ms N] [--delay-hint MS]
//               [--no-aec] [--ns 0-4] [--no-agc] [--no-hpf]
//
//   --ref-shift-ms N  delay the reference by N ms (negative: advance it), to test
//                     how AEC3 copes with misaligned or late references
//   --stats           one JSON line per 100 ms: AEC3 statistics and the CPU time
//                     spent on that 100 ms
// A summary line (CPU per 10 ms frame, real-time factor) goes to stderr.
#include "common/wav_io.h"
#include "dsp/apm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void put_opt(FILE* f, const char* key, const std::optional<double>& v) {
    if (v) fprintf(f, ", \"%s\": %.3f", key, *v);
    else fprintf(f, ", \"%s\": null", key);
}

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string mic_path, ref_path, out_path, lin_path, stats_path;
    int shift_ms = 0, delay_hint = -1;
    mintbot::dsp::ApmOptions opts;
    opts.enabled = true;
    opts.aec = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--mic") mic_path = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "--linear") lin_path = next();
        else if (a == "--stats") stats_path = next();
        else if (a == "--ref-shift-ms") shift_ms = atoi(next());
        else if (a == "--delay-hint") delay_hint = atoi(next());
        else if (a == "--no-aec") opts.aec = false;
        else if (a == "--ns") opts.ns_level = atoi(next());
        else if (a == "--no-agc") opts.agc2 = false;
        else if (a == "--no-hpf") opts.high_pass = false;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (mic_path.empty() || ref_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s --mic mic.wav --ref ref.wav -o out.wav [--linear lin.wav] [--stats stats.jsonl] "
                        "[--ref-shift-ms N] [--delay-hint MS] [--no-aec] [--ns 0-4] [--no-agc] [--no-hpf]\n", argv[0]);
        return 2;
    }
    if (!lin_path.empty() && !opts.aec) { fprintf(stderr, "--linear needs AEC (drop --no-aec)\n"); return 2; }
    opts.aec_linear_output = !lin_path.empty();

    std::vector<float> mic, ref;
    if (!read_wav16k(mic_path.c_str(), mic)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", mic_path.c_str()); return 1; }
    if (!read_wav16k(ref_path.c_str(), ref)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", ref_path.c_str()); return 1; }
    if (shift_ms > 0) ref.insert(ref.begin(), (size_t)shift_ms * 16, 0.0f);
    else if (shift_ms < 0) ref.erase(ref.begin(), ref.begin() + std::min(ref.size(), (size_t)-shift_ms * 16));
    ref.resize(mic.size(), 0.0f);  // silence once the reference runs out

    mintbot::dsp::AudioPreprocessor apm;
    std::string err;
    if (!apm.init(opts, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (delay_hint >= 0) apm.set_stream_delay_ms(delay_hint);

    FILE* st = nullptr;
    if (!stats_path.empty() && !(st = mintbot::fopen_utf8(stats_path, "wb"))) {
        fprintf(stderr, "cannot write %s\n", stats_path.c_str());
        return 1;
    }

    constexpr size_t kFrame = 160;
    std::vector<float> out, lin;
    out.reserve(mic.size());
    std::vector<double> frame_us;
    double block_us = 0;
    using clk = std::chrono::steady_clock;
    for (size_t i = 0; i + kFrame <= mic.size(); i += kFrame) {
        auto t0 = clk::now();
        apm.process_reverse(ref.data() + i, kFrame);
        apm.process(mic.data() + i, kFrame, out, lin_path.empty() ? nullptr : &lin);
        double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
        frame_us.push_back(us);
        block_us += us;
        if (st && frame_us.size() % 10 == 0) {
            auto s = apm.aec_stats();
            fprintf(st, "{\"t_ms\": %zu, \"cpu_us\": %.0f", (i + kFrame) / 16, block_us);
            put_opt(st, "erl", s.echo_return_loss);
            put_opt(st, "erle", s.echo_return_loss_enhancement);
            put_opt(st, "divergent", s.divergent_filter_fraction);
            put_opt(st, "residual_echo", s.residual_echo_likelihood);
            if (s.delay_ms) fprintf(st, ", \"delay_ms\": %d}\n", (int)*s.delay_ms);
            else fprintf(st, ", \"delay_ms\": null}\n");
            block_us = 0;
        }
    }
    if (st) fclose(st);
    if (!write_wav16k(out_path.c_str(), out)) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    if (!lin_path.empty() && !write_wav16k(lin_path.c_str(), lin)) { fprintf(stderr, "cannot write %s\n", lin_path.c_str()); return 1; }

    double total = 0;
    for (double u : frame_us) total += u;
    std::vector<double> sorted = frame_us;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted.empty() ? 0.0 : sorted[std::min(sorted.size() - 1, (size_t)(p * sorted.size()))]; };
    double audio_s = frame_us.size() * 0.01;
    fprintf(stderr,
            "{\"aec\": %s, \"frames\": %zu, \"audio_s\": %.2f, \"cpu_s\": %.3f, \"rtf\": %.4f, "
            "\"frame_us_p50\": %.1f, \"frame_us_p99\": %.1f, \"frame_us_max\": %.1f}\n",
            opts.aec ? "true" : "false", frame_us.size(), audio_s, total / 1e6, audio_s > 0 ? total / 1e6 / audio_s : 0.0,
            pct(0.5), pct(0.99), sorted.empty() ? 0.0 : sorted.back());
    return 0;
}
