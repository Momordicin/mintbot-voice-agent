// apm-cli: run a 16 kHz mono PCM16 WAV through the WebRTC APM front end in
// fixed-size chunks and write the processed audio as a PCM16 WAV.
//
//   apm-cli -a in.wav -o out.wav [--chunk-ms 10] [--ns 0-4] [--no-agc] [--no-hpf]
#include "common/wav_io.h"
#include "dsp/apm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string in, out;
    int chunk_ms = 10;
    mintbot::dsp::ApmOptions opts;
    opts.enabled = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "-a") in = next();
        else if (a == "-o") out = next();
        else if (a == "--chunk-ms") chunk_ms = atoi(next());
        else if (a == "--ns") opts.ns_level = atoi(next());
        else if (a == "--no-agc") opts.agc2 = false;
        else if (a == "--no-hpf") opts.high_pass = false;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (in.empty() || out.empty() || chunk_ms < 1) {
        fprintf(stderr, "usage: %s -a in.wav -o out.wav [--chunk-ms N] [--ns 0-4] [--no-agc] [--no-hpf]\n", argv[0]);
        return 2;
    }
    std::vector<float> pcm, res;
    if (!read_wav16k(in.c_str(), pcm)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", in.c_str()); return 1; }
    mintbot::dsp::AudioPreprocessor apm;
    std::string err;
    if (!apm.init(opts, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    const size_t step = (size_t)chunk_ms * 16;
    for (size_t i = 0; i < pcm.size(); i += step) apm.process(pcm.data() + i, std::min(step, pcm.size() - i), res);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!write_wav16k(out.c_str(), res)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    fprintf(stderr, "{\"in_samples\": %zu, \"out_samples\": %zu, \"seconds\": %.3f}\n", pcm.size(), res.size(), s);
    return 0;
}
