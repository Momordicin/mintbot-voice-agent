// pipeline-cli: run a 16 kHz mono PCM16 WAV through VoicePipeline in small
// frames, as a microphone would deliver it, and print one JSON line per
// utterance plus a stats line on stderr.
//
//   pipeline-cli --vad fsmn-vad.gguf -m sensevoice.gguf -a audio.wav [--frame-ms 10]
//                [--language auto] [--no-itn] [--threads 4] [--end-sil 800]
//                [--apm [--ns 0-4] [--no-agc] [--no-hpf]]
#include "pipeline/pipeline.h"
#include "common/wav_io.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string vad_path, asr_path, wav;
    int frame_ms = 10;
    mintbot::pipeline::PipelineOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--vad") vad_path = next();
        else if (a == "-m") asr_path = next();
        else if (a == "-a") wav = next();
        else if (a == "--frame-ms") frame_ms = atoi(next());
        else if (a == "--language") opts.asr.language = next();
        else if (a == "--no-itn") opts.asr.itn = false;
        else if (a == "--threads") opts.asr.n_threads = atoi(next());
        else if (a == "--end-sil") opts.vad.max_end_silence_ms = atoi(next());
        else if (a == "--max-seg") opts.vad.max_single_segment_ms = atoi(next());
        else if (a == "--apm") opts.apm.enabled = true;
        else if (a == "--ns") opts.apm.ns_level = atoi(next());
        else if (a == "--no-agc") opts.apm.agc2 = false;
        else if (a == "--no-hpf") opts.apm.high_pass = false;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (vad_path.empty() || asr_path.empty() || wav.empty() || frame_ms < 1) {
        fprintf(stderr, "usage: %s --vad fsmn-vad.gguf -m sensevoice.gguf -a audio.wav [--frame-ms N] "
                        "[--language L] [--no-itn] [--threads N] [--end-sil MS] [--max-seg MS] "
                        "[--apm [--ns 0-4] [--no-agc] [--no-hpf]]\n", argv[0]);
        return 2;
    }
    std::vector<float> pcm;
    if (!read_wav16k(wav.c_str(), pcm)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", wav.c_str()); return 1; }

    mintbot::pipeline::VoicePipeline pipe;
    if (!pipe.load(vad_path, asr_path, opts)) return 1;

    size_t max_buf = 0;
    int n_utt = 0;
    double asr_total = 0;
    auto emit = [&](const std::vector<mintbot::pipeline::Result>& rs) {
        for (const auto& r : rs) {
            const auto& t = r.transcript;
            printf("{\"start_ms\": %lld, \"end_ms\": %lld, \"language\": %s, \"emotion\": %s, \"event\": %s, "
                   "\"text\": %s, \"asr_ms\": %.1f}\n",
                   (long long)r.start_ms, (long long)r.end_ms, json_str(t.language).c_str(), json_str(t.emotion).c_str(),
                   json_str(t.event).c_str(), json_str(t.text).c_str(), r.asr_ms);
            fflush(stdout);
            ++n_utt;
            asr_total += r.asr_ms;
        }
    };
    auto t0 = std::chrono::steady_clock::now();
    const size_t step = (size_t)frame_ms * 16;
    for (size_t i = 0; i < pcm.size(); i += step) {
        emit(pipe.feed(pcm.data() + i, std::min(step, pcm.size() - i)));
        max_buf = std::max(max_buf, pipe.segmenter().buffered_samples());
    }
    emit(pipe.flush());
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "{\"audio_s\": %.1f, \"wall_s\": %.2f, \"utterances\": %d, \"asr_s\": %.2f, "
                    "\"max_buffered_samples\": %zu, \"anomalies\": %lld}\n",
            pcm.size() / 16000.0, wall, n_utt, asr_total / 1000, max_buf, (long long)pipe.segmenter().anomalies());
    return 0;
}
