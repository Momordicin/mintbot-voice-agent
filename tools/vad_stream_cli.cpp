// vad-stream-cli: feed a 16 kHz mono PCM16 WAV to FsmnVadStream in fixed-size
// chunks and print the resulting speech segments as JSON.
//
//   vad-stream-cli -m fsmn-vad.gguf -a audio.wav [--chunk-ms 200] [--end-sil 800]
//                  [--max-seg 30000] [--thres 0.6] [--probs-out probs.f32]
#include "vad/fsmn_vad_stream.h"
#include "common/wav_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string model, wav, probs_out;
    int chunk_ms = 200;
    mintbot::vad::FsmnVadOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-a") wav = next();
        else if (a == "--chunk-ms") chunk_ms = atoi(next());
        else if (a == "--end-sil") opts.max_end_silence_ms = atoi(next());
        else if (a == "--max-seg") opts.max_single_segment_ms = atoi(next());
        else if (a == "--thres") opts.speech_noise_thres = (float)atof(next());
        else if (a == "--idle-reset-ms") opts.idle_reset_ms = atoll(next());
        else if (a == "--probs-out") probs_out = next();
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || wav.empty() || chunk_ms < 1) {
        fprintf(stderr, "usage: %s -m fsmn-vad.gguf -a audio.wav [--chunk-ms N] [--end-sil MS] [--max-seg MS] "
                        "[--thres P] [--idle-reset-ms MS] [--probs-out FILE]\n", argv[0]);
        return 2;
    }
    std::vector<float> pcm;
    if (!read_wav16k(wav.c_str(), pcm)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", wav.c_str()); return 1; }

    mintbot::vad::FsmnVadStream vad;
    if (!vad.load(model, opts)) return 1;
    vad.record_probs(!probs_out.empty());

    std::vector<std::pair<int64_t, int64_t>> segs;
    int64_t open = -1;
    auto take = [&](const std::vector<mintbot::vad::VadEvent>& ev) {
        for (auto& e : ev) {
            if (e.type == mintbot::vad::VadEvent::SpeechStart) open = e.ms;
            else { segs.push_back({open, e.ms}); open = -1; }
        }
    };
    const size_t step = (size_t)chunk_ms * 16;
    for (size_t i = 0; i < pcm.size(); i += step) take(vad.feed(pcm.data() + i, std::min(step, pcm.size() - i)));
    take(vad.flush());

    printf("{\"segments\": [");
    for (size_t i = 0; i < segs.size(); i++) printf("%s[%lld, %lld]", i ? ", " : "", (long long)segs[i].first, (long long)segs[i].second);
    printf("]}\n");

    if (!probs_out.empty()) {
        const auto& p = vad.recorded_probs();
        FILE* f = mintbot::fopen_utf8(probs_out, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", probs_out.c_str()); return 1; }
        fwrite(p.data(), sizeof(float), p.size(), f);
        fclose(f);
    }
    return 0;
}
