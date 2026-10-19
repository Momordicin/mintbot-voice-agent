// spk-cli: CAM++ speaker embeddings for a 16 kHz mono PCM16 WAV.
//
//   spk-cli -m campplus.gguf -a audio.wav              one JSON line per sv_chunk window
//   spk-cli -m campplus.gguf -a audio.wav --vad fsmn-vad.gguf
//           one JSON line per VAD utterance (start/end ms + averaged, normalised embedding)
//   --apm [--ns N] [--no-agc]  run the agent's WebRTC preprocessing first (for raw --record files)
#include "common/wav_io.h"
#include "dsp/apm.h"
#include "spk/campplus.h"
#include "vad/fsmn_vad_stream.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_vec(const std::vector<float>& v) {
    printf("[");
    for (size_t i = 0; i < v.size(); i++) printf("%s%.7g", i ? "," : "", v[i]);
    printf("]");
}

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string model, wav, vad_path;
    int threads = 4;
    mintbot::dsp::ApmOptions apm_opts;  // --apm: same preprocessing as the agent before VAD / embedding
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-a") wav = next();
        else if (a == "--vad") vad_path = next();
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--apm") apm_opts.enabled = true;
        else if (a == "--ns") apm_opts.ns_level = atoi(next());
        else if (a == "--no-agc") apm_opts.agc2 = false;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || wav.empty()) {
        fprintf(stderr, "usage: %s -m campplus.gguf -a audio.wav [--vad fsmn-vad.gguf] [--apm [--ns N] [--no-agc]] [--threads N]\n",
                argv[0]);
        return 2;
    }
    std::vector<float> pcm;
    if (!read_wav16k(wav.c_str(), pcm)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", wav.c_str()); return 1; }
    if (apm_opts.enabled) {
        mintbot::dsp::AudioPreprocessor apm;
        std::string err;
        if (!apm.init(apm_opts, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<float> processed;
        const size_t step = 160;  // 10 ms, as captured
        for (size_t i = 0; i < pcm.size(); i += step) apm.process(pcm.data() + i, std::min(step, pcm.size() - i), processed);
        pcm.swap(processed);
    }
    mintbot::spk::CamPlus cam;
    if (!cam.load(model, threads)) return 1;

    if (vad_path.empty()) {
        std::vector<std::vector<float>> embs;
        if (!cam.embed_utterance(pcm.data(), pcm.size(), embs)) return 1;
        for (auto& e : embs) { printf("{\"embedding\": "); print_vec(e); printf("}\n"); }
        return 0;
    }
    mintbot::vad::FsmnVadStream vad;
    if (!vad.load(vad_path)) return 1;
    std::vector<mintbot::vad::VadEvent> ev = vad.feed(pcm.data(), pcm.size());
    for (auto& e : vad.flush()) ev.push_back(e);
    int64_t start = -1;
    for (auto& e : ev) {
        if (e.type == mintbot::vad::VadEvent::SpeechStart) { start = e.ms; continue; }
        if (start < 0) continue;
        size_t s = (size_t)start * 16, t = std::min((size_t)e.ms * 16, pcm.size());
        std::vector<std::vector<float>> embs;
        if (t > s && cam.embed_utterance(pcm.data() + s, t - s, embs)) {
            printf("{\"start_ms\": %lld, \"end_ms\": %lld, \"windows\": %zu, \"embedding\": ", (long long)start,
                   (long long)e.ms, embs.size());
            print_vec(mintbot::spk::average_normalized(embs));
            printf("}\n");
        }
        start = -1;
    }
    return 0;
}
