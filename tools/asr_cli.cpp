// asr-cli: transcribe one 16 kHz mono PCM16 WAV (one utterance) with SenseVoice.
//
//   asr-cli -m sensevoice.gguf -a audio.wav [--language auto] [--no-itn] [--threads 4] [--raw]
#include "asr/sensevoice.h"
#include "common/wav_io.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string model, wav;
    bool raw = false;
    mintbot::asr::SenseVoiceOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-a") wav = next();
        else if (a == "--language") opts.language = next();
        else if (a == "--no-itn") opts.itn = false;
        else if (a == "--threads") opts.n_threads = atoi(next());
        else if (a == "--raw") raw = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || wav.empty()) {
        fprintf(stderr, "usage: %s -m sensevoice.gguf -a audio.wav [--language L] [--no-itn] [--threads N] [--raw]\n", argv[0]);
        return 2;
    }
    std::vector<float> pcm;
    if (!read_wav16k(wav.c_str(), pcm)) { fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", wav.c_str()); return 1; }
    mintbot::asr::SenseVoice sv;
    if (!sv.load(model, opts)) return 1;
    auto t = sv.transcribe(pcm.data(), pcm.size());
    printf("%s\n", raw ? t.raw.c_str() : t.text.c_str());
    return 0;
}
