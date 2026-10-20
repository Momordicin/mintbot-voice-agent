// aec-rec: record what echo cancellation needs, for offline experiments. Three
// streams run at once, all as 16 kHz mono float (miniaudio converts):
//   mic.wav       the microphone
//   loopback.wav  the system mix on the default output device (WASAPI loopback):
//                 every app's sound, the AEC3 reference
//   played.wav    exactly what this program sent to the default output: a test
//                 signal (--play), or silence, which also keeps the loopback
//                 stream delivering packets while nothing else plays
// Each file starts at the same moment, estimated from the first callback of each
// stream. timing.csv logs every callback (stream, microseconds since start,
// frames) so clock drift and gaps can be measured; info.json has the device
// formats. Recording runs until Enter (or Ctrl+C), then the files are written.
//
//   aec-rec --list
//   aec-rec --out DIR [--mic N] [--play test.wav [--loop]] [--no-play]
#include "common/utf8.h"
#include "common/wav_io.h"

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
constexpr double kRate = 16000.0;

std::atomic<bool> g_stop{false};
clk::time_point g_t0;

struct Tick {
    int64_t t_us;
    uint32_t frames;
};

// One recorded stream. The audio thread appends to `in` under the lock; the main
// thread moves it to `all` every few tens of milliseconds, so the audio thread
// never waits on a large reallocation.
struct Stream {
    const char* name;
    std::mutex m;
    std::vector<float> in;
    std::vector<Tick> ticks;
    std::vector<float> all;
    double sumsq = 0;  // since the last status line
    size_t count = 0;
    int64_t total = 0;

    explicit Stream(const char* n) : name(n) { in.reserve(16000); }

    void push(const float* pcm, uint32_t frames) {
        int64_t t = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - g_t0).count();
        std::lock_guard<std::mutex> lk(m);
        if (pcm) in.insert(in.end(), pcm, pcm + frames);
        else in.insert(in.end(), frames, 0.0f);
        ticks.push_back({t, frames});
        for (uint32_t i = 0; pcm && i < frames; i++) sumsq += (double)pcm[i] * pcm[i];
        count += frames;
        total += frames;
    }
    void drain() {
        std::lock_guard<std::mutex> lk(m);
        all.insert(all.end(), in.begin(), in.end());
        in.clear();
    }
    // dBFS since the last call, and the sample total
    double level(int64_t* n) {
        std::lock_guard<std::mutex> lk(m);
        double db = count ? 10 * std::log10(sumsq / count + 1e-12) : -120.0;
        sumsq = 0;
        count = 0;
        *n = total;
        return db;
    }
    // When the first sample was captured or requested: the first callback's time
    // minus the duration of its frames.
    double start_us() const { return ticks.empty() ? 1e18 : ticks[0].t_us - ticks[0].frames * 1e6 / kRate; }
};

Stream g_mic("mic"), g_loop("loopback"), g_played("played");

std::vector<float> g_play;
size_t g_play_pos = 0;
bool g_loop_play = false;

void on_mic(ma_device*, void*, const void* input, ma_uint32 frames) { g_mic.push((const float*)input, frames); }
void on_loopback(ma_device*, void*, const void* input, ma_uint32 frames) { g_loop.push((const float*)input, frames); }
void on_playback(ma_device*, void* output, const void*, ma_uint32 frames) {
    float* out = (float*)output;
    for (ma_uint32 i = 0; i < frames; i++) {
        if (g_play_pos >= g_play.size() && g_loop_play && !g_play.empty()) g_play_pos = 0;
        out[i] = g_play_pos < g_play.size() ? g_play[g_play_pos++] : 0.0f;
    }
    g_played.push(out, frames);
}

#ifdef _WIN32
BOOL WINAPI on_ctrl(DWORD) {
    g_stop = true;
    return TRUE;
}
#endif

std::string dev_name(ma_device* d, ma_device_type t) {
    char nm[256] = {0};
    ma_device_get_name(d, t, nm, sizeof nm, nullptr);
    return nm;
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

}  // namespace

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string out_dir, play_path;
    int mic_index = -1;
    bool list = false, no_play = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--out") out_dir = next();
        else if (a == "--mic") mic_index = atoi(next());
        else if (a == "--play") play_path = next();
        else if (a == "--loop") g_loop_play = true;
        else if (a == "--no-play") no_play = true;
        else if (a == "--list") list = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (!list && out_dir.empty()) {
        fprintf(stderr, "usage: %s --list | --out DIR [--mic N] [--play test.wav [--loop]] [--no-play]\n", argv[0]);
        return 2;
    }
    if (no_play && !play_path.empty()) { fprintf(stderr, "--play and --no-play exclude each other\n"); return 2; }

    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) { fprintf(stderr, "ma_context_init failed\n"); return 1; }
    ma_device_info *pinfo = nullptr, *cinfo = nullptr;
    ma_uint32 np = 0, nc = 0;
    if (ma_context_get_devices(&ctx, &pinfo, &np, &cinfo, &nc) != MA_SUCCESS) { fprintf(stderr, "ma_context_get_devices failed\n"); return 1; }
    if (list) {
        printf("capture devices (--mic N):\n");
        for (ma_uint32 i = 0; i < nc; i++) printf("  %u%s %s\n", i, cinfo[i].isDefault ? " *" : "  ", cinfo[i].name);
        printf("playback devices (recorded and played on: the default, *):\n");
        for (ma_uint32 i = 0; i < np; i++) printf("     %s %s\n", pinfo[i].isDefault ? "*" : " ", pinfo[i].name);
        ma_context_uninit(&ctx);
        return 0;
    }
    if (mic_index >= (int)nc) { fprintf(stderr, "capture device %d not found (see --list)\n", mic_index); return 1; }
    if (!play_path.empty() && !read_wav16k(play_path.c_str(), g_play)) {
        fprintf(stderr, "need a 16 kHz mono PCM16 wav: %s\n", play_path.c_str());
        return 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(mintbot::utf8_path(out_dir), ec);

    ma_device mic{}, loop{}, play{};
    ma_device_id mic_id{};
    auto config = [](ma_device_type t, ma_device_data_proc cb) {
        ma_device_config c = ma_device_config_init(t);
        c.capture.format = c.playback.format = ma_format_f32;
        c.capture.channels = c.playback.channels = 1;
        c.sampleRate = 16000;
        c.periodSizeInMilliseconds = 10;
        c.dataCallback = cb;
        return c;
    };
    ma_device_config mc = config(ma_device_type_capture, on_mic);
    if (mic_index >= 0) {
        mic_id = cinfo[mic_index].id;
        mc.capture.pDeviceID = &mic_id;
    }
    ma_device_config lc = config(ma_device_type_loopback, on_loopback);  // default output device
    ma_device_config pc = config(ma_device_type_playback, on_playback);
    if (ma_result r = ma_device_init(&ctx, &mc, &mic); r != MA_SUCCESS) { fprintf(stderr, "microphone: %s\n", ma_result_description(r)); return 1; }
    if (ma_result r = ma_device_init(&ctx, &lc, &loop); r != MA_SUCCESS) { fprintf(stderr, "loopback: %s\n", ma_result_description(r)); return 1; }
    if (!no_play) {
        if (ma_result r = ma_device_init(&ctx, &pc, &play); r != MA_SUCCESS) { fprintf(stderr, "playback: %s\n", ma_result_description(r)); return 1; }
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#endif
    g_t0 = clk::now();
    // loopback first: it must already be listening when the first played samples come out
    if (ma_device_start(&loop) != MA_SUCCESS || ma_device_start(&mic) != MA_SUCCESS ||
        (!no_play && ma_device_start(&play) != MA_SUCCESS)) {
        fprintf(stderr, "cannot start the devices\n");
        return 1;
    }
    fprintf(stderr, "microphone: %s\noutput:     %s\n", dev_name(&mic, ma_device_type_capture).c_str(),
            dev_name(&loop, ma_device_type_capture).c_str());
    fprintf(stderr, "recording; press Enter to stop (levels in dBFS every 2 s)\n");
    std::thread([] {
        std::string line;
        std::getline(std::cin, line);
        g_stop = true;
    }).detach();

    auto next_status = clk::now() + std::chrono::seconds(2);
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        g_mic.drain();
        g_loop.drain();
        g_played.drain();
        if (clk::now() >= next_status) {
            next_status += std::chrono::seconds(2);
            double el = std::chrono::duration<double>(clk::now() - g_t0).count();
            int64_t nm, nl, npl;
            double dm = g_mic.level(&nm), dl = g_loop.level(&nl), dp = g_played.level(&npl);
            fprintf(stderr, "%6.1f s  mic %6.1f dB (%.1f s)  loopback %6.1f dB (%.1f s)  played %6.1f dB (%.1f s)\n", el,
                    dm, nm / kRate, dl, nl / kRate, dp, npl / kRate);
        }
    }
    double seconds = std::chrono::duration<double>(clk::now() - g_t0).count();
    if (!no_play) ma_device_stop(&play);
    ma_device_stop(&mic);
    ma_device_stop(&loop);
    g_mic.drain();
    g_loop.drain();
    g_played.drain();

    Stream* streams[3] = {&g_mic, &g_loop, &g_played};
    double t0 = 1e18;
    for (Stream* s : streams)
        if (!s->ticks.empty()) t0 = std::min(t0, s->start_us());
    auto path = [&](const char* f) { return out_dir + "/" + f; };
    std::string info = "{\"seconds\": " + std::to_string(seconds) + ", \"streams\": {";
    for (int k = 0; k < 3; k++) {
        Stream* s = streams[k];
        if (no_play && s == &g_played) continue;
        size_t pad = s->ticks.empty() ? 0 : (size_t)std::llround((s->start_us() - t0) * kRate / 1e6);
        std::vector<float> pcm(pad, 0.0f);
        pcm.insert(pcm.end(), s->all.begin(), s->all.end());
        std::string f = std::string(s->name) + ".wav";
        if (!write_wav16k(path(f.c_str()).c_str(), pcm)) { fprintf(stderr, "cannot write %s\n", f.c_str()); return 1; }
        ma_device* d = s == &g_mic ? &mic : s == &g_loop ? &loop : &play;
        bool cap = s != &g_played;
        ma_uint32 rate = cap ? d->capture.internalSampleRate : d->playback.internalSampleRate;
        ma_uint32 chans = cap ? d->capture.internalChannels : d->playback.internalChannels;
        ma_uint32 period = cap ? d->capture.internalPeriodSizeInFrames : d->playback.internalPeriodSizeInFrames;
        ma_uint32 periods = cap ? d->capture.internalPeriods : d->playback.internalPeriods;
        info += std::string(k ? ", " : "") + "\"" + s->name + "\": {\"device\": " +
                json_str(dev_name(d, cap ? ma_device_type_capture : ma_device_type_playback)) +
                ", \"samples\": " + std::to_string(s->all.size()) + ", \"pad_samples\": " + std::to_string(pad) +
                ", \"callbacks\": " + std::to_string(s->ticks.size()) +
                ", \"internal_rate\": " + std::to_string(rate) +
                ", \"internal_channels\": " + std::to_string(chans) +
                ", \"internal_period_frames\": " + std::to_string(period) +
                ", \"internal_periods\": " + std::to_string(periods) + "}";
    }
    info += "}, \"play\": " + json_str(play_path) + ", \"loop\": " + (g_loop_play ? "true" : "false") + "}\n";
    FILE* f = mintbot::fopen_utf8(path("info.json"), "wb");
    if (f) { fputs(info.c_str(), f); fclose(f); }
    f = mintbot::fopen_utf8(path("timing.csv"), "wb");
    if (f) {
        fprintf(f, "stream,t_us,frames\n");
        for (Stream* s : streams)
            for (const Tick& t : s->ticks) fprintf(f, "%s,%lld,%u\n", s->name, (long long)t.t_us, t.frames);
        fclose(f);
    }
    if (!no_play) ma_device_uninit(&play);
    ma_device_uninit(&mic);
    ma_device_uninit(&loop);
    ma_context_uninit(&ctx);
    fprintf(stderr, "%.1f s written to %s\n", seconds, out_dir.c_str());
    return 0;
}
