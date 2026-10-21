// mintbot-voice-agent: always-on voice input process for the MintBot core.
//
// Started by the core (non-detached child). Listens on loopback only, prints
// one ready line on stdout, then serves:
//   GET  /v1/health                       liveness (no token required)
//   GET  /v1/status                       engine state
//   GET  /v1/devices                      capture devices
//   POST /v1/listen/start[?device=N]      start capturing (default device if omitted)
//   POST /v1/listen/stop                  stop capturing, finish the open utterance
//   POST /v1/shutdown                     exit
//   GET  /v1/speaker                      owner voiceprint state (with --spk-model)
//   POST /v1/speaker/enroll/{start,commit,cancel}   enrollment from the next utterances
//   DELETE /v1/speaker                    forget the voiceprint (deletes the file)
//   GET  /v1/events                       SSE: status, speech_start, speech_end, final, speech_other,
//                                         enroll_progress, enroll_failed, speaker, device, error
// Every endpoint but /v1/health needs "Authorization: Bearer $MINTBOT_VOICE_TOKEN".
// The process exits when stdin reaches EOF (the parent went away), on
// /v1/shutdown, or on Ctrl+C. Logs go to stderr; stdout carries only the ready line.
#include "agent/agent.h"
#include "agent/audio_source.h"
#include "agent/event_hub.h"
#include "agent/json.h"
#include "dsp/apm.h"

#include "common/utf8.h"
#include "httplib/httplib.h"

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace mintbot;

namespace {

constexpr const char* kVersion = MINTBOT_VERSION;      // CMake project VERSION
constexpr int64_t kApiVersion = MINTBOT_API_VERSION;   // REST / SSE compatibility level

std::mutex g_quit_m;
std::condition_variable g_quit_cv;
bool g_quit = false;
std::string g_quit_reason;

void request_quit(const char* reason) {
    {
        std::lock_guard<std::mutex> lk(g_quit_m);
        if (g_quit) return;
        g_quit = true;
        g_quit_reason = reason;
    }
    g_quit_cv.notify_all();
}

void on_signal(int) { request_quit("signal"); }

bool same_secret(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

void reply_json(httplib::Response& res, int status, const std::string& json) {
    res.status = status;
    res.set_content(json, "application/json");
}

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s --vad fsmn-vad.gguf -m sensevoice.gguf [options]\n"
            "       %s --version   print {\"version\", \"api_version\"} and exit\n"
            "  --host H            bind address (default 127.0.0.1)\n"
            "  --port N            port, 0 = pick a free one (default 0)\n"
            "  --source S          mic (default) or wav:<path> (replayed in real time, for tests)\n"
            "  --replay-speed X    wav source speed factor (default 1)\n"
            "  --autostart         start listening immediately\n"
            "  --device N          capture device index for --autostart (default: system default)\n"
            "  --language L        auto (default), zh, en, yue, ja, ko\n"
            "  --no-itn            spoken-form output (no punctuation / digits)\n"
            "  --threads N         ASR threads (default 4)\n"
            "  --end-sil MS        silence that ends an utterance (default 800)\n"
            "  --max-seg MS        longest utterance (default 30000)\n"
            "  --early-asr MS      start recognising after this much silence, so the result is ready when the\n"
            "                      utterance ends (default 200, 0 = off; the output is the same either way)\n"
            "  --vad-block MS      audio per VAD call (default 50; larger is cheaper, events come up to MS later)\n"
            "  --no-apm            disable WebRTC preprocessing (high-pass, noise suppression, AGC2)\n"
            "  --ns N              noise suppression 0 off, 1 low, 2 moderate (default), 3 high, 4 very high\n"
            "  --no-agc            disable AGC2\n"
            "  --no-hpf            disable the high-pass filter\n"
            "  --record FILE       write the raw input (before preprocessing) to a 16 kHz WAV\n"
            "  --spk-model FILE    CAM++ GGUF: enables the owner-speaker gate\n"
            "  --speaker-db FILE   voiceprint file (created on enrollment; without it the voiceprint is not kept)\n"
            "  --speaker-threshold X  cosine threshold for the owner (default 0.5)\n"
            "  --speaker-min-ms MS    shorter utterances are 'uncertain' (default 1000)\n"
            "  --speaker-short-floor X  uncertain utterances scoring below X are dropped too (default 0.3)\n"
            "  --speaker-gate-short   drop every uncertain utterance below the threshold (default: transcribe them)\n"
            "  --speaker-observe      testing: transcribe every utterance, only report the speaker decision\n"
            "  --no-stdin-watch    do not exit when stdin closes\n"
            "  --no-auth           allow running without MINTBOT_VOICE_TOKEN (local debugging only)\n",
            argv0, argv0);
}

}  // namespace

int main(int argc, char** argv) {
    mintbot::use_utf8_argv(argc, argv);
    std::string vad_path, asr_path, host = "127.0.0.1";
    int port = 0, device = -1;
    bool autostart = false, stdin_watch = true, no_auth = false;
    agent::AgentOptions opts;
    opts.pipe.apm.enabled = dsp::AudioPreprocessor::available();  // on by default when built in
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--vad") vad_path = next();
        else if (a == "-m") asr_path = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = atoi(next());
        else if (a == "--source") opts.source = next();
        else if (a == "--replay-speed") opts.replay_speed = atof(next());
        else if (a == "--autostart") autostart = true;
        else if (a == "--device") device = atoi(next());
        else if (a == "--language") opts.pipe.asr.language = next();
        else if (a == "--no-itn") opts.pipe.asr.itn = false;
        else if (a == "--threads") opts.pipe.asr.n_threads = atoi(next());
        else if (a == "--end-sil") opts.pipe.vad.max_end_silence_ms = atoi(next());
        else if (a == "--max-seg") opts.pipe.vad.max_single_segment_ms = atoi(next());
        else if (a == "--early-asr") opts.early_asr_ms = atoi(next());
        else if (a == "--vad-block") opts.vad_block_ms = atoi(next());
        else if (a == "--no-apm") opts.pipe.apm.enabled = false;
        else if (a == "--ns") opts.pipe.apm.ns_level = atoi(next());
        else if (a == "--no-agc") opts.pipe.apm.agc2 = false;
        else if (a == "--no-hpf") opts.pipe.apm.high_pass = false;
        else if (a == "--record") opts.record_path = next();
        else if (a == "--spk-model") opts.speaker_model = next();
        else if (a == "--speaker-db") opts.speaker.db_path = next();
        else if (a == "--speaker-threshold" || a == "--speaker-short-floor") {
            const char* v = next();
            char* end = nullptr;
            double t = strtod(v, &end);
            if (end == v || *end != '\0' || !(t >= 0.0 && t <= 1.0)) {
                fprintf(stderr, "%s must be a number in [0, 1], got '%s'\n", a.c_str(), v);
                return 2;
            }
            (a == "--speaker-threshold" ? opts.speaker.threshold : opts.speaker.uncertain_floor) = (float)t;
        }
        else if (a == "--speaker-min-ms") opts.speaker.min_reliable_ms = atoi(next());
        else if (a == "--speaker-gate-short") opts.speaker.gate_uncertain = true;
        else if (a == "--speaker-observe") opts.speaker_observe = true;
        else if (a == "--no-stdin-watch") stdin_watch = false;
        else if (a == "--no-auth") no_auth = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--version") {  // for installers: no models, no server
            printf("%s\n", agent::JsonObject().str("version", kVersion).num("api_version", kApiVersion).dump().c_str());
            return 0;
        }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (vad_path.empty() || asr_path.empty()) { usage(argv[0]); return 2; }
    const char* tok = getenv("MINTBOT_VOICE_TOKEN");
    std::string token = tok ? tok : "";
    if (token.empty() && !no_auth) { fprintf(stderr, "MINTBOT_VOICE_TOKEN is not set (use --no-auth only for local debugging)\n"); return 2; }

    agent::EventHub hub;
    agent::Agent engine(hub);
    std::string err;
    fprintf(stderr, "[voice-agent] loading models...\n");
    if (!engine.load(vad_path, asr_path, opts, &err)) { fprintf(stderr, "[voice-agent] %s\n", err.c_str()); return 1; }

    httplib::Server svr;
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (req.path == "/v1/health" || token.empty()) return httplib::Server::HandlerResponse::Unhandled;
        if (same_secret(req.get_header_value("Authorization"), "Bearer " + token)) return httplib::Server::HandlerResponse::Unhandled;
        reply_json(res, 401, "{\"error\":\"unauthorized\"}");
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        reply_json(res, 200,
                   agent::JsonObject().str("status", "ok").str("version", kVersion).num("api_version", kApiVersion).dump());
    });
    svr.Get("/v1/status", [&](const httplib::Request&, httplib::Response& res) { reply_json(res, 200, engine.status_json()); });
    svr.Get("/v1/devices", [&](const httplib::Request&, httplib::Response& res) {
        std::string e;
        auto devs = agent::list_capture_devices(&e);
        std::string arr = "[";
        for (size_t i = 0; i < devs.size(); i++)
            arr += (i ? "," : "") + agent::JsonObject().num("index", (int64_t)devs[i].index).str("name", devs[i].name)
                                       .boolean("is_default", devs[i].is_default).dump();
        reply_json(res, e.empty() ? 200 : 500, agent::JsonObject().raw("devices", arr + "]").str("error", e).dump());
    });
    svr.Post("/v1/listen/start", [&](const httplib::Request& req, httplib::Response& res) {
        int dev = req.has_param("device") ? atoi(req.get_param_value("device").c_str()) : -1;
        std::string e;
        if (engine.start_listening(dev, &e)) reply_json(res, 200, engine.status_json());
        else reply_json(res, 500, agent::JsonObject().str("error", e).str("code", agent::capture_error_code(e)).dump());
    });
    svr.Post("/v1/listen/stop", [&](const httplib::Request&, httplib::Response& res) {
        engine.stop_listening();
        reply_json(res, 200, engine.status_json());
    });
    auto with_gate = [&](httplib::Response& res, const std::function<void(spk::SpeakerGate&)>& fn) {
        spk::SpeakerGate* g = engine.speaker();
        if (!g) { reply_json(res, 404, "{\"error\":\"speaker gate disabled (start with --spk-model)\"}"); return; }
        fn(*g);
    };
    auto enroll_json = [](const spk::EnrollState& s) {
        return agent::JsonObject().boolean("active", s.active).num("utterances", (int64_t)s.utterances)
            .num("speech_ms", s.speech_ms).boolean("ready", s.ready).dump();
    };
    svr.Get("/v1/speaker", [&](const httplib::Request&, httplib::Response& res) {
        with_gate(res, [&](spk::SpeakerGate& g) { reply_json(res, 200, g.status_json()); });
    });
    svr.Post("/v1/speaker/enroll/start", [&](const httplib::Request&, httplib::Response& res) {
        with_gate(res, [&](spk::SpeakerGate& g) {
            std::string j = enroll_json(g.enroll_begin());
            hub.publish("enroll_progress", j);
            reply_json(res, 200, j);
        });
    });
    svr.Post("/v1/speaker/enroll/commit", [&](const httplib::Request&, httplib::Response& res) {
        with_gate(res, [&](spk::SpeakerGate& g) {
            std::string e;
            if (!g.enroll_commit(&e)) { reply_json(res, 409, agent::JsonObject().str("error", e).dump()); return; }
            hub.publish("speaker", g.status_json());
            reply_json(res, 200, g.status_json());
        });
    });
    svr.Post("/v1/speaker/enroll/cancel", [&](const httplib::Request&, httplib::Response& res) {
        with_gate(res, [&](spk::SpeakerGate& g) {
            g.enroll_cancel();
            reply_json(res, 200, g.status_json());
        });
    });
    svr.Delete("/v1/speaker", [&](const httplib::Request&, httplib::Response& res) {
        with_gate(res, [&](spk::SpeakerGate& g) {
            std::string e;
            if (!g.forget(&e)) { reply_json(res, 500, agent::JsonObject().str("error", e).dump()); return; }
            hub.publish("speaker", g.status_json());
            reply_json(res, 200, g.status_json());
        });
    });
    svr.Post("/v1/shutdown", [&](const httplib::Request&, httplib::Response& res) {
        reply_json(res, 200, "{\"status\":\"shutting down\"}");
        request_quit("shutdown request");
    });
    svr.Get("/v1/events", [&](const httplib::Request&, httplib::Response& res) {
        auto sub = hub.subscribe();
        hub.publish("status", engine.status_json());  // also tells existing subscribers someone joined
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [sub](size_t, httplib::DataSink& sink) {
                agent::Event ev;
                if (sub->next(ev, std::chrono::seconds(15))) {
                    std::string msg = "id: " + std::to_string(ev.id) + "\nevent: " + ev.type + "\ndata: " + ev.json + "\n\n";
                    return sink.write(msg.data(), msg.size());
                }
                if (sub->closed()) { sink.done(); return true; }
                return sink.write(": ping\n\n", 8);  // keep-alive; also detects a vanished client
            },
            [&hub, sub](bool) { hub.unsubscribe(sub); });
    });

    int bound = port == 0 ? svr.bind_to_any_port(host) : (svr.bind_to_port(host, port) ? port : -1);
    if (bound <= 0) { fprintf(stderr, "[voice-agent] cannot bind %s:%d\n", host.c_str(), port); return 1; }
    std::thread http_th([&] { svr.listen_after_bind(); });

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (stdin_watch) {
        std::thread([] {
            while (fgetc(stdin) != EOF) {}
            request_quit("stdin closed");
        }).detach();
    }

    if (autostart && !engine.start_listening(device, &err)) fprintf(stderr, "[voice-agent] autostart failed: %s\n", err.c_str());

    printf("%s\n", agent::JsonObject().str("type", "ready").num("port", (int64_t)bound).num("pid", (int64_t)getpid())
                       .str("version", kVersion).num("api_version", kApiVersion).dump().c_str());
    fflush(stdout);
    fprintf(stderr, "[voice-agent] listening on http://%s:%d\n", host.c_str(), bound);

    {
        std::unique_lock<std::mutex> lk(g_quit_m);
        g_quit_cv.wait(lk, [] { return g_quit; });
    }
    fprintf(stderr, "[voice-agent] exiting (%s)\n", g_quit_reason.c_str());
    engine.shutdown();
    hub.close();
    svr.stop();
    if (http_th.joinable()) http_th.join();
    return 0;
}
