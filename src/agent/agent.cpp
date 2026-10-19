#include "agent/agent.h"

#include "agent/audio_source.h"
#include "agent/json.h"
#include "common/wav_io.h"
#include "dsp/apm.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace mintbot::agent {
namespace {

using Clock = std::chrono::steady_clock;

int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

}  // namespace

struct Agent::Impl {
    EventHub& hub;
    AgentOptions opts;
    dsp::AudioPreprocessor apm;  // front thread only
    std::vector<float> apm_out;  // front thread only
    WavWriter recorder;          // front thread only
    vad::FsmnVadStream vad;      // front thread only
    pipeline::Segmenter seg;     // front thread only
    std::vector<float> vad_in;   // front thread only: preprocessed audio not yet given to the VAD
    void feed_vad() {
        if (vad_in.empty()) return;
        handle_events(vad.feed(vad_in.data(), vad_in.size()));
        vad_in.clear();
    }
    asr::SenseVoice asr;         // ASR thread only
    std::unique_ptr<spk::SpeakerGate> gate;  // analyze() on the ASR thread only
    std::atomic<int64_t> skipped{0};         // utterances not transcribed by the gate
    bool loaded = false;

    // audio source; start/stop/restart under src_m
    std::mutex src_m;
    std::unique_ptr<AudioSource> src;
    std::string src_name;
    int device = -1;
    std::atomic<uint64_t> gen{0};  // callbacks from older sources are ignored
    std::atomic<bool> listening{false};
    std::atomic<bool> need_restart{false};
    std::atomic<int64_t> last_audio_ms{0};
    int64_t next_restart_ms = 0;  // front thread only
    std::atomic<int> restarts{0};

    // capture -> front
    std::mutex q_m;
    std::condition_variable q_cv;
    std::vector<float> q;
    bool flush_req = false;
    bool ended_req = false;
    std::atomic<int64_t> dropped_samples{0};

    // front -> ASR
    struct Pending {
        pipeline::Utterance utt;
        int64_t end_unix_ms;
        int64_t detected_steady_ms;
        int64_t detect_lag_ms;  // audio between the end of speech and the moment the VAD closed the segment
    };
    // Full speaker windows of the open segment, embedded on the ASR thread while
    // the speech is still going on, so only the tail is left when it ends.
    struct WindowJob {
        int64_t seg;  // segment start, absolute sample
        size_t off;   // window start within the segment
        std::vector<float> pcm;
    };
    std::mutex a_m;
    std::condition_variable a_cv;
    std::deque<Pending> aq;   // has priority over wq
    std::deque<WindowJob> wq;
    int64_t win_seg = -1;     // front thread only: segment being scheduled
    size_t win_next = 0;      // front thread only: next window start in it
    std::map<int64_t, spk::WindowCache> win_cache;  // ASR thread only, by segment start
    int64_t win_done_seg = -1;                      // ASR thread only: last segment analysed

    // Early recognition of the open segment, ended where the VAD predicts.
    struct EarlyJob {
        int64_t gen = -1;     // numbers the predictions, in order
        int64_t seg = -1;     // segment start, absolute sample
        int64_t end_ms = -1;  // predicted SpeechEnd
        std::vector<float> pcm;
    };
    std::deque<EarlyJob> eq;                 // under a_m; between aq and wq in priority
    int64_t early_seg = -1, early_end = -1;  // front thread only: last job queued
    int64_t early_gen = -1;                  // front thread only: its gen
    // A prediction proven wrong (speech resumed, or the segment ended elsewhere)
    // stops its run between two graph nodes instead of finishing it.
    std::atomic<int64_t> early_dead{-1};     // every gen <= this is dead
    std::atomic<int64_t> early_running{-1};  // gen the ASR thread is computing, -1 = none
    std::atomic<int64_t> early_aborted{0};
    static bool abort_early(void* self) {
        auto* m = static_cast<Impl*>(self);
        const int64_t g = m->early_running.load(std::memory_order_relaxed);
        return g >= 0 && g <= m->early_dead.load(std::memory_order_relaxed);
    }
    // Front thread: the prediction behind the last queued job no longer holds.
    void kill_early() {
        if (early_gen >= 0 && early_dead.load() < early_gen) early_dead = early_gen;
    }
    struct EarlyResult {
        bool valid = false;
        int64_t seg = -1, end_ms = -1;
        uint64_t spk_version = 0;
        spk::SpeakerDecision sd;
        asr::Transcript tr;
        double asr_ms = 0;
        int64_t busy_us = 0;
    } early;  // ASR thread only
    std::atomic<int64_t> early_runs{0}, early_hits{0};
    // Wall time the ASR thread spent computing (speaker + ASR), by kind of work.
    std::atomic<int64_t> busy_final_us{0}, busy_window_us{0}, busy_early_us{0}, busy_early_used_us{0};
    static int64_t us_since(Clock::time_point t0) {
        return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
    }

    std::atomic<bool> running{false};
    std::thread front_th, asr_th;
    std::atomic<int64_t> stream_samples{0};
    std::atomic<bool> in_speech{false};
    std::atomic<int64_t> utterances{0};
    int64_t started_steady_ms = steady_ms();

    explicit Impl(EventHub& h) : hub(h) {}

    void publish_error(const std::string& msg) { hub.publish("error", JsonObject().str("message", msg).dump()); }

    // What the capture is actually doing (listening = what was requested):
    // "running", "retrying" (source lost, restarting with back-off) or "stopped".
    std::mutex cap_m;
    std::string cap_state = "stopped", cap_code, cap_msg;
    // Publishes a "capture" event only when the state or error code changes; returns whether it did.
    bool set_capture(const std::string& state, const std::string& code = "", const std::string& msg = "") {
        {
            std::lock_guard<std::mutex> lk(cap_m);
            bool same = state == cap_state && code == cap_code;
            cap_msg = msg;
            if (same) return false;
            cap_state = state;
            cap_code = code;
        }
        hub.publish("capture", JsonObject().str("state", state).str("code", code).str("message", msg).dump());
        return true;
    }

    // ---- source callbacks (source threads) ----
    void on_pcm(uint64_t g, const float* pcm, size_t n) {
        if (g != gen.load()) return;
        last_audio_ms = steady_ms();
        const size_t cap = (size_t)opts.max_queue_ms * 16;
        {
            std::lock_guard<std::mutex> lk(q_m);
            q.insert(q.end(), pcm, pcm + n);
            if (q.size() > cap) {
                size_t drop = q.size() - cap;
                q.erase(q.begin(), q.begin() + drop);
                dropped_samples += (int64_t)drop;
            }
        }
        q_cv.notify_one();
    }

    void on_state(uint64_t g, const std::string& state, const std::string& detail) {
        hub.publish("device", JsonObject().str("state", state).str("name", detail).dump());
        if (g != gen.load() || !listening) return;
        if (state == "stopped" || state == "error") {
            need_restart = true;
        } else if (state == "ended") {  // wav replay finished: finish the open utterance, then stop
            {
                std::lock_guard<std::mutex> lk(q_m);
                ended_req = true;
                flush_req = true;
            }
            q_cv.notify_one();
        }
    }

    bool start_source_locked(std::string* error) {
        ++gen;
        uint64_t g = gen.load();
        std::unique_ptr<AudioSource> s;
        if (opts.source.rfind("wav:", 0) == 0) s = make_wav_source(opts.source.substr(4), opts.replay_speed);
        else s = make_capture_source(device);
        bool ok = s->start([this, g](const float* p, size_t n) { on_pcm(g, p, n); },
                           [this, g](const std::string& st, const std::string& d) { on_state(g, st, d); }, error);
        if (!ok) return false;
        src_name = s->name();
        src = std::move(s);
        last_audio_ms = steady_ms();
        return true;
    }

    void stop_source_locked() {
        ++gen;  // the stop notification of this source is no longer ours
        if (src) { src->stop(); src.reset(); }
    }

    // ---- front thread ----
    void handle_events(const std::vector<vad::VadEvent>& ev) {
        if (ev.empty()) return;
        const int64_t now_unix = unix_ms(), stream_end_ms = stream_samples / 16;
        for (const auto& e : ev) {
            bool start = e.type == vad::VadEvent::SpeechStart;
            in_speech = start;
            hub.publish(start ? "speech_start" : "speech_end", JsonObject()
                                                                    .num("stream_ms", e.ms)
                                                                    .num("unix_ms", now_unix - (stream_end_ms - e.ms))
                                                                    .num("detect_ms", stream_end_ms)
                                                                    .num("detect_lag_ms", stream_end_ms - e.ms)
                                                                    .dump());
        }
        auto utts = seg.on_events(ev);
        for (const auto& u : utts)
            if (u.start_sample == early_seg && u.end_ms != early_end) kill_early();
        if (utts.empty()) return;
        {
            std::lock_guard<std::mutex> lk(a_m);
            for (auto& u : utts) {
                int64_t end_unix = now_unix - (stream_end_ms - u.end_ms);
                const int64_t lag = stream_end_ms - u.end_ms;
                aq.push_back({std::move(u), end_unix, steady_ms(), lag});
            }
        }
        a_cv.notify_one();
    }

    void supervise(bool ended) {
        if (ended) {
            std::lock_guard<std::mutex> lk(src_m);
            listening = false;
            stop_source_locked();
            set_capture("stopped");
            return;
        }
        if (!listening) return;
        int64_t now = steady_ms();
        if (!need_restart && now - last_audio_ms > opts.watchdog_ms) {
            need_restart = true;
            hub.publish("device", JsonObject().str("state", "stalled").str("name", src_name).dump());
        }
        if (!need_restart || now < next_restart_ms) return;
        std::lock_guard<std::mutex> lk(src_m);
        if (!listening) return;
        stop_source_locked();
        std::string err;
        if (start_source_locked(&err)) {
            need_restart = false;
            ++restarts;
            set_capture("running");
        } else {
            next_restart_ms = now + opts.restart_backoff_ms;
            const std::string code = capture_error_code(err), msg = "capture restart failed: " + err;
            if (set_capture("retrying", code, msg))  // report each distinct failure once, not every retry
                hub.publish("error", JsonObject().str("message", msg).str("code", code).dump());
        }
    }

    void front_loop() {
        while (running) {
            std::vector<float> chunk;
            bool flush = false, ended = false;
            {
                std::unique_lock<std::mutex> lk(q_m);
                q_cv.wait_for(lk, std::chrono::milliseconds(100), [&] { return !q.empty() || flush_req || !running; });
                chunk.swap(q);
                flush = flush_req;
                ended = ended_req;
                flush_req = ended_req = false;
            }
            if (!chunk.empty()) {
                recorder.write(chunk.data(), chunk.size());
                apm_out.clear();
                apm.process(chunk.data(), chunk.size(), apm_out);
                if (!apm_out.empty()) {
                    seg.push_audio(apm_out.data(), apm_out.size());
                    stream_samples += (int64_t)apm_out.size();
                    // The VAD gives the same frames for any chunking; fewer, larger
                    // calls cost less (one graph per call) and delay events by < one block.
                    vad_in.insert(vad_in.end(), apm_out.begin(), apm_out.end());
                    if (vad_in.size() >= (size_t)opts.vad_block_ms * 16) feed_vad();
                }
            }
            if (flush) {
                feed_vad();
                handle_events(vad.flush());
            }
            schedule_windows();
            schedule_early();
            supervise(ended);
        }
    }

    // Queues every full window of the open segment that the buffer now covers.
    void schedule_windows() {
        if (!gate) return;
        const int64_t open = seg.open_start();
        if (open < 0) return;
        if (open != win_seg) {
            if (!gate->wants_windows()) return;
            win_seg = open;
            win_next = 0;
        }
        bool queued = false;
        {
            std::lock_guard<std::mutex> lk(a_m);
            while (open + (int64_t)(win_next + spk::kSvWindow) <= seg.end_sample()) {
                WindowJob j{open, win_next, std::vector<float>(spk::kSvWindow)};
                if (!seg.copy(open + (int64_t)win_next, spk::kSvWindow, j.pcm.data())) break;
                wq.push_back(std::move(j));
                win_next += spk::kSvShift;
                queued = true;
            }
        }
        if (queued) a_cv.notify_one();
    }

    // Queues early recognition once the silence after speech reaches early_asr_ms.
    void schedule_early() {
        if (opts.early_asr_ms <= 0) return;
        const int64_t open = seg.open_start();
        if (open < 0) return;
        int64_t sil = 0;
        const int64_t end_ms = vad.pending_end_ms(&sil);
        if (open == early_seg && end_ms != early_end) kill_early();  // speech resumed
        if (end_ms < 0 || sil < opts.early_asr_ms || (open == early_seg && end_ms == early_end)) return;
        const int64_t end = end_ms * 16;
        if (end <= open || end > seg.end_sample()) return;
        EarlyJob j{early_gen + 1, open, end_ms, std::vector<float>((size_t)(end - open))};
        if (!seg.copy(open, j.pcm.size(), j.pcm.data())) return;
        kill_early();  // the previous prediction, if still running
        early_gen = j.gen;
        early_seg = open;
        early_end = end_ms;
        {
            std::lock_guard<std::mutex> lk(a_m);
            eq.clear();  // an older prediction is stale now
            eq.push_back(std::move(j));
        }
        a_cv.notify_one();
    }

    // ---- ASR thread ----
    void run_early(EarlyJob& j) {
        if (j.seg <= win_done_seg || j.gen <= early_dead) return;
        if (gate && gate->enrolling()) return;  // enrollment must see each utterance exactly once
        ++early_runs;
        early_running = j.gen;
        const auto t_run = Clock::now();
        EarlyResult r;
        r.seg = j.seg;
        r.end_ms = j.end_ms;
        if (gate) {
            r.spk_version = gate->version();
            auto it = win_cache.find(j.seg);
            r.sd = gate->analyze(j.pcm.data(), j.pcm.size(), it == win_cache.end() ? nullptr : &it->second);
        }
        if (r.sd.transcribe) {
            auto t0 = Clock::now();
            r.tr = asr.transcribe(j.pcm.data(), j.pcm.size());
            r.asr_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }
        early_running = -1;
        r.busy_us = us_since(t_run);
        busy_early_us += r.busy_us;
        if (j.gen <= early_dead) {  // aborted or proven wrong while running: worthless
            ++early_aborted;
            return;
        }
        r.valid = true;
        early = std::move(r);
    }

    void precompute(const WindowJob& j) {
        if (j.seg <= win_done_seg) return;  // its utterance was already analysed
        std::vector<float> e;
        const auto t0 = Clock::now();
        if (gate->embed_window(j.pcm.data(), e)) win_cache[j.seg][j.off] = std::move(e);
        busy_window_us += us_since(t0);
    }

    void asr_loop() {
        while (true) {
            Pending p;
            WindowJob job;
            EarlyJob ej;
            enum { kUtt, kEarly, kWindow } what;
            {
                std::unique_lock<std::mutex> lk(a_m);
                a_cv.wait(lk, [&] { return !aq.empty() || !eq.empty() || !wq.empty() || !running; });
                if (!aq.empty()) {
                    p = std::move(aq.front());
                    aq.pop_front();
                    what = kUtt;
                    // work still queued for this segment is cheaper to do inline now
                    while (!wq.empty() && wq.front().seg <= p.utt.start_sample) wq.pop_front();
                    while (!eq.empty() && eq.front().seg <= p.utt.start_sample) eq.pop_front();
                } else if (!running) {
                    return;
                } else if (!eq.empty()) {
                    ej = std::move(eq.front());
                    eq.pop_front();
                    what = kEarly;
                } else {
                    job = std::move(wq.front());
                    wq.pop_front();
                    what = kWindow;
                }
            }
            if (what == kWindow) {
                precompute(job);
                continue;
            }
            if (what == kEarly) {
                run_early(ej);
                continue;
            }
            const int64_t queue_ms = steady_ms() - p.detected_steady_ms;
            // the early result counts only if it was made on exactly these samples,
            // with the voiceprint and enrollment state that hold now
            const bool reuse = early.valid && early.seg == p.utt.start_sample && early.end_ms == p.utt.end_ms &&
                               (int64_t)p.utt.pcm.size() == p.utt.end_ms * 16 - early.seg &&
                               (!gate || (!gate->enrolling() && gate->version() == early.spk_version));
            EarlyResult used;
            if (reuse) {
                used = std::move(early);
                ++early_hits;
                busy_early_used_us += used.busy_us;
            }
            const auto t_final = Clock::now();
            early.valid = false;
            spk::SpeakerDecision sd;
            if (reuse) {
                sd = used.sd;
                if (gate) {
                    win_done_seg = p.utt.start_sample;
                    win_cache.erase(win_cache.begin(), win_cache.upper_bound(p.utt.start_sample));
                }
            } else if (gate) {
                auto it = win_cache.find(p.utt.start_sample);
                sd = gate->analyze(p.utt.pcm.data(), p.utt.pcm.size(), it == win_cache.end() ? nullptr : &it->second);
                win_done_seg = p.utt.start_sample;
                win_cache.erase(win_cache.begin(), win_cache.upper_bound(p.utt.start_sample));
            }
            if (sd.enrolling) {
                auto st = gate->enroll_state();
                hub.publish("enroll_progress", JsonObject()
                                                   .num("utterances", (int64_t)st.utterances)
                                                   .num("speech_ms", st.speech_ms)
                                                   .boolean("ready", st.ready)
                                                   .dump());
            }
            auto speaker = [&](JsonObject& o) -> JsonObject& {
                return o.boolean("speaker_enrolled", sd.enrolled)
                    .num("speaker_score", (double)sd.score)
                    .boolean("is_owner", sd.is_owner)
                    .boolean("speaker_uncertain", sd.uncertain)
                    .num("speaker_ms", sd.embed_ms)
                    .num("speaker_windows", (int64_t)sd.windows)
                    .num("speaker_reused", (int64_t)sd.reused);
            };
            if (!sd.transcribe) {  // not the owner: no ASR, report only
                busy_final_us += us_since(t_final);
                ++skipped;
                JsonObject o;
                o.num("start_ms", p.utt.start_ms).num("end_ms", p.utt.end_ms).num("end_unix_ms", p.end_unix_ms);
                const int64_t lat = steady_ms() - p.detected_steady_ms;
                hub.publish("speech_other", speaker(o)
                                                .num("latency_ms", lat)
                                                .num("queue_ms", queue_ms)
                                                .num("detect_lag_ms", p.detect_lag_ms)
                                                .num("total_ms", p.detect_lag_ms + lat)
                                                .boolean("early", reuse)
                                                .dump());
                continue;
            }
            asr::Transcript tr;
            double asr_ms = 0;
            if (reuse) {
                tr = std::move(used.tr);
                asr_ms = used.asr_ms;
            } else {
                auto t0 = Clock::now();
                tr = asr.transcribe(p.utt.pcm.data(), p.utt.pcm.size());
                asr_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            }
            busy_final_us += us_since(t_final);
            ++utterances;
            JsonObject o;
            o.num("start_ms", p.utt.start_ms)
                .num("end_ms", p.utt.end_ms)
                .num("end_unix_ms", p.end_unix_ms)
                .str("text", tr.text)
                .str("language", tr.language)
                .str("emotion", tr.emotion)
                .str("event", tr.event)
                .str("raw", tr.raw)
                .num("asr_ms", asr_ms)
                .boolean("early", reuse);
            if (gate) speaker(o).boolean("enrolling", sd.enrolling);
            const int64_t lat = steady_ms() - p.detected_steady_ms;
            hub.publish("final", o.num("latency_ms", lat)
                                     .num("queue_ms", queue_ms)
                                     .num("detect_lag_ms", p.detect_lag_ms)
                                     .num("total_ms", p.detect_lag_ms + lat)
                                     .dump());
        }
    }
};

Agent::Agent(EventHub& hub) : p_(std::make_unique<Impl>(hub)) {}
Agent::~Agent() { shutdown(); }

bool Agent::load(const std::string& vad_gguf, const std::string& asr_gguf, const AgentOptions& opts, std::string* error) {
    Impl& m = *p_;
    m.opts = opts;
    m.seg = pipeline::Segmenter(opts.pipe.keep_ms);
    if (!m.apm.init(opts.pipe.apm, error)) return false;
    if (!opts.record_path.empty() && !m.recorder.open(opts.record_path)) {
        if (error) *error = "cannot open " + opts.record_path + " for recording";
        return false;
    }
    if (!m.vad.load(vad_gguf, opts.pipe.vad)) { if (error) *error = "cannot load VAD model " + vad_gguf; return false; }
    if (!m.asr.load(asr_gguf, opts.pipe.asr)) { if (error) *error = "cannot load ASR model " + asr_gguf; return false; }
    if (!opts.speaker_model.empty()) {
        m.gate = std::make_unique<spk::SpeakerGate>();
        if (!m.gate->load(opts.speaker_model, opts.speaker, error)) return false;
        m.gate->set_abort_callback(&Impl::abort_early, &m);
    }
    m.asr.set_abort_callback(&Impl::abort_early, &m);
    m.loaded = true;
    m.running = true;
    m.front_th = std::thread([&m] { m.front_loop(); });
    m.asr_th = std::thread([&m] { m.asr_loop(); });
    return true;
}

bool Agent::start_listening(int device, std::string* error) {
    Impl& m = *p_;
    if (!m.loaded) { if (error) *error = "models not loaded"; return false; }
    std::lock_guard<std::mutex> lk(m.src_m);
    if (m.listening) return true;
    m.device = device;
    if (!m.start_source_locked(error)) return false;
    m.need_restart = false;
    m.listening = true;
    m.set_capture("running");
    return true;
}

void Agent::stop_listening() {
    Impl& m = *p_;
    {
        std::lock_guard<std::mutex> lk(m.src_m);
        if (!m.listening) return;
        m.listening = false;
        m.stop_source_locked();
        m.set_capture("stopped");
    }
    {
        std::lock_guard<std::mutex> lk(m.q_m);
        m.flush_req = true;
    }
    m.q_cv.notify_one();
}

std::string Agent::status_json() {
    Impl& m = *p_;
    std::string name;
    int device;
    {
        std::lock_guard<std::mutex> lk(m.src_m);
        name = m.src ? m.src_name : "";
        device = m.device;
    }
    size_t aq;
    {
        std::lock_guard<std::mutex> lk(m.a_m);
        aq = m.aq.size();
    }
    std::string cap_state, cap_code, cap_msg;
    {
        std::lock_guard<std::mutex> lk(m.cap_m);
        cap_state = m.cap_state;
        cap_code = m.cap_code;
        cap_msg = m.cap_msg;
    }
    return JsonObject()
        .boolean("listening", m.listening)
        .str("capture", cap_state)
        .str("capture_error_code", cap_code)
        .str("capture_error", cap_msg)
        .str("source", m.opts.source)
        .str("device_name", name)
        .num("device_index", (int64_t)device)
        .boolean("in_speech", m.in_speech)
        .num("stream_ms", m.stream_samples / 16)
        .num("dropped_ms", m.dropped_samples / 16)
        .boolean("apm", m.opts.pipe.apm.enabled)
        .num("ns_level", (int64_t)m.opts.pipe.apm.ns_level)
        .boolean("agc2", m.opts.pipe.apm.enabled && m.opts.pipe.apm.agc2)
        .boolean("recording", !m.opts.record_path.empty())
        .num("utterances", m.utterances.load())
        .num("skipped_other_speaker", m.skipped.load())
        .raw("speaker", m.gate ? m.gate->status_json() : "null")
        .num("asr_queue", (int64_t)aq)
        .num("early_runs", m.early_runs.load())
        .num("early_hits", m.early_hits.load())
        .num("early_aborted", m.early_aborted.load())
        .raw("busy_ms", JsonObject()
                            .num("final", m.busy_final_us.load() / 1000)
                            .num("windows", m.busy_window_us.load() / 1000)
                            .num("early_used", m.busy_early_used_us.load() / 1000)
                            .num("early_wasted", (m.busy_early_us.load() - m.busy_early_used_us.load()) / 1000)
                            .dump())
        .num("restarts", (int64_t)m.restarts.load())
        .num("subscribers", (int64_t)m.hub.subscribers())
        .num("uptime_s", (steady_ms() - m.started_steady_ms) / 1000)
        .dump();
}

spk::SpeakerGate* Agent::speaker() { return p_->gate.get(); }

void Agent::shutdown() {
    Impl& m = *p_;
    {
        std::lock_guard<std::mutex> lk(m.src_m);
        m.listening = false;
        m.stop_source_locked();
    }
    if (!m.running.exchange(false)) return;
    m.q_cv.notify_all();
    m.a_cv.notify_all();
    if (m.front_th.joinable()) m.front_th.join();
    if (m.asr_th.joinable()) m.asr_th.join();
    m.recorder.close();
}

}  // namespace mintbot::agent
