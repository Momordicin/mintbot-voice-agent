#include "agent/player.h"

#include "agent/json.h"

// declarations only: the implementation is compiled in audio_source.cpp
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace mintbot::agent {
namespace {

using Clock = std::chrono::steady_clock;

// After the queue drains the device stays open this long, so the next sentence starts without reopening it.
constexpr auto kIdleClose = std::chrono::seconds(2);
// A device stopped by the system gets this long to be started again (miniaudio reroutes it to
// the new default device) before its current clip is given up as stopped.
constexpr auto kRerouteGrace = std::chrono::seconds(2);

struct Clip {
    uint64_t id = 0;
    int rate = 0;
    std::vector<int16_t> pcm;
    bool started = false;     // its first sample went to the device
    bool ended = false;       // its last sample went to the device
    bool start_sent = false;  // the "start" event is queued
};

using Clips = std::vector<std::unique_ptr<Clip>>;
using Events = std::vector<std::pair<const char*, uint64_t>>;  // (state, id)

// What a locked section lets go of; freed after the lock is released.
struct Freed {
    Clips clips;    // removed from the queue
    Clips queue;    // a grown or the replaced buffer of the queue
    Events events;  // a grown or the replaced buffer of the pending events
};

// Moves v into `bigger` when that has more room; the old buffer of v is left in `bigger`.
template <class V>
void adopt(V& v, V& bigger) {
    if (bigger.capacity() <= v.capacity() || bigger.capacity() < v.size()) return;
    for (auto& x : v) bigger.push_back(std::move(x));
    v.swap(bigger);
    bigger.clear();
}

}  // namespace

// Threads: HTTP handlers (enqueue, stop, status), the audio callback, miniaudio's notification
// threads, and one control thread that opens, reopens and closes the device and turns clip
// progress into events. `m` guards the queue, the callback's position and the device flags.
// The audio callback takes it, so nothing allocates or frees memory while holding it:
// lock_with_room() grows the vectors with `m` released, removed clips and replaced buffers are
// freed after it is released, and JSON is built outside it. Nobody holds it while opening or
// closing the device or publishing.
struct Player::Impl {
    EventHub& hub;

    std::mutex m;
    std::condition_variable cv;
    Clips queue;  // oldest first; [0, head) handed to the device completely
    size_t head = 0;
    size_t pos = 0;    // samples of queue[head] handed to the device
    int dev_rate = 0;  // rate of the running device; 0: the callback plays nothing
    bool changed = false, quit = false;
    bool stalled = false;  // the device stopped without us; lost unless it starts again by stalled_until
    Clock::time_point stalled_until;
    uint64_t next_id = 0;
    Events events;  // published in order by flush()

    std::mutex pub_m;  // one flush at a time, so events from different threads keep their order
    Events publishing;  // pub_m only

    // control thread only
    ma_device dev{};
    bool open = false;
    std::atomic<bool> closing{false};  // the device's "stopped" notification is ours
    std::thread th;

    explicit Impl(EventHub& h) : hub(h) { th = std::thread([this] { run(); }); }

    // Takes `m` once the queue, the pending events and freed.clips have room for all one locked
    // section adds: one clip, and per clip two events and one removal. Grows them with `m` released.
    std::unique_lock<std::mutex> lock_with_room(Freed& freed) {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            const size_t clips = queue.size() + 1, evs = events.size() + 2 * clips, removed = freed.clips.size() + clips;
            const bool grow_queue = queue.capacity() < clips, grow_events = events.capacity() < evs;
            if (!grow_queue && !grow_events && freed.clips.capacity() >= removed) return lk;
            lk.unlock();
            freed.queue = Clips();
            freed.events = Events();
            freed.clips.reserve(freed.clips.size() + 2 * clips);
            if (grow_queue) freed.queue.reserve(2 * clips);
            if (grow_events) freed.events.reserve(2 * evs);
            lk.lock();
            adopt(queue, freed.queue);
            adopt(events, freed.events);
        }
    }

    void note_start_locked(Clip& c) {
        if (c.started && !c.start_sent) {
            c.start_sent = true;
            events.push_back({"start", c.id});
        }
    }

    // Removes the first n clips: "end" for those handed to the device completely, else "stopped".
    void drop_front_locked(size_t n, Freed& freed) {
        const size_t k = std::min(n, queue.size());
        for (size_t i = 0; i < k; i++) {
            Clip& c = *queue[i];
            note_start_locked(c);
            events.push_back({c.ended ? "end" : "stopped", c.id});
            freed.clips.push_back(std::move(queue[i]));
        }
        queue.erase(queue.begin(), queue.begin() + k);
        if (n > head) pos = 0;
        head -= std::min(head, n);
    }

    // Events for the callback's progress; finished clips leave the queue.
    void reap_locked(Freed& freed) {
        while (!queue.empty()) {
            note_start_locked(*queue.front());
            if (!queue.front()->ended) break;
            drop_front_locked(1, freed);
        }
    }

    void flush() {
        std::lock_guard<std::mutex> pl(pub_m);
        {
            std::lock_guard<std::mutex> lk(m);
            publishing.swap(events);
        }
        for (const auto& e : publishing) hub.publish("playback", JsonObject().str("state", e.first).num("id", (int64_t)e.second).dump());
        publishing.clear();
    }

    void publish_error(const std::string& msg) {
        hub.publish("error", JsonObject().str("message", msg).str("code", "playback_failed").dump());
    }

    // ---- audio callback: copies samples, sets flags, never waits beyond `m` ----
    static void on_data(ma_device* d, void* output, const void*, ma_uint32 frames) {
        auto* self = (Impl*)d->pUserData;
        auto* out = (int16_t*)output;
        bool progressed = false;
        {
            std::lock_guard<std::mutex> lk(self->m);
            while (frames > 0 && self->head < self->queue.size()) {
                Clip& c = *self->queue[self->head];
                if (c.rate != self->dev_rate) break;
                if (!c.started) c.started = progressed = true;
                const size_t n = std::min((size_t)frames, c.pcm.size() - self->pos);
                memcpy(out, c.pcm.data() + self->pos, n * sizeof(int16_t));
                out += n;
                frames -= (ma_uint32)n;
                self->pos += n;
                if (self->pos == c.pcm.size()) {
                    c.ended = progressed = true;
                    self->head++;
                    self->pos = 0;
                }
            }
            if (progressed) self->changed = true;
        }
        if (progressed) self->cv.notify_one();
    }

    // miniaudio's threads. A "stopped" we did not cause starts the grace period and a "started"
    // ends it. "rerouted" does not: miniaudio reroutes a stopped device, and starts it afterwards
    // only when it was playing.
    static void on_notification(const ma_device_notification* n) {
        auto* self = (Impl*)n->pDevice->pUserData;
        {
            std::lock_guard<std::mutex> lk(self->m);
            if (n->type == ma_device_notification_type_stopped && !self->closing) {
                self->stalled = true;
                self->stalled_until = Clock::now() + kRerouteGrace;
            } else if (n->type == ma_device_notification_type_started && self->stalled) {
                self->stalled = false;
            } else {
                return;
            }
            self->changed = true;
        }
        self->cv.notify_one();
    }

    // ---- control thread ----
    bool open_device(int rate, std::string* error) {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_s16;
        cfg.playback.channels = 1;
        cfg.sampleRate = (ma_uint32)rate;
        cfg.dataCallback = &Impl::on_data;
        cfg.notificationCallback = &Impl::on_notification;
        cfg.pUserData = this;
        ma_result r = ma_device_init(nullptr, &cfg, &dev);
        if (r != MA_SUCCESS) {
            *error = std::string("ma_device_init failed: ") + ma_result_description(r) + " (" + std::to_string((int)r) + ")";
            return false;
        }
        open = true;
        {
            std::lock_guard<std::mutex> lk(m);
            dev_rate = rate;
        }
        if ((r = ma_device_start(&dev)) != MA_SUCCESS) {
            close_device();
            *error = std::string("ma_device_start failed: ") + ma_result_description(r) + " (" + std::to_string((int)r) + ")";
            return false;
        }
        return true;
    }

    // Stopping a playback device first plays out what it has buffered.
    void close_device() {
        if (!open) return;
        {
            std::lock_guard<std::mutex> lk(m);
            dev_rate = 0;
        }
        closing = true;
        ma_device_uninit(&dev);
        closing = false;
        open = false;
        {
            std::lock_guard<std::mutex> lk(m);
            stalled = false;
        }
    }

    // While the device is stalled only shutdown closes it; nothing else here stops, starts,
    // closes or reopens it.
    void run() {
        auto idle_since = Clock::now();
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);
                auto wake = [&] { return changed || quit; };
                if (stalled) cv.wait_until(lk, stalled_until, wake);
                else if (open && queue.empty()) cv.wait_until(lk, idle_since + kIdleClose, wake);
                else cv.wait(lk, wake);
            }
            Freed freed;
            bool close = false, quitting = false, was_lost = false;
            int open_rate = 0;
            {
                auto lk = lock_with_room(freed);
                changed = false;
                const auto now = Clock::now();
                const bool busy = !queue.empty();
                if (quit) {
                    drop_front_locked(queue.size(), freed);
                    quitting = close = true;
                } else if (stalled && now >= stalled_until) {
                    stalled = false;
                    if (open) {
                        close = true;
                        was_lost = head < queue.size() && queue[head]->started;
                        drop_front_locked(head + (was_lost ? 1 : 0), freed);
                    }
                }
                reap_locked(freed);
                if (busy) idle_since = now;
                const bool decide = !quitting && !stalled;
                if (decide && !queue.empty()) {
                    if (!open || close) {
                        open_rate = queue.front()->rate;
                    } else if (queue.front()->rate != dev_rate) {
                        close = true;
                        open_rate = queue.front()->rate;
                    }
                } else if (decide && open && now >= idle_since + kIdleClose) {
                    close = true;
                }
            }
            if (was_lost) publish_error("playback device stopped");
            if (close) close_device();
            if (open_rate > 0) {
                std::string err;
                if (!open_device(open_rate, &err)) {
                    {
                        auto lk = lock_with_room(freed);
                        size_t n = 0;
                        while (n < queue.size() && queue[n]->rate == open_rate && !queue[n]->started) n++;
                        drop_front_locked(n, freed);
                        changed = true;
                    }
                    publish_error("playback device cannot be opened: " + err);
                }
            }
            flush();
            if (quitting) return;
        }
    }
};

Player::Player(EventHub& hub) : p_(std::make_unique<Impl>(hub)) {}
Player::~Player() { shutdown(); }

uint64_t Player::enqueue(int rate, std::vector<int16_t> pcm) {
    Impl& m = *p_;
    auto c = std::make_unique<Clip>();
    c->rate = rate;
    c->pcm = std::move(pcm);
    Freed freed;
    uint64_t id;
    bool rejected;
    {
        auto lk = m.lock_with_room(freed);
        id = c->id = ++m.next_id;
        rejected = m.quit;
        if (rejected) m.events.push_back({"stopped", id});
        else m.queue.push_back(std::move(c));
        m.changed = true;
    }
    if (rejected) m.flush();
    else m.cv.notify_one();
    return id;
}

void Player::stop() {
    Impl& m = *p_;
    Freed freed;
    {
        auto lk = m.lock_with_room(freed);
        m.drop_front_locked(m.queue.size(), freed);
        m.changed = true;
    }
    m.cv.notify_one();
    m.flush();
}

std::string Player::status_json() {
    Impl& m = *p_;
    int64_t queued = 0;
    bool playing;
    {
        std::lock_guard<std::mutex> lk(m.m);
        for (const auto& c : m.queue) queued += c->started ? 0 : 1;
        playing = m.head < m.queue.size() && m.queue[m.head]->started;
    }
    return JsonObject().boolean("playing", playing).num("queued", queued).dump();
}

void Player::shutdown() {
    Impl& m = *p_;
    {
        std::lock_guard<std::mutex> lk(m.m);
        m.quit = true;
    }
    m.cv.notify_one();
    if (m.th.joinable()) m.th.join();
}

}  // namespace mintbot::agent
