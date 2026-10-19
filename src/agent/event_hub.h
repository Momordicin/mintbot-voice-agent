// Fan-out of JSON events to SSE subscribers. Each subscriber has its own
// bounded queue; a slow subscriber loses its oldest events, never blocks the
// publisher.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mintbot::agent {

struct Event {
    uint64_t id;
    std::string type;
    std::string json;  // payload object
};

class EventHub {
public:
    class Subscription {
    public:
        // Waits up to `timeout` for the next event. Returns false on timeout or
        // after close(); `closed()` tells which.
        bool next(Event& out, std::chrono::milliseconds timeout) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, timeout, [&] { return closed_ || !q_.empty(); });
            if (q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }
        bool closed() {
            std::lock_guard<std::mutex> lk(m_);
            return closed_;
        }
        uint64_t dropped() {
            std::lock_guard<std::mutex> lk(m_);
            return dropped_;
        }

    private:
        friend class EventHub;
        void push(const Event& e, size_t cap) {
            {
                std::lock_guard<std::mutex> lk(m_);
                if (q_.size() >= cap) { q_.pop_front(); ++dropped_; }
                q_.push_back(e);
            }
            cv_.notify_one();
        }
        void close() {
            {
                std::lock_guard<std::mutex> lk(m_);
                closed_ = true;
            }
            cv_.notify_all();
        }
        std::mutex m_;
        std::condition_variable cv_;
        std::deque<Event> q_;
        bool closed_ = false;
        uint64_t dropped_ = 0;
    };

    explicit EventHub(size_t per_subscriber_cap = 256) : cap_(per_subscriber_cap) {}

    std::shared_ptr<Subscription> subscribe() {
        auto s = std::make_shared<Subscription>();
        std::lock_guard<std::mutex> lk(m_);
        if (closed_) s->close();
        subs_.push_back(s);
        return s;
    }
    void unsubscribe(const std::shared_ptr<Subscription>& s) {
        std::lock_guard<std::mutex> lk(m_);
        for (auto it = subs_.begin(); it != subs_.end(); ++it)
            if (*it == s) { subs_.erase(it); break; }
    }
    void publish(const std::string& type, const std::string& json) {
        std::lock_guard<std::mutex> lk(m_);
        Event e{++next_id_, type, json};
        for (auto& s : subs_) s->push(e, cap_);
    }
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        for (auto& s : subs_) s->close();
    }
    size_t subscribers() {
        std::lock_guard<std::mutex> lk(m_);
        return subs_.size();
    }

private:
    std::mutex m_;
    std::vector<std::shared_ptr<Subscription>> subs_;
    size_t cap_;
    uint64_t next_id_ = 0;
    bool closed_ = false;
};

}  // namespace mintbot::agent
