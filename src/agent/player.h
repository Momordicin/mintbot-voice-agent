// Speech playback on the system default output device: PCM16 mono clips from a FIFO
// queue, played back to back, with a "playback" event when each starts, ends or is stopped.
#pragma once

#include "agent/event_hub.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mintbot::agent {

class Player {
public:
    explicit Player(EventHub& hub);
    ~Player();
    Player(const Player&) = delete;
    Player& operator=(const Player&) = delete;

    // Appends a clip of s16 mono samples at `rate` Hz; returns its id (1, 2, ...). Any thread.
    uint64_t enqueue(int rate, std::vector<int16_t> pcm);
    // Stops the clip being played and drops the queued ones. Any thread.
    void stop();
    std::string status_json();  // {"playing", "queued"}
    // Stops playback and releases the device; later clips are reported stopped at once.
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace mintbot::agent
