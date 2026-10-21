#include "spk/speaker_gate.h"

#include "agent/json.h"
#include "common/utf8.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace mintbot::spk {
namespace {

constexpr const char* kModelTag = "campplus-zh-cn-16k-common";

int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Minimal reader for the file save_locked() writes.
bool find_number(const std::string& s, const char* key, double& out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos || (p = s.find(':', p)) == std::string::npos) return false;
    out = strtod(s.c_str() + p + 1, nullptr);
    return true;
}
bool find_string(const std::string& s, const char* key, std::string& out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos || (p = s.find('"', s.find(':', p))) == std::string::npos) return false;
    size_t e = s.find('"', p + 1);
    if (e == std::string::npos) return false;
    out = s.substr(p + 1, e - p - 1);
    return true;
}
bool find_array(const std::string& s, const char* key, std::vector<float>& out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos || (p = s.find('[', p)) == std::string::npos) return false;
    size_t e = s.find(']', p);
    if (e == std::string::npos) return false;
    const char* c = s.c_str() + p + 1;
    const char* end = s.c_str() + e;
    while (c < end) {
        char* next = nullptr;
        float v = strtof(c, &next);
        if (next == c) { ++c; continue; }
        out.push_back(v);
        c = next;
    }
    return true;
}

}  // namespace

bool SpeakerGate::load(const std::string& gguf, const SpeakerGateOptions& opts, std::string* error) {
    opts_ = opts;
    if (!cam_.load(gguf, opts.n_threads)) {
        if (error) *error = "cannot load speaker model " + gguf;
        return false;
    }
    if (opts.db_path.empty() || !std::filesystem::exists(utf8_path(opts.db_path))) return true;
    std::ifstream f(utf8_path(opts.db_path), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str(), model;
    std::vector<float> c;
    double utts = 0, ms = 0, dim = 0;
    if (!find_string(s, "model", model) || model != kModelTag || !find_number(s, "dim", dim) || !find_array(s, "center", c) ||
        (int)c.size() != cam_.dim() || (int)dim != cam_.dim()) {
        if (error) *error = "voiceprint file " + opts.db_path + " is not a " + kModelTag + " voiceprint";
        return false;
    }
    find_number(s, "utterances", utts);
    find_number(s, "speech_ms", ms);
    std::string device;
    find_string(s, "device", device);
    std::lock_guard<std::mutex> lk(m_);
    center_ = std::move(c);
    center_device_ = device;
    center_utterances_ = (int)utts;
    center_ms_ = (int64_t)ms;
    return true;
}

bool SpeakerGate::wants_windows() {
    std::lock_guard<std::mutex> lk(m_);
    return !center_.empty() || enrolling_;
}

bool SpeakerGate::enrolling() {
    std::lock_guard<std::mutex> lk(m_);
    return enrolling_;
}

uint64_t SpeakerGate::version() {
    std::lock_guard<std::mutex> lk(m_);
    return version_;
}

SpeakerDecision SpeakerGate::analyze(const float* pcm, size_t n, const WindowCache* pre) {
    SpeakerDecision d;
    const int64_t ms = (int64_t)n / 16;
    std::vector<float> center;
    bool enrolling;
    float threshold, floor;
    {
        std::lock_guard<std::mutex> lk(m_);
        center = center_;
        enrolling = enrolling_;
        limits_locked(threshold, floor);
    }
    d.threshold = threshold;
    d.enrolled = !center.empty();
    d.uncertain = ms < opts_.min_reliable_ms;
    if (!d.enrolled && !enrolling) return d;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> windows;
    bool ok = cam_.embed_utterance(pcm, n, windows, pre, &d.reused);
    d.embed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    d.windows = (int)windows.size();
    if (!ok) return d;  // too short to embed: let it through

    if (enrolling) {
        std::lock_guard<std::mutex> lk(m_);
        if (enrolling_) {
            enroll_windows_.insert(enroll_windows_.end(), windows.begin(), windows.end());
            ++enroll_utts_;
            enroll_ms_ += ms;
            d.enrolling = true;
        }
    }
    if (d.enrolled) {
        d.score = cosine(average_normalized(windows), center);
        d.is_owner = d.score >= threshold;
        d.transcribe = d.is_owner || (d.uncertain && !opts_.gate_uncertain && d.score >= floor);
    }
    if (d.enrolling) d.transcribe = true;
    return d;
}

void SpeakerGate::limits_locked(float& threshold, float& floor) const {
    threshold = opts_.threshold;
    floor = opts_.uncertain_floor;
    // unknown on either side (a source without a device, a voiceprint without one): as the same device
    if (device_.empty() || center_device_.empty() || device_ == center_device_) return;
    threshold = std::max(opts_.threshold * 0.5f, std::min(0.2f, opts_.threshold));
    floor = (opts_.uncertain_floor + threshold) / 2;
}

bool SpeakerGate::set_device(const std::string& name) {
    std::lock_guard<std::mutex> lk(m_);
    if (name == device_) return true;
    device_ = name;
    if (!enrolling_ || name == enroll_device_) return true;
    enrolling_ = false;
    enroll_device_changed_ = true;
    enroll_windows_.clear();
    enroll_utts_ = 0;
    enroll_ms_ = 0;
    return false;
}

EnrollState SpeakerGate::enroll_begin() {
    {
        std::lock_guard<std::mutex> lk(m_);
        enrolling_ = true;
        enroll_windows_.clear();
        enroll_utts_ = 0;
        enroll_ms_ = 0;
        enroll_device_ = device_;
        enroll_device_changed_ = false;
    }
    return enroll_state();
}

EnrollState SpeakerGate::enroll_state() {
    std::lock_guard<std::mutex> lk(m_);
    EnrollState s;
    s.active = enrolling_;
    s.utterances = enroll_utts_;
    s.speech_ms = enroll_ms_;
    s.ready = enroll_utts_ >= opts_.min_enroll_utterances && enroll_ms_ >= opts_.min_enroll_ms;
    return s;
}

bool SpeakerGate::enroll_commit(std::string* error) {
    std::lock_guard<std::mutex> lk(m_);
    if (!enrolling_) {
        if (error) *error = enroll_device_changed_ ? "enrollment stopped: the microphone changed" : "no enrollment in progress";
        return false;
    }
    if (enroll_utts_ < opts_.min_enroll_utterances || enroll_ms_ < opts_.min_enroll_ms) {
        if (error)
            *error = "need at least " + std::to_string(opts_.min_enroll_utterances) + " utterances and " +
                     std::to_string(opts_.min_enroll_ms / 1000) + " s of speech, have " + std::to_string(enroll_utts_) +
                     " and " + std::to_string(enroll_ms_ / 1000) + " s";
        return false;
    }
    auto old_center = center_;
    int old_u = center_utterances_;
    int64_t old_ms = center_ms_;
    std::string old_device = center_device_;
    center_ = average_normalized(enroll_windows_);
    center_utterances_ = enroll_utts_;
    center_ms_ = enroll_ms_;
    center_device_ = enroll_device_;
    if (!save_locked(error)) {
        center_ = std::move(old_center);
        center_utterances_ = old_u;
        center_ms_ = old_ms;
        center_device_ = std::move(old_device);
        return false;
    }
    enrolling_ = false;
    enroll_windows_.clear();
    ++version_;
    return true;
}

void SpeakerGate::enroll_cancel() {
    std::lock_guard<std::mutex> lk(m_);
    enrolling_ = false;
    enroll_windows_.clear();
    enroll_utts_ = 0;
    enroll_ms_ = 0;
}

bool SpeakerGate::forget(std::string* error) {
    std::lock_guard<std::mutex> lk(m_);
    // Files first: if one cannot be deleted nothing changes, so the gate never reports
    // "not enrolled" (and stops gating) while the voiceprint is still on disk. The
    // .tmp is what a failed save may have left behind; it holds a voiceprint too.
    if (!opts_.db_path.empty()) {
        for (const std::string& f : {opts_.db_path + ".tmp", opts_.db_path}) {
            std::error_code ec;
            std::filesystem::remove(utf8_path(f), ec);  // a missing file is not an error
            if (ec) {
                if (error) *error = "cannot delete " + f + ": " + error_text(ec);
                return false;
            }
        }
    }
    center_.clear();
    center_utterances_ = 0;
    center_ms_ = 0;
    center_device_.clear();
    ++version_;
    return true;
}

bool SpeakerGate::save_locked(std::string* error) {
    if (opts_.db_path.empty()) return true;
    std::string arr = "[";
    char b[32];
    for (size_t i = 0; i < center_.size(); i++) { snprintf(b, sizeof b, "%s%.8g", i ? "," : "", center_[i]); arr += b; }
    std::string json = agent::JsonObject()
                           .num("version", (int64_t)1)
                           .str("model", kModelTag)
                           .num("dim", (int64_t)center_.size())
                           .num("utterances", (int64_t)center_utterances_)
                           .num("speech_ms", center_ms_)
                           .str("device", center_device_)
                           .num("created_unix_ms", unix_ms())
                           .raw("center", arr + "]")
                           .dump();
    const std::string tmp = opts_.db_path + ".tmp";
    {
        std::ofstream f(utf8_path(tmp), std::ios::binary | std::ios::trunc);
        f << json << "\n";
        if (!f) {
            f.close();
            std::error_code ignored;
            std::filesystem::remove(utf8_path(tmp), ignored);  // a partial voiceprint must not stay behind
            if (error) *error = "cannot write " + tmp;
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(utf8_path(tmp), utf8_path(opts_.db_path), ec);  // replaces the old file atomically
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(utf8_path(tmp), ignored);  // nor the new one that could not be put in place
        if (error) *error = "cannot replace " + opts_.db_path + ": " + error_text(ec);
        return false;
    }
    return true;
}

std::string SpeakerGate::status_json() {
    std::lock_guard<std::mutex> lk(m_);
    float threshold, floor;
    limits_locked(threshold, floor);
    return agent::JsonObject()
        .boolean("enrolled", !center_.empty())
        .num("utterances", (int64_t)center_utterances_)
        .num("speech_ms", center_ms_)
        .str("enrolled_device", center_device_)
        .str("device", device_)
        .num("threshold", (double)opts_.threshold)
        .num("active_threshold", (double)threshold)  // the one applied on the current device
        .num("min_reliable_ms", (int64_t)opts_.min_reliable_ms)
        .boolean("gate_uncertain", opts_.gate_uncertain)
        .num("uncertain_floor", (double)opts_.uncertain_floor)
        .num("active_uncertain_floor", (double)floor)
        .boolean("enrolling", enrolling_)
        .num("enroll_utterances", (int64_t)enroll_utts_)
        .num("enroll_speech_ms", enroll_ms_)
        .boolean("enroll_ready", enroll_utts_ >= opts_.min_enroll_utterances && enroll_ms_ >= opts_.min_enroll_ms)
        .boolean("persistent", !opts_.db_path.empty())
        .dump();
}

}  // namespace mintbot::spk
