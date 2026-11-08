// Hotword correction of recognized text, and detection of the agent being called by name.
// List format and replacement rules follow asr-hotword (github.com/HaujetZhao/asr-hotword,
// MIT), with exact aliases only: no phoneme matching.
//
//   target | alias | alias ~~~ blacklisted | blacklisted     (one entry per line, # comments)
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mintbot::asr {

struct HotwordMatch {
    std::string from, to;
};

struct HotwordResult {
    std::string text;
    std::vector<HotwordMatch> matches;
};

struct Address {
    bool addressed = false;
    std::string rest;  // what follows (or precedes) the name
};

namespace hotwords_detail {

inline bool is_alnum(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }

inline std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

inline std::vector<std::string> split_bar(const std::string& s) {
    std::vector<std::string> out;
    size_t p = 0;
    for (;;) {
        size_t e = s.find('|', p);
        std::string w = trim(s.substr(p, e == std::string::npos ? std::string::npos : e - p));
        if (!w.empty()) out.push_back(w);
        if (e == std::string::npos) break;
        p = e + 1;
    }
    return out;
}

// code point at s[i] and its byte length (1 for invalid bytes)
inline uint32_t decode(const std::string& s, size_t i, size_t& len) {
    unsigned char c = (unsigned char)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + n > s.size()) n = 1;
    uint32_t cp = n == 1 ? c : c & (0x7f >> n);
    for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3f);
    len = (size_t)n;
    return cp;
}

// asr-hotword's semantic units: one CJK character, an ASCII word, a number
inline std::vector<std::pair<size_t, size_t>> tokens(const std::string& s) {
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < s.size();) {
        size_t len;
        uint32_t cp = decode(s, i, len);
        if (is_alnum(s[i])) {
            const bool digit = s[i] >= '0' && s[i] <= '9';
            size_t j = i + 1;
            while (j < s.size() && is_alnum(s[j]) && ((s[j] >= '0' && s[j] <= '9') == digit)) j++;
            out.push_back({i, j});
            i = j;
            continue;
        }
        if ((cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)) out.push_back({i, i + len});
        i += len;
    }
    return out;
}

// an ASCII word must not be cut out of a longer one ("I miss" does not match "I missed")
inline bool at_boundary(const std::string& s, size_t pos, size_t len) {
    if (is_alnum(s[pos]) && pos > 0 && is_alnum(s[pos - 1])) return false;
    if (is_alnum(s[pos + len - 1]) && pos + len < s.size() && is_alnum(s[pos + len])) return false;
    return true;
}

inline bool is_separator(uint32_t cp) {
    static const uint32_t seps[] = {' ', '\t', ',', '.', '!', '?', ';', ':', 0x3001, 0x3002, 0xff0c, 0xff01, 0xff1f,
                                    0xff1b, 0xff1a, 0x2026, 0xff0e};  // 、。，！？；：…．
    return std::find(std::begin(seps), std::end(seps), cp) != std::end(seps);
}

inline std::string trim_separators(const std::string& s, bool front, bool back) {
    size_t a = 0, b = s.size();
    while (front && a < b) {
        size_t len;
        if (!is_separator(decode(s, a, len))) break;
        a += len;
    }
    while (back && b > a) {
        size_t start = b - 1;
        while (start > a && ((unsigned char)s[start] & 0xc0) == 0x80) start--;
        size_t len;
        if (!is_separator(decode(s, start, len))) break;
        b = start;
    }
    return s.substr(a, b - a);
}

}  // namespace hotwords_detail

inline bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) return false;
        for (int k = 1; k < n; k++)
            if (((unsigned char)s[i + k] & 0xc0) != 0x80) return false;
        i += n;
    }
    return true;
}

class Hotwords {
public:
    static constexpr int kBlacklistWindow = 5;  // semantic units on each side, as asr-hotword

    // Each line is its own entry, so one target may have aliases with different blacklists.
    void parse(const std::string& text) {
        using namespace hotwords_detail;
        entries_.clear();
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            size_t b = line.find("~~~");
            std::vector<std::string> words = split_bar(b == std::string::npos ? line : line.substr(0, b));
            if (words.empty()) continue;
            Entry e;
            e.target = words[0];
            for (const auto& w : words) e.aliases.push_back(lower(w));
            if (b != std::string::npos)
                for (const auto& w : split_bar(line.substr(b + 3))) e.blacklist.push_back(lower(w));
            entries_.push_back(std::move(e));
        }
    }

    size_t size() const { return entries_.size(); }

    HotwordResult apply(const std::string& text) const {
        using namespace hotwords_detail;
        HotwordResult r{text, {}};
        if (entries_.empty() || text.empty()) return r;
        const std::string low = lower(text);
        const auto toks = tokens(text);
        struct Cand {
            size_t pos, len;
            const Entry* e;
        };
        std::vector<Cand> cands;
        for (const auto& e : entries_)
            for (const auto& a : e.aliases)
                for (size_t pos = low.find(a); pos != std::string::npos; pos = low.find(a, pos + 1))
                    if (at_boundary(low, pos, a.size()) && !blacklisted(low, toks, pos, a.size(), e))
                        cands.push_back({pos, a.size(), &e});
        // longer aliases first, then the earlier one; overlapping matches are dropped
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& x, const Cand& y) { return x.len != y.len ? x.len > y.len : x.pos < y.pos; });
        std::vector<Cand> chosen;
        for (const auto& c : cands) {
            bool overlap = false;
            for (const auto& k : chosen) overlap = overlap || (c.pos < k.pos + k.len && k.pos < c.pos + c.len);
            if (!overlap) chosen.push_back(c);
        }
        std::sort(chosen.begin(), chosen.end(), [](const Cand& x, const Cand& y) { return x.pos > y.pos; });
        for (const auto& c : chosen) {
            std::string from = text.substr(c.pos, c.len);
            if (from == c.e->target) continue;  // already written as the target
            r.text.replace(c.pos, c.len, c.e->target);
            r.matches.insert(r.matches.begin(), {from, c.e->target});
        }
        return r;
    }

private:
    struct Entry {
        std::string target;
        std::vector<std::string> aliases;    // lower-case, the target included
        std::vector<std::string> blacklist;  // lower-case
    };

    static bool blacklisted(const std::string& low, const std::vector<std::pair<size_t, size_t>>& toks, size_t pos,
                            size_t len, const Entry& e) {
        using namespace hotwords_detail;
        if (e.blacklist.empty()) return false;
        size_t from = pos, to = pos + len;
        int first = -1, last = -1;
        for (int i = 0; i < (int)toks.size(); i++)
            if (toks[i].second > pos && toks[i].first < pos + len) {
                if (first < 0) first = i;
                last = i;
            }
        if (first >= 0) {
            from = toks[std::max(0, first - kBlacklistWindow)].first;
            to = toks[std::min((int)toks.size() - 1, last + kBlacklistWindow)].second;
        }
        for (const auto& w : e.blacklist)
            for (size_t p = low.find(w, from); p != std::string::npos && p + w.size() <= to; p = low.find(w, p + 1))
                if (at_boundary(low, p, w.size())) return true;
        return false;
    }

    std::vector<Entry> entries_;
};

// The agent is called when the text starts with the name followed by punctuation (or
// nothing else), or ends with it after punctuation or after 你 / 您 ("我相信你爱弥斯").
// A name inside the sentence, or at the start with no break ("爱弥斯的技能"), is only
// talked about.
inline Address find_address(const std::string& text, const std::string& name) {
    using namespace hotwords_detail;
    Address a;
    if (name.empty()) return a;
    const std::string t = trim_separators(trim(text), true, true);
    if (t.compare(0, name.size(), name) == 0) {
        const std::string after = t.substr(name.size());
        size_t len;
        if (after.empty() || is_separator(decode(after, 0, len))) {
            a.addressed = true;
            a.rest = trim_separators(after, true, false);
            return a;
        }
    }
    if (t.size() > name.size() && t.compare(t.size() - name.size(), name.size(), name) == 0) {
        const std::string before = t.substr(0, t.size() - name.size());
        size_t start = before.size() - 1;
        while (start > 0 && ((unsigned char)before[start] & 0xc0) == 0x80) start--;
        size_t len;
        const uint32_t cp = decode(before, start, len);
        if (is_separator(cp) || cp == 0x4f60 || cp == 0x60a8) {  // 你, 您
            a.addressed = true;
            a.rest = is_separator(cp) ? trim_separators(before, false, true) : before;
        }
    }
    return a;
}

}  // namespace mintbot::asr
