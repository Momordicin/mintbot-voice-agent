// Hotword correction of recognized text, and detection of the agent being called by name.
// List format follows asr-hotword (github.com/HaujetZhao/asr-hotword, MIT): one entry per
// line, "target | alias | alias", # comments, longest non-overlapping matches. A comment
// "# <character> <date> <id>" starts that character's section.
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

struct HotwordRole {
    std::string id, name;
    size_t entries = 0;
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

inline std::vector<std::string> split_space(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

inline bool is_ascii(const std::string& s) {
    for (unsigned char c : s)
        if (c >= 0x80) return false;
    return true;
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

inline bool ends_sentence(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == '\n' || cp == 0x3002 || cp == 0xff01 || cp == 0xff1f ||
           cp == 0x2026;  // 。！？…
}

inline bool is_pause(uint32_t cp) {
    return cp == ',' || cp == ';' || cp == ':' || cp == 0x3001 || cp == 0xff0c || cp == 0xff1b || cp == 0xff1a;  // 、，；：
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

// [start, end) of each sentence, the terminating punctuation left out
inline std::vector<std::pair<size_t, size_t>> sentences(const std::string& s) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t start = 0;
    for (size_t i = 0; i < s.size();) {
        size_t len;
        if (ends_sentence(decode(s, i, len))) {
            out.push_back({start, i});
            start = i + len;
        }
        i += len;
    }
    out.push_back({start, s.size()});
    return out;
}

// An English misspelling is replaced only where it cannot be part of an ordinary sentence:
// at the start followed by a pause ("Eith, could you"), at the end ("Good night Eith."),
// or as its own sentence ("Eith.").
inline bool english_position_ok(const std::string& s, const std::vector<std::pair<size_t, size_t>>& sents, size_t pos,
                                size_t len) {
    for (const auto& [a, b] : sents) {
        if (pos < a || pos + len > b) continue;
        const std::string before = trim(s.substr(a, pos - a)), after = trim(s.substr(pos + len, b - pos - len));
        size_t n;
        return after.empty() || (before.empty() && is_pause(decode(after, 0, n)));
    }
    return false;
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
    void parse(const std::string& text) {
        using namespace hotwords_detail;
        entries_.clear();
        roles_.clear();
        role_ = -1;
        int section = -1;
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty()) continue;
            if (line[0] == '#') {
                const std::vector<std::string> w = split_space(line.substr(1));
                if (w.size() < 3) continue;  // an ordinary comment
                section = find_role(w.back());
                if (section < 0) {
                    std::string name;
                    for (size_t i = 0; i + 2 < w.size(); i++) name += (i ? " " : "") + w[i];
                    roles_.push_back({w.back(), name, 0});
                    section = (int)roles_.size() - 1;
                }
                continue;
            }
            const size_t cut = line.find("~~~");  // asr-hotword's blacklist: not supported, ignored
            const std::vector<std::string> words = split_bar(cut == std::string::npos ? line : line.substr(0, cut));
            if (words.empty()) continue;
            Entry e;
            e.target = words[0];
            e.target_low = lower(words[0]);
            for (size_t i = 1; i < words.size(); i++) {
                const std::string a = lower(words[i]);
                if (a != e.target_low && std::find(e.aliases.begin(), e.aliases.end(), a) == e.aliases.end())
                    e.aliases.push_back(a);
            }
            e.role = section;
            if (section >= 0) roles_[section].entries++;
            entries_.push_back(std::move(e));
        }
        if (!roles_.empty()) role_ = 0;
    }

    bool set_role(const std::string& id) {
        role_ = find_role(id);
        return role_ >= 0;
    }

    std::string role() const { return role_ < 0 ? std::string() : roles_[role_].id; }
    const std::vector<HotwordRole>& roles() const { return roles_; }
    size_t size() const { return entries_.size(); }

    HotwordResult apply(const std::string& text) const {
        using namespace hotwords_detail;
        HotwordResult r{text, {}};
        if (entries_.empty() || text.empty()) return r;
        const std::string low = lower(text);
        const auto sents = sentences(text);
        struct Cand {
            size_t pos, len;
            size_t entry;
            bool correct;
        };
        std::vector<char> taken(text.size(), 0);  // locked or replaced
        std::vector<Cand> chosen;
        auto is_free = [&](size_t pos, size_t len) {
            return std::find(taken.begin() + pos, taken.begin() + pos + len, char(1)) == taken.begin() + pos + len;
        };
        auto find = [&](bool current, bool correct, std::vector<Cand>& out) {
            for (size_t i = 0; i < entries_.size(); i++) {
                if ((role_ >= 0 && entries_[i].role == role_) != current) continue;
                auto add = [&](const std::string& w) {
                    for (size_t pos = low.find(w); pos != std::string::npos; pos = low.find(w, pos + 1)) {
                        if (!at_boundary(low, pos, w.size()) || !is_free(pos, w.size())) continue;
                        if (!correct && is_ascii(w) && !english_position_ok(text, sents, pos, w.size())) continue;
                        out.push_back({pos, w.size(), i, correct});
                    }
                };
                if (correct) add(entries_[i].target_low);
                else
                    for (const auto& a : entries_[i].aliases) add(a);
            }
        };
        auto take = [&](std::vector<Cand> cands) {
            std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
                if (x.len != y.len) return x.len > y.len;
                if (x.pos != y.pos) return x.pos < y.pos;
                if (x.correct != y.correct) return x.correct;
                return x.entry < y.entry;
            });
            for (const auto& c : cands) {
                if (!is_free(c.pos, c.len)) continue;
                std::fill(taken.begin() + c.pos, taken.begin() + c.pos + c.len, char(1));
                chosen.push_back(c);
            }
        };
        for (const bool current : {true, false}) {
            if (current && role_ < 0) continue;
            std::vector<Cand> cands;
            find(current, true, cands);
            find(current, false, cands);
            take(cands);
        }
        std::sort(chosen.begin(), chosen.end(), [](const Cand& x, const Cand& y) { return x.pos > y.pos; });
        for (const auto& c : chosen) {
            if (c.correct) continue;
            const std::string& target = entries_[c.entry].target;
            const std::string from = text.substr(c.pos, c.len);
            if (from == target) continue;
            r.text.replace(c.pos, c.len, target);
            r.matches.insert(r.matches.begin(), {from, target});
        }
        return r;
    }

private:
    struct Entry {
        std::string target, target_low;
        std::vector<std::string> aliases;  // lower-case, the target left out
        int role = -1;                     // section index; -1 before the first section
    };

    int find_role(const std::string& id) const {
        for (size_t i = 0; i < roles_.size(); i++)
            if (roles_[i].id == id) return (int)i;
        return -1;
    }

    std::vector<Entry> entries_;
    std::vector<HotwordRole> roles_;
    int role_ = -1;
};


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
