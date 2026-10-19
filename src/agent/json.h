// Tiny JSON object builder for event payloads (output only).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace mintbot::agent {

inline std::string json_escape(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

class JsonObject {
public:
    JsonObject& str(const char* k, const std::string& v) { return raw(k, json_escape(v)); }
    JsonObject& num(const char* k, int64_t v) { return raw(k, std::to_string(v)); }
    JsonObject& num(const char* k, double v) { char b[32]; snprintf(b, sizeof b, "%.3f", v); return raw(k, b); }
    JsonObject& boolean(const char* k, bool v) { return raw(k, v ? "true" : "false"); }
    JsonObject& raw(const char* k, const std::string& json) {
        body_ += (body_.empty() ? "" : ",") + json_escape(k) + ":" + json;
        return *this;
    }
    std::string dump() const { return "{" + body_ + "}"; }

private:
    std::string body_;
};

}  // namespace mintbot::agent
