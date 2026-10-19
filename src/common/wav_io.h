// Minimal WAV I/O: 16 kHz mono PCM16 only. Paths are UTF-8.
#pragma once

#include <algorithm>
#include <cmath>
#include "common/utf8.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Streaming PCM16 writer. The header is rewritten every `sync_samples`
// samples, so the file stays playable if the process is killed.
class WavWriter {
public:
    ~WavWriter() { close(); }
    bool open(const std::string& path, uint32_t sync_samples = 16000 * 5) {
        close();
        f_ = mintbot::fopen_utf8(path, "wb");
        if (!f_) return false;
        sync_ = sync_samples;
        n_ = since_sync_ = 0;
        header();
        return true;
    }
    void write(const float* pcm, size_t n) {
        if (!f_) return;
        std::vector<int16_t> s(n);
        for (size_t i = 0; i < n; i++) s[i] = (int16_t)std::lround(std::clamp(pcm[i], -1.0f, 32767.0f / 32768.0f) * 32768.0f);
        fwrite(s.data(), 2, n, f_);
        n_ += (uint32_t)n;
        since_sync_ += (uint32_t)n;
        if (since_sync_ >= sync_) { header(); since_sync_ = 0; }
    }
    void close() {
        if (!f_) return;
        header();
        fclose(f_);
        f_ = nullptr;
    }
    bool is_open() const { return f_ != nullptr; }

private:
    void header() {
        long pos = ftell(f_);
        uint32_t data = n_ * 2, riff = 36 + data, fmt_len = 16, rate = 16000, byte_rate = 32000;
        uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
        fseek(f_, 0, SEEK_SET);
        fwrite("RIFF", 1, 4, f_); fwrite(&riff, 4, 1, f_); fwrite("WAVEfmt ", 1, 8, f_);
        fwrite(&fmt_len, 4, 1, f_); fwrite(&pcm, 2, 1, f_); fwrite(&ch, 2, 1, f_); fwrite(&rate, 4, 1, f_);
        fwrite(&byte_rate, 4, 1, f_); fwrite(&align, 2, 1, f_); fwrite(&bits, 2, 1, f_);
        fwrite("data", 1, 4, f_); fwrite(&data, 4, 1, f_);
        fseek(f_, pos > 44 ? pos : 44, SEEK_SET);
        fflush(f_);
    }
    FILE* f_ = nullptr;
    uint32_t n_ = 0, since_sync_ = 0, sync_ = 0;
};

inline bool write_wav16k(const char* path, const std::vector<float>& pcm) {
    WavWriter w;
    if (!w.open(path)) return false;
    w.write(pcm.data(), pcm.size());
    w.close();
    return true;
}

inline bool read_wav16k(const char* path, std::vector<float>& pcm) {
    FILE* f = mintbot::fopen_utf8(path, "rb");
    if (!f) return false;
    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) || memcmp(riff + 8, "WAVE", 4)) { fclose(f); return false; }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    bool ok = false;
    char id[4];
    uint32_t size;
    while (fread(id, 1, 4, f) == 4 && fread(&size, 4, 1, f) == 1) {
        if (!memcmp(id, "fmt ", 4)) {
            std::vector<uint8_t> b(size);
            if (fread(b.data(), 1, size, f) != size || size < 16) break;
            memcpy(&fmt, &b[0], 2); memcpy(&ch, &b[2], 2); memcpy(&rate, &b[4], 4); memcpy(&bits, &b[14], 2);
        } else if (!memcmp(id, "data", 4)) {
            if (fmt != 1 || ch != 1 || rate != 16000 || bits != 16) break;
            std::vector<int16_t> s(size / 2);
            size_t got = fread(s.data(), 2, s.size(), f);
            pcm.resize(got);
            for (size_t i = 0; i < got; i++) pcm[i] = s[i] / 32768.0f;
            ok = true;
            break;
        } else {
            fseek(f, size + (size & 1), SEEK_CUR);
        }
    }
    fclose(f);
    return ok;
}
