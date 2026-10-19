#include "common/utf8.h"

#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace mintbot {

#ifdef _WIN32
namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string narrow(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

}  // namespace

void use_utf8_argv(int& argc, char**& argv) {
    static std::vector<std::string> storage;
    static std::vector<char*> ptrs;
    int n = 0;
    LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!w) return;
    storage.clear();
    for (int i = 0; i < n; i++) storage.push_back(narrow(w[i]));
    LocalFree(w);
    ptrs.clear();
    for (auto& s : storage) ptrs.push_back(s.data());
    ptrs.push_back(nullptr);
    argc = n;
    argv = ptrs.data();
}

FILE* fopen_utf8(const std::string& path, const char* mode) {
    return _wfopen(widen(path).c_str(), widen(mode).c_str());
}

std::filesystem::path utf8_path(const std::string& path) { return std::filesystem::path(widen(path)); }

std::string getenv_utf8(const char* name) {
    const wchar_t* v = _wgetenv(widen(name).c_str());
    return v ? narrow(v) : std::string();
}

std::string error_text(const std::error_code& ec) {
    if (ec.category() != std::system_category()) return ec.message();
    wchar_t* buf = nullptr;
    DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, (DWORD)ec.value(), 0, (LPWSTR)&buf, 0, nullptr);
    std::string s = n && buf ? narrow(buf) : "error " + std::to_string(ec.value());
    if (buf) LocalFree(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

#else

void use_utf8_argv(int&, char**&) {}
FILE* fopen_utf8(const std::string& path, const char* mode) { return fopen(path.c_str(), mode); }
std::filesystem::path utf8_path(const std::string& path) { return std::filesystem::path(path); }
std::string getenv_utf8(const char* name) {
    const char* v = getenv(name);
    return v ? v : "";
}
std::string error_text(const std::error_code& ec) { return ec.message(); }

#endif

}  // namespace mintbot
