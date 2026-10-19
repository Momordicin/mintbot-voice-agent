#include "common/cpu_backend.h"

#include "common/utf8.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace mintbot {
namespace {

std::once_flag g_load_once;

// ggml_backend_load() builds a narrow std::filesystem::path, which Windows reads
// in the ANSI code page, so load the DLL with a wide path and register it ourselves.
bool load_forced(const std::string& path) {
#ifdef _WIN32
    HMODULE h = LoadLibraryW(utf8_path(path).c_str());
    if (!h) { fprintf(stderr, "cpu_backend: cannot load %s (error %lu)\n", path.c_str(), GetLastError()); return false; }
    auto score = (int (*)())GetProcAddress(h, "ggml_backend_score");
    if (score && score() == 0) {
        fprintf(stderr, "cpu_backend: cannot load %s: not supported on this CPU\n", path.c_str());
        FreeLibrary(h);
        return false;
    }
    auto init = (ggml_backend_reg_t(*)())GetProcAddress(h, "ggml_backend_init");
    if (!init) { fprintf(stderr, "cpu_backend: cannot load %s: no ggml_backend_init\n", path.c_str()); FreeLibrary(h); return false; }
    ggml_backend_register(init());  // the module stays loaded for the life of the process
    fprintf(stderr, "cpu_backend: loaded CPU backend from %s\n", path.c_str());
    return true;
#else
    if (ggml_backend_load(path.c_str())) return true;
    fprintf(stderr, "cpu_backend: cannot load %s\n", path.c_str());
    return false;
#endif
}

void load_backends() {
    std::call_once(g_load_once, [] {
        if (std::string forced = getenv_utf8("MINTBOT_GGML_CPU_LIB"); !forced.empty()) {
            load_forced(forced);
        } else {
            ggml_backend_load_all();  // no-op for backends that are linked in
        }
    });
}

}  // namespace

ggml_backend_t cpu_backend_init() {
    load_backends();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) fprintf(stderr, "cpu_backend: no CPU backend available\n");
    return be;
}

void cpu_backend_set_threads(ggml_backend_t backend, int n_threads) {
    static ggml_backend_set_n_threads_t fn = [backend] {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        return (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    }();
    if (fn) fn(backend, n_threads);
}

void cpu_backend_set_abort(ggml_backend_t backend, ggml_abort_callback cb, void* data) {
    static ggml_backend_set_abort_callback_t fn = [backend] {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        return (ggml_backend_set_abort_callback_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
    }();
    if (fn) fn(backend, cb, data);
}

std::string cpu_backend_description() {
    load_backends();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return dev ? std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")" : std::string();
}

}  // namespace mintbot
