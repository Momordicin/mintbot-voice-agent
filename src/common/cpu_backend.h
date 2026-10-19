// CPU backend through the ggml registry, so the same code works whether
// ggml-cpu is linked in (developer build) or loaded at run time from the
// best-matching ggml-cpu-<variant>.dll next to the executable
// (MINTBOT_PORTABLE: GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS).
//
// MINTBOT_GGML_CPU_LIB=<path to a ggml-cpu-*.dll> forces one variant (tests).
#pragma once

#include "ggml-backend.h"

#include <string>

namespace mintbot {

ggml_backend_t cpu_backend_init();
void cpu_backend_set_threads(ggml_backend_t backend, int n_threads);
// ggml checks `cb` after every graph node; returning true stops the computation
// (ggml_backend_graph_compute() then returns GGML_STATUS_ABORTED).
void cpu_backend_set_abort(ggml_backend_t backend, ggml_abort_callback cb, void* data);
// e.g. "CPU (13th Gen Intel(R) Core(TM) ...)"; empty if no CPU backend could be loaded
std::string cpu_backend_description();

}  // namespace mintbot
