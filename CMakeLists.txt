cmake_minimum_required(VERSION 3.18)
# VERSION: release version of the binaries (manifest, ready line, /v1/health).
# MINTBOT_API_VERSION: bump only on incompatible changes to the REST / SSE API.
project(mintbot-voice-agent VERSION 0.2.2 LANGUAGES C CXX)
set(MINTBOT_API_VERSION 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ggml: the exact version the SenseVoice / FSMN-VAD GGUF graphs were validated
# against (ggml 0.15.1 inside llama.cpp 8086439a). Only the ggml/ subdirectory
# is built. Point FETCHCONTENT_SOURCE_DIR_GGML at a local ggml/ to build offline.
include(FetchContent)

# MINTBOT_PORTABLE: release build for any x86-64 CPU and no VC++ runtime install.
# ggml-cpu is built once per instruction-set level (ggml-cpu-<variant>.dll) and
# the best one for the machine is picked at start-up; the C runtime is linked
# statically and OpenMP (vcomp140.dll, no static form) is replaced by ggml's
# own thread pool. Off: a static build tuned for this machine (GGML_NATIVE).
option(MINTBOT_PORTABLE "Build for redistribution (CPU variants, static CRT)" OFF)
if(MINTBOT_PORTABLE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "" FORCE)
    set(APM_VSCRT mt)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(APM_VSCRT md)
endif()
# Both builds use ggml's thread pool, so CPU measured on a developer build is
# what users get: MSVC's OpenMP workers keep spinning after every graph, which
# made an always-on process cost 2-3x the CPU.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# llama.cpp's top-level CMake turns this on and the reference outputs were
# produced that way; ggml on its own defaults it to OFF, which changes the
# matmul kernels and can flip borderline CTC tokens.
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
FetchContent_Declare(ggml
    URL      https://github.com/ggml-org/llama.cpp/archive/8086439a4cea94c71a5dfb8fe4ad1546aebd640f.tar.gz
    URL_HASH SHA256=1984103666eb25bd45110a40cba22b9d4286116f26e51bbc76f6f41dc86bc7b5
    SOURCE_SUBDIR ggml
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(ggml)

# WebRTC AudioProcessing (noise suppression / AGC2 / high-pass; AEC later):
# the freedesktop standalone release, built with meson (pip install meson) from
# the same developer shell. It needs C++20 under MSVC.
option(MINTBOT_WITH_APM "Build the WebRTC audio-processing front end" ON)
if(MINTBOT_WITH_APM)
    find_program(MESON_EXECUTABLE meson)
    if(MESON_EXECUTABLE)
        set(MESON ${MESON_EXECUTABLE})
    else()
        find_package(Python3 COMPONENTS Interpreter)
        set(_no_meson 1)
        if(Python3_EXECUTABLE)
            execute_process(COMMAND ${Python3_EXECUTABLE} -c "import mesonbuild" RESULT_VARIABLE _no_meson OUTPUT_QUIET ERROR_QUIET)
        endif()
        if(_no_meson)
            message(FATAL_ERROR "meson is required for MINTBOT_WITH_APM: pip install meson, put it on PATH or pass "
                                "-DMESON_EXECUTABLE=<path> (or configure with -DMINTBOT_WITH_APM=OFF). "
                                "Python found: '${Python3_EXECUTABLE}'")
        endif()
        set(MESON ${Python3_EXECUTABLE} -m mesonbuild.mesonmain)
    endif()
    include(ExternalProject)
    set(APM_PREFIX ${CMAKE_BINARY_DIR}/webrtc-apm)
    set(APM_SRC ${CMAKE_BINARY_DIR}/webrtc-apm-src)
    set(APM_LIB ${APM_PREFIX}/lib/libwebrtc-audio-processing-2.a)
    ExternalProject_Add(webrtc_apm_ep
        URL      https://freedesktop.org/software/pulseaudio/webrtc-audio-processing/webrtc-audio-processing-2.1.tar.xz
        URL_HASH SHA256=ae9302824b2038d394f10213cab05312c564a038434269f11dbf68f511f9f9fe
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR ${APM_SRC}
        CONFIGURE_COMMAND ${MESON} setup <BINARY_DIR> <SOURCE_DIR> --buildtype=release --default-library=static
                          -Dcpp_std=c++20 -Db_vscrt=${APM_VSCRT} --prefix=${APM_PREFIX} --libdir=lib
        BUILD_COMMAND ${MESON} compile -C <BINARY_DIR>
        INSTALL_COMMAND ${MESON} install -C <BINARY_DIR>
        BUILD_BYPRODUCTS ${APM_LIB})
    # abseil is a meson subproject; the public API needs its headers but meson does not
    # install them, so copy them next to the installed headers. Nothing may be created
    # inside APM_SRC beforehand: meson skips fetching a subproject whose directory exists.
    ExternalProject_Add_Step(webrtc_apm_ep install_abseil_headers
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/subprojects/abseil-cpp-20240722.0/absl
                ${APM_PREFIX}/include/abseil/absl
        DEPENDEES install)
    set(APM_INCLUDES ${APM_PREFIX}/include/webrtc-audio-processing-2 ${APM_PREFIX}/include/abseil)
    file(MAKE_DIRECTORY ${APM_INCLUDES})
    add_library(webrtc_apm STATIC IMPORTED GLOBAL)
    set_target_properties(webrtc_apm PROPERTIES IMPORTED_LOCATION ${APM_LIB})
    target_include_directories(webrtc_apm INTERFACE ${APM_INCLUDES})
    target_compile_definitions(webrtc_apm INTERFACE WEBRTC_LIBRARY_IMPL __STDC_FORMAT_MACROS=1)
    if(WIN32)
        target_compile_definitions(webrtc_apm INTERFACE WEBRTC_WIN)
        target_link_libraries(webrtc_apm INTERFACE winmm)
    endif()
    add_dependencies(webrtc_apm webrtc_apm_ep)
endif()

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
endif()

add_library(mcommon STATIC src/common/cpu_backend.cpp src/common/utf8.cpp)
target_include_directories(mcommon PUBLIC src)
target_link_libraries(mcommon PUBLIC ggml)
if(WIN32)
    target_link_libraries(mcommon PUBLIC shell32)  # CommandLineToArgvW
endif()

add_library(mvad STATIC src/vad/fsmn_vad_stream.cpp)
target_include_directories(mvad PUBLIC src PRIVATE third_party)
target_link_libraries(mvad PUBLIC mcommon)

add_library(masr STATIC src/asr/sensevoice.cpp)
target_include_directories(masr PUBLIC src)
target_link_libraries(masr PUBLIC mcommon)

add_library(mspk STATIC src/spk/campplus.cpp src/spk/speaker_gate.cpp)
target_include_directories(mspk PUBLIC src)
target_link_libraries(mspk PUBLIC mcommon)

add_executable(spk-cli tools/spk_cli.cpp)
target_link_libraries(spk-cli PRIVATE mspk mvad mdsp)

add_library(mdsp STATIC src/dsp/apm.cpp)
target_include_directories(mdsp PUBLIC src)
set_target_properties(mdsp PROPERTIES CXX_STANDARD 20)
if(MINTBOT_WITH_APM)
    target_compile_definitions(mdsp PRIVATE MINTBOT_HAVE_APM)
    target_link_libraries(mdsp PRIVATE webrtc_apm)
    # AEC3's own class (modules/audio_processing/aec3) is not an installed header;
    # it is needed to switch on its linear-filter output
    target_include_directories(mdsp PRIVATE ${APM_SRC}/webrtc)
    target_compile_definitions(mdsp PRIVATE WEBRTC_APM_DEBUG_DUMP=0)  # as the library is built
endif()

add_library(mpipeline STATIC src/pipeline/segmenter.cpp src/pipeline/pipeline.cpp)
target_include_directories(mpipeline PUBLIC src)
target_link_libraries(mpipeline PUBLIC mvad masr mdsp)

add_library(magent STATIC src/agent/agent.cpp src/agent/audio_source.cpp)
target_include_directories(magent PUBLIC src PRIVATE third_party)
target_link_libraries(magent PUBLIC mpipeline mspk)

add_executable(mintbot-voice-agent app/voice_agent.cpp)
target_include_directories(mintbot-voice-agent PRIVATE third_party)
target_link_libraries(mintbot-voice-agent PRIVATE magent)
target_compile_definitions(mintbot-voice-agent PRIVATE MINTBOT_VERSION="${PROJECT_VERSION}"
                           MINTBOT_API_VERSION=${MINTBOT_API_VERSION})
if(WIN32)
    target_link_libraries(mintbot-voice-agent PRIVATE ws2_32)
endif()

add_executable(vad-stream-cli tools/vad_stream_cli.cpp)
target_link_libraries(vad-stream-cli PRIVATE mvad)

add_executable(apm-cli tools/apm_cli.cpp)
target_link_libraries(apm-cli PRIVATE mdsp mcommon)

add_executable(asr-cli tools/asr_cli.cpp)
target_link_libraries(asr-cli PRIVATE masr)

add_executable(pipeline-cli tools/pipeline_cli.cpp)
target_link_libraries(pipeline-cli PRIVATE mpipeline)

# R2-0 echo-cancellation experiments
add_executable(aec-offline tools/aec_offline.cpp)
target_link_libraries(aec-offline PRIVATE mdsp mcommon)

add_executable(aec-rec tools/aec_rec.cpp)
target_include_directories(aec-rec PRIVATE third_party)
target_link_libraries(aec-rec PRIVATE mcommon)
