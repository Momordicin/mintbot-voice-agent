# Third-party notices

## FunASR — MIT License

- `third_party/funasr/e2e-vad.h`: copied from
  https://github.com/modelscope/FunASR (`runtime/onnxruntime/src/e2e-vad.h`,
  commit `28a889cba5a00333dfb2dee115bd1cfab51f25e1`), with one added read-only
  accessor (`PendingEndFrame`, marked `mintbot`).
- `src/vad/fsmn_vad_stream.cpp`: the online feature caching follows
  `runtime/onnxruntime/src/fsmn-vad-online.cpp` from the same repository.
- `src/pipeline/segmenter.cpp`: follows `Audio::Split` (ASR_OFFLINE path) in
  `runtime/onnxruntime/src/audio.cpp` from the same repository.

Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
MIT License (https://opensource.org/licenses/MIT).

## SenseVoice — MIT License

`src/vad/fsmn_vad_stream.cpp` reuses the fbank / LFR / CMVN front end and the
FSMN graph from https://github.com/FunAudioLLM/SenseVoice
(`runtime/llama.cpp/funasr-common/funasr_vad.h`); `src/asr/sensevoice.cpp`
reuses the SenseVoice front end, SAN-M encoder graph and CTC decode from
`runtime/llama.cpp/funasr-sensevoice/funasr-sensevoice.cpp` and
`runtime/llama.cpp/sensevoice-server/sensevoice-server.cpp`.

Copyright (c) 2025 FunASR. MIT License.

## ggml — MIT License

Fetched at build time from https://github.com/ggml-org/llama.cpp (commit
`8086439a4cea94c71a5dfb8fe4ad1546aebd640f`, `ggml/` subdirectory).

## miniaudio — public domain or MIT No Attribution

`third_party/miniaudio/miniaudio.h`, version 0.11.25 from
https://github.com/mackron/miniaudio (tag `0.11.25`), unmodified. License:
`third_party/miniaudio/LICENSE`.

## cpp-httplib — MIT License

`third_party/httplib/httplib.h`, version 0.53.0 from
https://github.com/yhirose/cpp-httplib (tag `v0.53.0`), unmodified.
Copyright (c) 2017 yhirose. License: `third_party/httplib/LICENSE`.

## WebRTC AudioProcessing — BSD 3-Clause

Downloaded and built at configure/build time (not vendored):
https://freedesktop.org/software/pulseaudio/webrtc-audio-processing/
release 2.1 (`webrtc-audio-processing-2.1.tar.xz`, SHA-256
`ae9302824b2038d394f10213cab05312c564a038434269f11dbf68f511f9f9fe`), and
statically linked into the executables. Its license is `COPYING` in the
tarball (copied to `build/webrtc-apm-src/COPYING`) and must ship with
binaries.

It pulls in abseil-cpp 20240722.0 (Apache License 2.0) as a meson
subproject, which is linked in as well.

## CAM++ (3D-Speaker) — Apache License 2.0

`src/spk/campplus.cpp` is a C++ / ggml re-implementation of the CAM++ network
in `funasr/models/campplus/{model,components}.py` (FunASR, MIT, "Copyright
FunASR"), which FunASR states it modified from 3D-Speaker
(https://github.com/modelscope/3D-Speaker, Apache License 2.0; text in
`third_party/licenses/3D-Speaker-LICENSE.txt`, taken at commit
`065629c313eaf1a01c65c640c46d77e61e9607b4`; 3D-Speaker has no NOTICE file).
Changes: rewritten in C++ for inference only, BatchNorm folded into a
per-channel affine, the Kaldi fbank front end reimplemented; the file says so
in its header. `src/spk/speaker_gate.cpp` follows the sv_chunk windowing and
cosine matching of `funasr/bin/realtime_ws.py` (FunASR, MIT). The weights
`iic/speech_campplus_sv_zh-cn_16k-common` are Apache-2.0 per their model card.

## Model weights

Not part of this repository; the model package of a release
(`mintbot-voice-models-*.zip`, see its `README.md`) contains GGUF conversions of:

- FSMN-VAD `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` and SenseVoiceSmall
  `iic/SenseVoiceSmall` (q8_0) by Tongyi Lab / Alibaba Group (FunASR,
  FunAudioLLM): FunASR Model Open Source License Agreement v1.1
  (`third_party/licenses/FunASR-MODEL_LICENSE-v1.1.txt`, which requires the
  source, authors and model names to be kept); their ModelScope cards also
  state Apache-2.0.
- CAM++ `iic/speech_campplus_sv_zh-cn_16k-common` (3D-Speaker): Apache-2.0.

## License texts in the release packages

`licenses/` in the program package holds the texts for ggml, WebRTC
AudioProcessing (`COPYING`, `webrtc/LICENSE`, `webrtc/PATENTS`), abseil-cpp,
miniaudio, cpp-httplib, FunASR, SenseVoice and 3D-Speaker; the model package
holds the FunASR Model Open Source License Agreement v1.1 and Apache-2.0.
