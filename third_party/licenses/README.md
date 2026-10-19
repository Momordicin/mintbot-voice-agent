# License texts shipped with release packages

Texts that are not available from the build tree are kept here and copied into
the release packages by `tools/package_release.py`.

| File | Source |
| --- | --- |
| `FunASR-LICENSE.txt` | MIT license of FunASR, from the `funasr` 1.4.16 wheel (`funasr-1.4.16.dist-info/licenses/LICENSE`) |
| `SenseVoice-LICENSE.txt` | MIT license of https://github.com/FunAudioLLM/SenseVoice (`LICENSE`) |
| `3D-Speaker-LICENSE.txt` | Apache License 2.0 of https://github.com/modelscope/3D-Speaker (`LICENSE` at commit `065629c313eaf1a01c65c640c46d77e61e9607b4`, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`); `src/spk/campplus.cpp` derives from its CAM++ code through FunASR |
| `FunASR-MODEL_LICENSE-v1.1.txt` | FunASR Model Open Source License Agreement v1.1, https://github.com/modelscope/FunASR/blob/58830eca4012644aac0c3218c3ccc7d98f003fda/MODEL_LICENSE (SHA-256 `7dba975a2069691db4992b0592d70828b330d2f8a30a71450f4e152a554e84f8`) |

The ggml, WebRTC AudioProcessing, abseil, miniaudio and cpp-httplib license
texts come from their sources in the build tree / `third_party/`.
