#!/usr/bin/env python3
"""Builds the release packages of the voice agent from a portable build.

    python tools/package_release.py --bin build-p/bin --models models --out dist

Writes to <out>/<version>/:
  mintbot-voice-agent-<version>-win-x64.zip  executable, ggml DLLs, licenses
  mintbot-voice-models-<models-version>.zip  the three GGUF models, model licenses
  manifest.json                              versions, sizes and SHA-256 of both
                                             archives and of every file inside
  mintbot-voice-agent-client-<cv>.tgz        the Node.js client package (client/),
                                             built and packed with npm; the app
                                             depends on it (unless --no-client).
                                             Versioned on its own (client/package.json);
                                             it must speak the agent's API_VERSION.

    python tools/package_release.py --bin build-p/bin --out dist --client-only

packs only the client, into <out>/<client version>/.

Binaries and models are separate archives so that an update of the program
does not download the models again. Only files on the allowlists below are
packaged: recordings, voiceprints and anything else in the build or model
directories can never end up in a release.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {  # role -> file name in --models
    "vad": "fsmn-vad.gguf",
    "asr": "sensevoice-small-q8_0.gguf",
    "speaker": "campplus-zh-cn.gguf",
}

MODELS_README = """# MintBot voice models

| File | Model | Source | License |
| --- | --- | --- | --- |
| `fsmn-vad.gguf` | FSMN-VAD (`speech_fsmn_vad_zh-cn-16k-common-pytorch`) | https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch, converted to GGUF with `runtime/llama.cpp/export_vad_gguf.py` of https://github.com/FunAudioLLM/SenseVoice | FunASR Model Open Source License Agreement v1.1; Apache-2.0 per the ModelScope card |
| `sensevoice-small-q8_0.gguf` | SenseVoiceSmall (`iic/SenseVoiceSmall`), q8_0 | https://modelscope.cn/models/iic/SenseVoiceSmall, converted with `runtime/llama.cpp/export_sensevoice_gguf.py --wtype q8_0` of the same repository | FunASR Model Open Source License Agreement v1.1; Apache-2.0 per the ModelScope card |
| `campplus-zh-cn.gguf` | CAM++ (`speech_campplus_sv_zh-cn_16k-common`) | https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common (3D-Speaker, https://github.com/modelscope/3D-Speaker), converted with `tools/export_campplus_gguf.py` of mintbot-voice-agent | Apache-2.0 |

Authors: Tongyi Lab / Alibaba Group (FunASR, FunAudioLLM, 3D-Speaker).
The model names above are kept as required by Section 2.2 of the FunASR Model
Open Source License Agreement (`licenses/FunASR-MODEL_LICENSE-v1.1.txt`).
These files are format conversions of the official weights; the q8_0 file is
quantised. No weights were trained or fine-tuned.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(path):
    if not os.path.isfile(path):
        sys.exit(f"missing: {path}")
    return path


def agent_version(exe):
    out = subprocess.run([exe, "--version"], capture_output=True, check=True).stdout.decode()
    return json.loads(out)


def write_zip(path, entries):
    """entries: [(archive name, source path or bytes)] -> {name: sha256}"""
    files = {}
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w") as z:
        for name, src in entries:
            if isinstance(src, bytes):
                data = src
                files[name] = hashlib.sha256(data).hexdigest()
                z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
            else:
                files[name] = sha256(src)
                # GGUF weights barely compress; store them so installs extract quickly
                method = zipfile.ZIP_STORED if name.endswith(".gguf") else zipfile.ZIP_DEFLATED
                z.write(src, name, method)
    os.replace(tmp, path)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="portable build output (build-p/bin)")
    ap.add_argument("--build", help="portable build tree for the license files (default: parent of --bin)")
    ap.add_argument("--models")
    ap.add_argument("--models-version", default="1", help="bump when any model file changes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-client", action="store_true", help="skip building and packing client/")
    ap.add_argument("--client-only", action="store_true",
                    help="only the client tarball, into <out>/<client version>/ (the agent and model "
                         "packages are unchanged); --bin is used to check the API version")
    args = ap.parse_args()
    # paths may hold characters the console code page cannot (cp1252 on an English
    # system): print them escaped instead of failing
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    build = args.build or os.path.dirname(os.path.abspath(args.bin))

    exe = need(os.path.join(args.bin, "mintbot-voice-agent.exe"))
    ver = agent_version(exe)
    if args.client_only:
        pack_client(os.path.join(args.out, client_info()[0]), ver["api_version"])
        return
    if not args.models:
        sys.exit("--models is required")
    cpu = sorted(f for f in os.listdir(args.bin) if f.startswith("ggml-cpu-") and f.endswith(".dll"))
    if not cpu:
        sys.exit("no ggml-cpu-*.dll: --bin must be a MINTBOT_PORTABLE=ON build")

    lic = [  # (archive name, source)
        ("LICENSE.txt", os.path.join(ROOT, "LICENSE")),
        ("THIRD_PARTY_NOTICES.md", os.path.join(ROOT, "THIRD_PARTY_NOTICES.md")),
        ("licenses/ggml-LICENSE.txt", os.path.join(build, "_deps", "ggml-src", "LICENSE")),
        ("licenses/webrtc-audio-processing-COPYING.txt", os.path.join(build, "webrtc-apm-src", "COPYING")),
        ("licenses/webrtc-LICENSE.txt", os.path.join(build, "webrtc-apm-src", "webrtc", "LICENSE")),
        ("licenses/webrtc-PATENTS.txt", os.path.join(build, "webrtc-apm-src", "webrtc", "PATENTS")),
        ("licenses/abseil-cpp-LICENSE.txt",
         os.path.join(build, "webrtc-apm-src", "subprojects", "abseil-cpp-20240722.0", "LICENSE")),
        ("licenses/miniaudio-LICENSE.txt", os.path.join(ROOT, "third_party", "miniaudio", "LICENSE")),
        ("licenses/cpp-httplib-LICENSE.txt", os.path.join(ROOT, "third_party", "httplib", "LICENSE")),
        ("licenses/FunASR-LICENSE.txt", os.path.join(ROOT, "third_party", "licenses", "FunASR-LICENSE.txt")),
        ("licenses/SenseVoice-LICENSE.txt", os.path.join(ROOT, "third_party", "licenses", "SenseVoice-LICENSE.txt")),
        ("licenses/3D-Speaker-LICENSE.txt", os.path.join(ROOT, "third_party", "licenses", "3D-Speaker-LICENSE.txt")),
    ]
    agent_entries = [(f, need(os.path.join(args.bin, f)))
                     for f in ["mintbot-voice-agent.exe", "ggml.dll", "ggml-base.dll"] + cpu]
    agent_entries += [(n, need(p)) for n, p in lic]

    model_entries = [(f, need(os.path.join(args.models, f))) for f in MODELS.values()]
    model_entries += [
        ("README.md", MODELS_README.encode("utf-8")),
        ("licenses/FunASR-MODEL_LICENSE-v1.1.txt",
         need(os.path.join(ROOT, "third_party", "licenses", "FunASR-MODEL_LICENSE-v1.1.txt"))),
        ("licenses/Apache-2.0.txt",  # the standard text, as shipped by abseil
         need(os.path.join(build, "webrtc-apm-src", "subprojects", "abseil-cpp-20240722.0", "LICENSE"))),
    ]

    out = os.path.join(args.out, ver["version"])
    os.makedirs(out, exist_ok=True)
    pkgs = {}
    for key, name, entries, extra in (
        ("agent", f"mintbot-voice-agent-{ver['version']}-win-x64.zip", agent_entries, {"version": ver["version"]}),
        ("models", f"mintbot-voice-models-{args.models_version}.zip", model_entries, {"version": args.models_version}),
    ):
        path = os.path.join(out, name)
        files = write_zip(path, entries)
        for f in files:  # defence in depth: user data must never be packaged
            if f.lower().endswith((".wav", ".json", ".jsonl")) or "voiceprint" in f.lower():
                sys.exit(f"refusing to package {f}")
        pkgs[key] = {**extra, "file": name, "size": os.path.getsize(path), "sha256": sha256(path), "files": files}
        print(f"{name}: {len(files)} files, {os.path.getsize(path) / 2**20:.1f} MB")

    manifest = {
        "schema": 1,
        "name": "mintbot-voice-agent",
        "version": ver["version"],
        "api_version": ver["api_version"],
        "platform": "win-x64",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry": "mintbot-voice-agent.exe",
        "models": MODELS,
        "packages": pkgs,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {os.path.join(out, 'manifest.json')}")

    if not args.no_client:
        pack_client(out, ver["api_version"])


CLIENT = os.path.join(ROOT, "client")


def client_info():
    """(package version, API_VERSION in client/src/protocol.ts)"""
    with open(os.path.join(CLIENT, "package.json"), encoding="utf-8") as f:
        version = json.load(f)["version"]
    with open(os.path.join(CLIENT, "src", "protocol.ts"), encoding="utf-8") as f:
        m = re.search(r"export const API_VERSION = (\d+);", f.read())
    if not m:
        sys.exit("API_VERSION not found in client/src/protocol.ts")
    return version, int(m.group(1))


def pack_client(out, agent_api):
    """Builds client/ and writes its npm tarball to out. The client is versioned on its
    own; it must speak the agent's API_VERSION."""
    version, api = client_info()
    if api != agent_api:
        sys.exit(f"client speaks API {api}, the agent {agent_api}")
    npm = shutil.which("npm")
    if not npm:
        sys.exit("npm not found (or pass --no-client)")
    if not os.path.isdir(os.path.join(CLIENT, "node_modules")):
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=CLIENT, check=True)
    subprocess.run([npm, "run", "build"], cwd=CLIENT, check=True, stdout=subprocess.DEVNULL)
    os.makedirs(out, exist_ok=True)
    packed = json.loads(subprocess.run([npm, "pack", "--json", "--pack-destination", os.path.abspath(out)],
                                       cwd=CLIENT, check=True, capture_output=True).stdout)
    print(f"client {version}: {os.path.join(out, packed[0]['filename'])} ({len(packed[0]['files'])} files)")


if __name__ == "__main__":
    main()
