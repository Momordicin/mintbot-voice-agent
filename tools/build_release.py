#!/usr/bin/env python3
"""One release, from a clean checkout: portable build -> models re-exported from the
official weights (tools/models.lock.json) -> packages -> tests against those
packages -> SHA256SUMS.txt and RELEASE_NOTES.md. Used by
.github/workflows/release.yml and runnable locally (MSVC, Ninja, CMake, Node >= 22,
Python with tools/requirements-release.txt):

    python tools/build_release.py --out dist [--tag v0.1.1] [--build-dir build-p] [--skip-build]
        [--source ROLE=DIR ...]   (original model files already on disk, still hash-checked)

Writes dist/<version>/ (see tools/package_release.py) plus SHA256SUMS.txt, and
dist/RELEASE_NOTES.md. Nothing from models/ or any local recording is used.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    env = kw.get("env")
    if env and "PATH" in {k.upper() for k in env}:
        # Windows looks executables up on this process's PATH, not on the child's env
        path = next(v for k, v in env.items() if k.upper() == "PATH")
        cmd = [shutil.which(cmd[0], path=path) or cmd[0]] + cmd[1:]
    return subprocess.run(cmd, check=True, **kw)


def project_version():
    text = open(os.path.join(ROOT, "CMakeLists.txt"), encoding="utf-8").read()
    v = re.search(r"project\(mintbot-voice-agent VERSION ([0-9.]+)", text)
    api = re.search(r"set\(MINTBOT_API_VERSION (\d+)\)", text)
    if not v or not api:
        sys.exit("version or MINTBOT_API_VERSION not found in CMakeLists.txt")
    return v.group(1), int(api.group(1))


def without_gnu_link(env):
    """Moves Git's usr/bin to the end of PATH: its link.exe (a GNU file tool) would
    otherwise shadow MSVC's linker, which meson then refuses (Git Bash, e.g. a CI
    `shell: bash` step, puts it first)."""
    key = next((k for k in env if k.upper() == "PATH"), "PATH")
    parts = env.get(key, "").split(os.pathsep)
    gnu = [p for p in parts if os.path.normcase(os.path.normpath(p)).endswith(os.path.normcase(os.path.join("git", "usr", "bin")))]
    env[key] = os.pathsep.join([p for p in parts if p not in gnu] + gnu)
    link = shutil.which("link", path=env[key]) or ""
    if "msvc" not in link.lower():
        print(f"warning: the first link.exe on PATH is not MSVC's: {link or 'none'}", flush=True)
    return env


def msvc_env():
    """The environment of a VS developer prompt: the current one if cl.exe is on PATH,
    else the one vcvars64.bat sets up; Git's GNU link.exe moved out of the way."""
    if shutil.which("cl"):
        return without_gnu_link(dict(os.environ))
    candidates = []
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.exists(vswhere):
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            candidates.append(out)
    candidates += sorted(glob.glob(r"C:\Program Files\Microsoft Visual Studio\*\*"), reverse=True)
    for vs in candidates:
        bat = os.path.join(vs, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if os.path.exists(bat):
            out = subprocess.run(f'cmd /c ""{bat}" >nul && set"', capture_output=True, text=True, shell=True).stdout
            return without_gnu_link(dict(line.split("=", 1) for line in out.splitlines() if "=" in line))
    sys.exit("MSVC not found: run from a Developer PowerShell for VS 2022")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", help="the git tag being released; must be v<CMake project VERSION>")
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build-p"))
    ap.add_argument("--skip-build", action="store_true", help="reuse --build-dir as it is")
    ap.add_argument("--source", action="append", default=[], help="passed to tools/export_models.py")
    args = ap.parse_args()
    # paths may hold characters the console code page cannot (cp1252 on an English
    # system): print them escaped instead of failing
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    version, api = project_version()
    if args.tag and args.tag != f"v{version}":
        sys.exit(f"tag {args.tag} does not match CMakeLists.txt version {version}")
    out = os.path.abspath(args.out)
    lock = json.load(open(os.path.join(ROOT, "tools", "models.lock.json"), encoding="utf-8"))

    # 1. portable build
    if not args.skip_build:
        env = msvc_env()
        run(["cmake", "-S", ROOT, "-B", args.build_dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DMINTBOT_PORTABLE=ON"], env=env)
        run(["cmake", "--build", args.build_dir], env=env)
    bin_dir = os.path.join(args.build_dir, "bin")

    # 2. models from the official weights, byte-identical to the validated ones
    models = os.path.join(ROOT, "build-models", "out")
    exp = run([sys.executable, os.path.join(ROOT, "tools", "export_models.py"), "--out", models]
              + sum((["--source", s] for s in args.source), []), stdout=subprocess.PIPE,
              env={**os.environ, "PYTHONUTF8": "1"}, encoding="utf-8")  # the smoke WAV path comes back verbatim
    print(exp.stdout, end="")
    smoke_wav = exp.stdout.strip().splitlines()[-1]

    # 3. packages
    run([sys.executable, os.path.join(ROOT, "tools", "package_release.py"), "--bin", bin_dir,
         "--build", args.build_dir, "--models", models, "--models-version", lock["models_version"], "--out", out])
    rel = os.path.join(out, version)
    manifest = json.load(open(os.path.join(rel, "manifest.json"), encoding="utf-8"))
    if manifest["version"] != version or manifest["api_version"] != api:
        sys.exit(f"built agent reports {manifest['version']} / API {manifest['api_version']}, CMakeLists.txt says {version} / {api}")

    # 4. tests: the client against a fake agent, and install / repair / tamper / supervise
    #    on exactly these packages, replaying the VAD model's own example recording
    client = os.path.join(ROOT, "client")
    npm = shutil.which("npm") or sys.exit("npm not found")
    run([npm, "ci", "--no-audit", "--no-fund"], cwd=client)
    run([npm, "test"], cwd=client, env={**os.environ, "MINTBOT_TEST_MANIFEST": os.path.join(rel, "manifest.json"),
                                         "MINTBOT_TEST_WAV": smoke_wav, "MINTBOT_TEST_REQUIRE": "1"})

    # 5. checksums and notes
    files = sorted(f for f in os.listdir(rel) if f != "SHA256SUMS.txt")
    with open(os.path.join(rel, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        for name in files:
            f.write(f"{sha256(os.path.join(rel, name))}  {name}\n")
    client_tgz = next(n for n in files if n.endswith(".tgz"))
    repo = os.environ.get("GITHUB_REPOSITORY", "<owner>/<repo>")
    base = f"https://github.com/{repo}/releases/download/v{version}"
    notes = f"""mintbot-voice-agent {version} (API {api}), models {manifest['packages']['models']['version']}, client {client_tgz[len('mintbot-voice-agent-client-'):-4]}

Windows x64. Models rebuilt from the official ModelScope weights and verified byte-identical
to the validated ones (tools/models.lock.json).

Install from a Node.js (>= 22) project:

    pnpm add {base}/{client_tgz}

```ts
import {{ API_VERSION, installVoiceAgent, VoiceAgentSupervisor }} from "@mintbot/voice-agent-client";
const inst = await installVoiceAgent({{ manifest: "{base}/manifest.json", root, apiVersion: API_VERSION }});
```

| File | SHA-256 |
| --- | --- |
""" + "".join(f"| `{n}` | `{sha256(os.path.join(rel, n))}` |\n" for n in files)
    with open(os.path.join(out, "RELEASE_NOTES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(notes)
    print(f"\nrelease {version} in {rel}:")
    for n in files + ["SHA256SUMS.txt"]:
        print(f"  {n}  {os.path.getsize(os.path.join(rel, n)) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
