#!/usr/bin/env python3
"""Rebuilds models/*.gguf from the official ModelScope weights, as pinned in
tools/models.lock.json: downloads the original files (or takes them from
--source), checks every SHA-256, fetches the pinned SenseVoice export scripts,
converts, and checks that each GGUF is byte-identical to the validated one.

    python tools/export_models.py --out build-models
    python tools/export_models.py --out build-models --source asr=<dir with model.pt, am.mvn, ...>

Needs tools/requirements-release.txt. Prints the smoke-test WAV path last.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path, expected, what):
    got = sha256(path)
    if got != expected:
        sys.exit(f"{what}: SHA-256 {got}, expected {expected} ({path})")


def download_originals(model, cache):
    from modelscope import snapshot_download  # build-time dependency only

    return snapshot_download(model["modelscope"], cache_dir=cache, allow_patterns=list(model["files"]))


def exporter_script(name, spec, work):
    if "path" in spec:
        return os.path.join(ROOT, spec["path"])
    path = os.path.join(work, os.path.basename(spec["url"]))
    if not os.path.exists(path) or sha256(path) != spec["sha256"]:
        with urllib.request.urlopen(spec["url"], timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    check(path, spec["sha256"], f"exporter {name}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for the GGUF files")
    ap.add_argument("--cache", default=os.path.join(ROOT, "build-models", "modelscope"),
                    help="ModelScope download cache")
    ap.add_argument("--source", action="append", default=[], metavar="ROLE=DIR",
                    help="use original files already on disk for a model role (asr, vad, speaker); still hash-checked")
    args = ap.parse_args()
    # paths may hold characters the console code page cannot (cp1252 on an English
    # system): print them escaped instead of failing
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    lock = json.load(open(os.path.join(ROOT, "tools", "models.lock.json"), encoding="utf-8"))
    sources = dict(s.split("=", 1) for s in args.source)
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_exporters")
    os.makedirs(work, exist_ok=True)

    model_dirs = {}
    for m in lock["models"]:
        src = sources.get(m["role"]) or download_originals(m, args.cache)
        for name, expected in m["files"].items():
            check(os.path.join(src, name), expected, f"{m['modelscope']}/{name}")
        model_dirs[m["role"]] = src
        out = os.path.join(args.out, m["out"])
        if os.path.exists(out) and sha256(out) == m["out_sha256"]:
            print(f"{m['out']}: up to date")
            continue
        script = exporter_script(m["exporter"], lock["exporters"][m["exporter"]], work)
        subst = {"{out}": out, **{"{" + n + "}": os.path.join(src, n) for n in m["files"]}}
        cmd = [sys.executable, script] + [subst.get(a, a) for a in m["args"]]
        print(f"{m['out']}: exporting from {m['modelscope']}", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        check(out, m["out_sha256"], m["out"])
        print(f"{m['out']}: identical to the validated model")

    smoke = lock["smoke_wav"]
    print(os.path.join(model_dirs[smoke["role"]], smoke["file"]))


if __name__ == "__main__":
    main()
