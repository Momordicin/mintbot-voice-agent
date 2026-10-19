#!/usr/bin/env python3
"""Portable build: every ggml-cpu-<variant>.dll must give the same results.

For each variant DLL in --bin (forced with MINTBOT_GGML_CPU_LIB), and for the
automatic choice, compares against a reference build (--ref-bin, e.g. the
native developer build):
  * vad-stream-cli segments           -> must be identical
  * pipeline-cli texts                -> reported (kernels differ per ISA level)
  * spk-cli embeddings                -> cosine vs reference
Variants the CPU cannot run must be refused, not crash.

    python tests/check_cpu_variants.py --bin build-p/bin --ref-bin build/bin --models models a.wav b.wav
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np


def run(exe, args, env=None):
    t = time.perf_counter()
    r = subprocess.run([exe] + args, capture_output=True, env=env)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"), time.perf_counter() - t


def outputs(bindir, models, wavs, env):
    ext = ".exe" if os.name == "nt" else ""
    vad, asr, spk = (os.path.join(models, m) for m in ("fsmn-vad.gguf", "sensevoice-small-q8_0.gguf", "campplus-zh-cn.gguf"))
    res = {"segments": [], "texts": [], "emb": [], "pipe_s": 0.0, "spk_s": 0.0, "stderr": ""}
    for w in wavs:
        rc, out, err, _ = run(os.path.join(bindir, "vad-stream-cli" + ext), ["-m", vad, "-a", w], env)
        if rc != 0:
            res["error"] = f"vad-stream-cli exit {rc}: {err.strip()[-200:]}"
            return res
        res["segments"].append(json.loads(out)["segments"])
        rc, out, err, dt = run(os.path.join(bindir, "pipeline-cli" + ext), ["--vad", vad, "-m", asr, "-a", w], env)
        res["texts"] += [json.loads(l)["text"] for l in out.splitlines() if l.strip()]
        res["pipe_s"] += dt
        res["stderr"] += err
        rc, out, err, dt = run(os.path.join(bindir, "spk-cli" + ext), ["-m", spk, "-a", w], env)
        res["emb"] += [np.asarray(json.loads(l)["embedding"]) for l in out.splitlines() if l.strip()]
        res["spk_s"] += dt
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--ref-bin", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()
    a.bin, a.ref_bin, a.models = map(os.path.abspath, (a.bin, a.ref_bin, a.models))

    ref = outputs(a.ref_bin, a.models, a.wavs, dict(os.environ))
    print(f"reference ({a.ref_bin}): {len(ref['texts'])} utterances, pipeline {ref['pipe_s']:.2f}s, speaker {ref['spk_s']:.2f}s\n")
    variants = sorted(glob.glob(os.path.join(a.bin, "ggml-cpu-*.dll")) + glob.glob(os.path.join(a.bin, "libggml-cpu-*.so")))
    failed = False
    for v in [None] + variants:
        env = dict(os.environ)
        env.pop("MINTBOT_GGML_CPU_LIB", None)
        if v:
            env["MINTBOT_GGML_CPU_LIB"] = v
        name = os.path.basename(v) if v else "auto"
        o = outputs(a.bin, a.models, a.wavs, env)
        if "error" in o:
            refused = "cannot load" in o["error"] or "no CPU backend" in o["error"]
            print(f"{'OK  ' if refused else 'FAIL'}  {name:26s} not usable on this CPU ({'refused cleanly' if refused else o['error']})")
            failed |= not refused
            continue
        seg_ok = o["segments"] == ref["segments"]
        same_text = sum(x == y for x, y in zip(o["texts"], ref["texts"]))
        cos = [float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)) for x, y in zip(o["emb"], ref["emb"])]
        ok = seg_ok and len(o["texts"]) == len(ref["texts"]) and min(cos) >= 0.999
        failed |= not ok
        picked = ""
        if not v:
            picked = next((l.split("from")[-1].strip() for l in o["stderr"].splitlines() if "loaded CPU backend" in l), "?")
            picked = " -> " + os.path.basename(picked)
        print(f"{'PASS' if ok else 'FAIL'}  {name:26s} segments {'=' if seg_ok else '!='} ref, texts {same_text}/{len(ref['texts'])} identical, "
              f"speaker cos min {min(cos):.6f}, pipeline {o['pipe_s']:.2f}s, speaker {o['spk_s']:.2f}s{picked}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
