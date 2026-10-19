#!/usr/bin/env python3
"""Calibrate the owner-speaker threshold from real recordings.

  --owner : recordings of the owner (ideally several sessions / days)
  --other : recordings of anything else the microphone hears (other people,
            games, videos, TV ...)
  --voiceprint : optional; the agent's --speaker-db file. Scores every owner
            utterance against the voiceprint actually in use.

Each file is cut into utterances with the FSMN-VAD and embedded with CAM++
(spk-cli --vad). Owner utterances are scored leave-one-file-out against a
centre built from the other owner files (one owner file: first half enrolls,
second half is scored). Prints EER, the thresholds for 1% / 0.1% false accepts
and accuracy per utterance length.

    python tests/calibrate_speaker.py --cli build/bin/spk-cli.exe --model models/campplus-zh-cn.gguf \
        --vad models/fsmn-vad.gguf --owner me1.wav me2.wav --other room.wav game.wav
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BINS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 10**9)]


def utterances(args, wav):
    cmd = [args.cli, "-m", args.model, "--vad", args.vad, "-a", wav]
    if not args.no_apm:  # --record files are raw; the agent scores audio after its default preprocessing
        cmd += ["--apm", "--ns", str(args.ns)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = [json.loads(l) for l in out.splitlines() if l.strip()]
    return [(np.asarray(r["embedding"]), r["end_ms"] - r["start_ms"]) for r in rows]


def center(embs):
    c = np.mean([e / np.linalg.norm(e) for e in embs], axis=0)
    return c / np.linalg.norm(c)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--owner", nargs="+", required=True)
    ap.add_argument("--other", nargs="+", required=True)
    ap.add_argument("--voiceprint", help="score against this agent voiceprint file (--speaker-db) instead of "
                                         "building centres from the owner files")
    ap.add_argument("--no-apm", action="store_true", help="the agent runs with --no-apm")
    ap.add_argument("--ns", type=int, default=2, help="the agent's --ns level (default 2)")
    args = ap.parse_args()
    args.cli = os.path.abspath(args.cli)

    owner = {f: utterances(args, f) for f in args.owner}
    other = [u for f in args.other for u in utterances(args, f)]
    print(f"owner: {sum(len(v) for v in owner.values())} utterances in {len(owner)} files; other: {len(other)} utterances")

    genuine, impostor = [], []  # (score, duration_ms)
    files = list(owner)
    if args.voiceprint:
        vp = json.load(open(args.voiceprint, encoding="utf-8"))
        c = np.asarray(vp["center"], dtype=np.float64)
        c /= np.linalg.norm(c)
        print(f"voiceprint: {vp['utterances']} utterances, {vp['speech_ms'] / 1000:.1f} s ({args.voiceprint})")
        folds = [([], [u for f in files for u in owner[f]])]
    elif len(files) == 1:
        us = owner[files[0]]
        half = len(us) // 2
        folds = [(us[:half], us[half:])]
    else:
        folds = [([u for g in files if g != f for u in owner[g]], owner[f]) for f in files]
    for enroll, test in folds:
        if enroll:
            c = center([e for e, _ in enroll])
        genuine += [(float(e @ c / np.linalg.norm(e)), d) for e, d in test]
        impostor += [(float(e @ c / np.linalg.norm(e)), d) for e, d in other]
    g = np.array([s for s, _ in genuine])
    i = np.array([s for s, _ in impostor])
    print(f"owner scores  : n={len(g)} mean={g.mean():.3f} min={g.min():.3f} p5={np.percentile(g, 5):.3f}")
    print(f"other scores  : n={len(i)} mean={i.mean():.3f} max={i.max():.3f} p95={np.percentile(i, 95):.3f}")

    grid = np.linspace(-0.2, 1.0, 1201)
    far = np.array([(i >= t).mean() for t in grid])
    frr = np.array([(g < t).mean() for t in grid])
    k = int(np.argmin(np.abs(far - frr)))
    if g.min() > i.max():
        mid = (g.min() + i.max()) / 2
        k = int(np.argmin(np.abs(grid - mid)))
        print(f"fully separated on this data: any threshold in ({i.max():.3f}, {g.min():.3f}); "
              f"suggest the midpoint {mid:.3f} (collect more data before trusting the margin)")
    else:
        print(f"EER ~ {(far[k] + frr[k]) / 2 * 100:.2f}% at threshold {grid[k]:.3f}")
    for target in (0.01, 0.001):
        ok = np.where(far <= target)[0]
        if len(ok):
            t = grid[ok[0]]
            print(f"false accepts <= {target * 100:g}%: threshold {t:.3f}, owner rejected {(g < t).mean() * 100:.1f}%")
    print(f"\nby utterance length at threshold {grid[k]:.3f}:")
    t = grid[k]
    for lo, hi in BINS:
        gs = [s for s, d in genuine if lo <= d < hi]
        is_ = [s for s, d in impostor if lo <= d < hi]
        name = f"{lo / 1000:g}-{hi / 1000:g}s" if hi < 10**9 else f">{lo / 1000:g}s"
        fr = f"{np.mean(np.array(gs) < t) * 100:5.1f}% of {len(gs)}" if gs else "    - "
        fa = f"{np.mean(np.array(is_) >= t) * 100:5.1f}% of {len(is_)}" if is_ else "    - "
        print(f"  {name:8s} owner rejected {fr:14s} other accepted {fa}")


if __name__ == "__main__":
    main()
