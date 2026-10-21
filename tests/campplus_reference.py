#!/usr/bin/env python3
"""Compares spk-cli (CAM++ on ggml) with FunASR's PyTorch CAM++, window by window.

    python tests/campplus_reference.py --cli build/bin/spk-cli.exe --model models/campplus-zh-cn.gguf \
        --funasr-model <speech_campplus_sv_zh-cn_16k-common dir> a.wav b.wav ...
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf


def sv_chunks(x, fs=16000, dur=1.5, shift=0.75):
    L, S = int(dur * fs), int(shift * fs)
    out, last = [], 0
    for st in range(0, len(x), S):
        ed = min(st + L, len(x))
        if ed <= last:
            break
        last = ed
        c = x[max(0, ed - L):ed]
        out.append(np.pad(c, (0, L - len(c))) if len(c) < L else c)
    return out


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--funasr-model", required=True)
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    from funasr import AutoModel
    ref = AutoModel(model=args.funasr_model, disable_update=True, device="cpu", disable_log=True)

    failed, utt = False, {}
    for wav in args.wavs:
        x, sr = sf.read(wav, dtype="int16")
        assert sr == 16000 and x.ndim == 1, f"{wav}: need 16 kHz mono"
        xf = (x / 32768.0).astype(np.float32)
        r = ref.generate(input=sv_chunks(xf), cache={}, is_final=True, disable_pbar=True)
        tref = np.concatenate([np.asarray(o["spk_embedding"].detach().cpu()) for o in r], axis=0)
        out = subprocess.run([args.cli, "-m", args.model, "-a", wav], capture_output=True, text=True, check=True).stdout
        mine = np.array([json.loads(l)["embedding"] for l in out.splitlines() if l.strip()])
        if mine.shape != tref.shape:
            print(f"FAIL  {os.path.basename(wav)}: shapes ggml {mine.shape} vs torch {tref.shape}")
            failed = True
            continue
        cos = [float(unit(a) @ unit(b)) for a, b in zip(mine, tref)]
        rel = float(np.abs(mine - tref).max() / np.abs(tref).max())
        ok = min(cos) >= args.min_cos
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(wav):24s} windows={len(cos):3d}  cos min={min(cos):.6f} "
              f"mean={np.mean(cos):.6f}  max|diff|/max|ref|={rel:.2e}")
        utt[os.path.basename(wav)] = (unit(np.mean([unit(v) for v in mine], 0)), unit(np.mean([unit(v) for v in tref], 0)))

    names = list(utt)
    if len(names) > 1:
        print("\nutterance-level cosine (ggml | torch):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = utt[names[i]], utt[names[j]]
                print(f"  {names[i]:24s} vs {names[j]:24s}  {a[0] @ b[0]:.4f} | {a[1] @ b[1]:.4f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
