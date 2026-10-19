#!/usr/bin/env python3
"""Acceptance B: compare vad-stream-cli with FunASR's own streaming fsmn-vad.

Runs FunASR (PyTorch, test-only dependency) in streaming mode with the same
settings as the C++ default (chunk 200 ms, max_end_silence_time 800 ms,
max_single_segment_time 30000 ms), captures the encoder's per-frame output,
and compares segments and silence probabilities with the C++ port.

    python tests/funasr_reference_vad.py --cli build/bin/vad-stream-cli.exe \
        --model models/fsmn-vad.gguf --funasr-model <fsmn-vad model dir> a.wav ...
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def funasr_streaming(model, wav, chunk_ms, end_sil, max_seg):
    x, sr = sf.read(wav, dtype="float32")
    assert sr == 16000 and x.ndim == 1
    model.model.vad_opts.max_single_segment_time = max_seg
    probs = []
    hook = model.model.encoder.register_forward_hook(
        lambda m, i, o: probs.append((o[0] if isinstance(o, tuple) else o)[0, :, 0].detach().cpu().numpy()))
    step = chunk_ms * 16
    cache, segs, open_start = {}, [], None
    n = (len(x) + step - 1) // step
    for k in range(n):
        res = model.generate(input=x[k * step:(k + 1) * step], cache=cache, is_final=(k == n - 1),
                             chunk_size=chunk_ms, max_end_silence_time=end_sil, disable_pbar=True)
        for s, e in res[0]["value"]:
            if s != -1:
                open_start = s
            if e != -1:
                segs.append([open_start, e])
                open_start = None
    hook.remove()
    return segs, np.concatenate(probs) if probs else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--funasr-model", required=True)
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--end-sil", type=int, default=800)
    ap.add_argument("--max-seg", type=int, default=30000)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    from funasr import AutoModel
    ref_model = AutoModel(model=args.funasr_model, disable_update=True, device="cpu", disable_log=True)

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for wav in args.wavs:
            ref_segs, ref_p = funasr_streaming(ref_model, wav, args.chunk_ms, args.end_sil, args.max_seg)
            pf = os.path.join(tmp, "p.f32")
            out = subprocess.run([args.cli, "-m", args.model, "-a", wav, "--chunk-ms", str(args.chunk_ms),
                                  "--end-sil", str(args.end_sil), "--max-seg", str(args.max_seg), "--probs-out", pf],
                                 capture_output=True, text=True, check=True).stdout
            segs, p = json.loads(out)["segments"], np.fromfile(pf, dtype=np.float32)
            m = min(len(p), len(ref_p))
            dp = np.abs(p[:m] - ref_p[:m])
            flips = int(np.sum(((1 - p[:m]) >= p[:m] + 0.6) != ((1 - ref_p[:m]) >= ref_p[:m] + 0.6)))
            bdiff = max((abs(a - b) for s, r in zip(segs, ref_segs) for a, b in zip(s, r)), default=0)
            ok = len(segs) == len(ref_segs) and bdiff <= 10
            failed |= not ok
            print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(wav):16s} frames c++={len(p)} funasr={len(ref_p)}"
                  f"  max|dp|={dp.max():.2e} mean={dp.mean():.2e}  decision flips={flips}"
                  f"  segments {len(segs)}/{len(ref_segs)}  max boundary diff={bdiff}ms")
            if not ok or bdiff:
                print(f"      funasr: {ref_segs}\n      c++:    {segs}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
