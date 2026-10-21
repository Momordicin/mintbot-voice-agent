#!/usr/bin/env python3
"""Checks that FsmnVadStream gives the same output however the audio is chunked.

    python tests/check_vad_chunking.py --cli build/bin/vad-stream-cli.exe \
        --model models/fsmn-vad.gguf a.wav b.wav ...
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

CHUNKS_MS = [10, 20, 100, 200, 1000]


def run(cli, model, wav, chunk_ms, end_sil, probs_path):
    out = subprocess.run(
        [cli, "-m", model, "-a", wav, "--chunk-ms", str(chunk_ms), "--end-sil", str(end_sil),
         "--probs-out", probs_path],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["segments"], np.fromfile(probs_path, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--end-sil", type=int, default=800)
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for wav in args.wavs:
            ref_segs, ref_p = run(args.cli, args.model, wav, 10**8, args.end_sil, os.path.join(tmp, "ref.f32"))
            for ms in CHUNKS_MS:
                segs, p = run(args.cli, args.model, wav, ms, args.end_sil, os.path.join(tmp, f"{ms}.f32"))
                same_len = len(p) == len(ref_p)
                diff = float(np.abs(p - ref_p).max()) if same_len and len(p) else float("nan")
                ok = same_len and diff <= 1e-5 and segs == ref_segs
                failed |= not ok
                print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(wav):16s} chunk={ms:5d}ms  frames={len(p)}/{len(ref_p)}"
                      f"  max|dp|={diff:.2e}  segments {'equal' if segs == ref_segs else 'DIFFER'} ({len(segs)})")
                if segs != ref_segs:
                    print(f"      whole:   {ref_segs}\n      chunked: {segs}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
