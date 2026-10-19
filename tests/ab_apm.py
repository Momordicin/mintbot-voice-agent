#!/usr/bin/env python3
"""M4: APM checks.

1. Chunking invariance: apm-cli output must be bit-identical for any chunk size.
2. A/B: pipeline-cli with several APM settings on wavs that have clip ground
   truth (<wav>.json: [{"clip", "start", "end"}]) and whole-clip reference
   transcripts (--refs). Reports CER, utterance count and noise-only segments.
3. Noise-only wavs (--noise): number of utterances (false triggers) per setting.

    python tests/ab_apm.py --bin build/bin --vad models/fsmn-vad.gguf --asr models/sensevoice-small-q8_0.gguf \
        --refs refs.json --noise noise.wav a.wav b.wav
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

CONFIGS = {
    "off": [],
    "hpf only": ["--apm", "--ns", "0", "--no-agc"],
    "ns1 + agc": ["--apm", "--ns", "1"],
    "ns2 + agc (default)": ["--apm", "--ns", "2"],
    "ns3 + agc": ["--apm", "--ns", "3"],
    "ns2, no agc": ["--apm", "--ns", "2", "--no-agc"],
}


def cer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(hyp, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1] / max(1, len(ref))


def norm(s):
    return re.sub(r"[\W_]+", "", s.lower())


def run_pipeline(args, wav, flags):
    r = subprocess.run([os.path.join(args.bin, "pipeline-cli"), "--vad", args.vad, "-m", args.asr, "-a", wav] + flags,
                       capture_output=True, check=True)
    return [json.loads(l) for l in r.stdout.decode("utf-8").splitlines() if l.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--noise", action="append", default=[])
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    refs = json.load(open(args.refs, encoding="utf-8"))
    failed = False

    with tempfile.TemporaryDirectory() as tmp:
        base = None
        for ms in (10, 7, 23, 100, 1000):
            out = os.path.join(tmp, f"{ms}.wav")
            subprocess.run([os.path.join(args.bin, "apm-cli"), "-a", args.wavs[0], "-o", out, "--chunk-ms", str(ms)],
                           capture_output=True, check=True)
            x, _ = sf.read(out, dtype="int16")
            same = base is None or (len(x) == len(base) and np.array_equal(x, base))
            base = x if base is None else base
            failed |= not same
            print(f"{'PASS' if same else 'FAIL'}  apm chunk={ms:4d}ms samples={len(x)} {'identical' if same else 'DIFFERS'}")

    print()
    for wav in args.wavs:
        meta = json.load(open(os.path.splitext(wav)[0] + ".json"))
        for name, flags in CONFIGS.items():
            utts = run_pipeline(args, wav, flags)
            per = {i: [] for i in range(len(meta))}
            extra = 0
            for u in utts:
                ov = [max(0, min(u["end_ms"] / 1000, m["end"]) - max(u["start_ms"] / 1000, m["start"])) for m in meta]
                if max(ov) > 0:
                    per[int(np.argmax(ov))].append(u["text"])
                else:
                    extra += 1
            e = n = 0
            for i, m in enumerate(meta):
                rr = norm(refs[m["clip"]])
                e += cer(rr, norm("".join(per[i]))) * len(rr)
                n += len(rr)
            print(f"{os.path.basename(wav):18s} {name:20s} CER={e / n * 100:5.1f}%  utterances={len(utts):2d}  noise-only={extra}")
        print()
    for wav in args.noise:
        for name, flags in CONFIGS.items():
            utts = run_pipeline(args, wav, flags)
            print(f"{os.path.basename(wav):18s} {name:20s} false utterances={len(utts)}  {[u['text'][:12] for u in utts][:4]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
