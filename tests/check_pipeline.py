#!/usr/bin/env python3
"""Checks pipeline-cli against independent references: VAD segments, a reference SenseVoice CLI and, optionally, CER.

    python tests/check_pipeline.py --pipeline build/bin/pipeline-cli.exe --vad-cli build/bin/vad-stream-cli.exe \
        --vad models/fsmn-vad.gguf --asr models/sensevoice-small-q8_0.gguf \
        --ref-cli <llama-funasr-sensevoice.exe> a.wav ...
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


def cer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(hyp, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1] / max(1, len(ref))


def norm(s):
    return re.sub(r"[\W_]+", "", s.lower())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--vad-cli", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--ref-cli", required=True)
    ap.add_argument("--frame-ms", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--meta-suffix", default=".json", help="clip ground truth next to each wav (optional)")
    ap.add_argument("--refs", help="json: clip name -> whole-clip transcript (optional)")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    refs = json.load(open(args.refs, encoding="utf-8")) if args.refs else None

    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for wav in args.wavs:
            r = subprocess.run([args.pipeline, "--vad", args.vad, "-m", args.asr, "-a", wav, "--frame-ms", str(args.frame_ms), "--threads", str(args.threads)],
                               capture_output=True, check=True)
            utts = [json.loads(l) for l in r.stdout.decode("utf-8").splitlines() if l.strip()]
            stats = json.loads(r.stderr.decode("utf-8").strip().splitlines()[-1])
            vad = json.loads(subprocess.run([args.vad_cli, "-m", args.vad, "-a", wav], capture_output=True, text=True,
                                            check=True).stdout)["segments"]
            seg_ok = [[u["start_ms"], u["end_ms"]] for u in utts] == vad

            y, sr = sf.read(wav, dtype="int16")
            mismatches = []
            for u in utts:
                seg = os.path.join(tmp, "seg.wav")
                sf.write(seg, y[u["start_ms"] * 16:u["end_ms"] * 16] / 32768.0, 16000, subtype="FLOAT")
                ref = subprocess.run([args.ref_cli, "-m", args.asr, "-a", seg], capture_output=True).stdout.decode("utf-8").strip()
                if ref != u["text"]:
                    mismatches.append((u["start_ms"], ref, u["text"]))
            ok = seg_ok and not mismatches and stats["anomalies"] == 0
            failed |= not ok
            line = (f"{'PASS' if ok else 'FAIL'}  {os.path.basename(wav):16s} utterances={len(utts)}"
                    f"  segments {'= vad' if seg_ok else '!= vad'}  text mismatches={len(mismatches)}"
                    f"  audio={stats['audio_s']}s wall={stats['wall_s']}s asr={stats['asr_s']}s"
                    f"  max_buffer={stats['max_buffered_samples'] / 16000:.1f}s")
            meta_path = os.path.splitext(wav)[0] + args.meta_suffix
            if refs and os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                per = {i: [] for i in range(len(meta))}
                for u in utts:
                    ov = [max(0, min(u["end_ms"] / 1000, m["end"]) - max(u["start_ms"] / 1000, m["start"])) for m in meta]
                    if max(ov) > 0:
                        per[int(np.argmax(ov))].append(u["text"])
                e = n = 0
                for i, m in enumerate(meta):
                    rr = norm(refs[m["clip"]])
                    e += cer(rr, norm("".join(per[i]))) * len(rr)
                    n += len(rr)
                line += f"  CER={e / n * 100:.1f}%"
            print(line)
            for s, ref, got in mismatches:
                print(f"      @{s}ms ref: {ref}\n             got: {got}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
