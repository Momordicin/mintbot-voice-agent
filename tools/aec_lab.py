"""Echo-cancellation experiments (R2-0): test signals, delay/drift analysis and
end-to-end evaluation of aec-rec recordings through aec-offline and the pipeline.

  python tools/aec_lab.py signal --out test.wav [--seconds 120] [--period 5]
      chirps (for delay tracking) and noise bursts, 16 kHz mono PCM16
  python tools/aec_lab.py synth --ref ref.wav --out DIR [--delay-ms 120] [--gain 0.3]
                                [--near speech.wav] [--drift-ppm 0] [--noise-db -60]
      a synthetic recording: mic = ref through a small decaying room response,
      delayed, plus optional near-end speech; writes mic.wav, loopback.wav, near.wav
  python tools/aec_lab.py delay DIR [--ref loopback|played]
      reference-to-mic delay over time (GCC-PHAT), its drift in ppm, and each
      stream's sample rate measured against the callback clock (timing.csv)
  python tools/aec_lab.py eval DIR [--voiceprint owner.voiceprint.json] [--truth text.txt]
      raw (the agent today) vs AEC3 vs AEC3 linear output: utterances, text, owner
      score per utterance, CER against --truth (of all text, and of the text that
      passes the owner gate), ERLE where only the reference plays (--echo-only)

Only numpy is needed. Binaries are taken from build/bin, models from build-models/out.
"""

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "bin"
MODELS = ROOT / "build-models" / "out"
SR = 16000


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{p}: need 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write_wav(p, x):
    y = np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


def db(x):
    return 10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)


# ---- signal ----------------------------------------------------------------

def chirp(seconds, f0=100.0, f1=7000.0):
    t = np.arange(int(seconds * SR)) / SR
    k = np.log(f1 / f0) / seconds
    x = np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1) / k)
    ramp = int(0.01 * SR)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return x


def cmd_signal(a):
    rng = np.random.default_rng(1)
    period = int(a.period * SR)
    out = np.zeros(int(a.seconds * SR))
    c = chirp(0.4) * 10 ** (-12 / 20)
    for start in range(SR, len(out) - period, period):  # 1 s of silence first
        out[start:start + len(c)] += c
        n0 = start + int(1.5 * SR)
        noise = rng.standard_normal(SR) * 10 ** (-24 / 20)
        out[n0:n0 + SR] += noise
    write_wav(a.out, out)
    print(f"{a.out}: {a.seconds:.0f} s, a chirp every {a.period:g} s and a 1 s noise burst 1.5 s after each")


# ---- synth -----------------------------------------------------------------

def cmd_synth(a):
    rng = np.random.default_rng(2)
    ref = read_wav(a.ref).astype(np.float64)
    n = len(ref)
    # room: direct path plus exponentially decaying reflections (RT60 about 0.25 s)
    ir = rng.standard_normal(int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.25 * SR / 6.9)) * 0.15
    ir[0] = 1.0
    echo = np.convolve(ref, ir)[:n] * a.gain
    if a.drift_ppm:
        # the microphone clock runs drift_ppm faster: it sees the reference stretched in time
        t = np.arange(n) / (1 + a.drift_ppm * 1e-6)
        echo = np.interp(t, np.arange(n), echo)
    d = int(a.delay_ms * SR / 1000)
    echo = np.concatenate([np.zeros(d), echo])[:n]
    near = np.zeros(n)
    if a.near:
        s = read_wav(a.near).astype(np.float64)
        off = int(a.near_at * SR)
        m = min(len(s), n - off)
        near[off:off + m] = s[:m]
    mic = echo + near + rng.standard_normal(n) * 10 ** (a.noise_db / 20)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "mic.wav", mic)
    write_wav(out / "loopback.wav", ref)
    write_wav(out / "near.wav", near)
    (out / "synth.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: echo {db(echo):.1f} dBFS, near {db(near) if a.near else -120:.1f} dBFS, delay {a.delay_ms} ms")


# ---- delay -----------------------------------------------------------------

def gcc_phat(x, y, max_lag):
    """Lag (samples) by which y follows x, and the peak's prominence."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n)
    R = Y * np.conj(X)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    k = int(np.argmax(np.abs(cc)))
    peak = abs(cc[k])
    return k - max_lag, peak / (np.median(np.abs(cc)) + 1e-12)


def stream_rates(d):
    p = Path(d) / "timing.csv"
    if not p.exists():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines()[1:]:
        s, t, f = line.split(",")
        rows.setdefault(s, []).append((int(t), int(f)))
    out = {}
    for s, r in rows.items():
        t = np.array([x[0] for x in r], dtype=np.float64) / 1e6
        cum = np.cumsum([x[1] for x in r]).astype(np.float64)
        if len(t) < 100:
            continue
        # skip the first second (start-up buffering), fit samples against time
        k = t > t[0] + 1
        slope = np.polyfit(t[k], cum[k], 1)[0]
        gaps = np.diff(t)
        out[s] = {"callbacks": len(r), "rate_hz": round(slope, 2), "ppm": round((slope / SR - 1) * 1e6, 1),
                  "max_gap_ms": round(float(gaps.max()) * 1000, 1)}
    return out


def delay_track(ref, mic, win_s=2.0, hop_s=1.0, max_lag_ms=1000, min_prominence=8.0):
    w, h, L = int(win_s * SR), int(hop_s * SR), int(max_lag_ms * SR / 1000)
    pts = []
    for s in range(0, min(len(ref), len(mic)) - w - L, h):
        r = ref[s:s + w]
        if db(r) < -50:
            continue
        lag, prom = gcc_phat(r, mic[s:s + w + L], L)
        if prom >= min_prominence and lag >= 0:
            pts.append(((s + w / 2) / SR, lag * 1000 / SR, prom))
    return pts


def cmd_delay(a):
    d = Path(a.dir)
    ref = read_wav(d / f"{a.ref}.wav")
    mic = read_wav(d / "mic.wav")
    pts = delay_track(ref, mic)
    res = {"streams": stream_rates(d), "reference": a.ref, "windows": len(pts)}
    if pts:
        t = np.array([p[0] for p in pts])
        ms = np.array([p[1] for p in pts])
        res["delay_ms"] = {"median": round(float(np.median(ms)), 1), "min": round(float(ms.min()), 1),
                           "max": round(float(ms.max()), 1)}
        if t[-1] - t[0] > 20:
            slope = np.polyfit(t, ms, 1)[0]  # ms per s
            res["drift_ppm"] = round(slope * 1000, 1)
            res["delay_ms"]["fit_start"] = round(float(np.polyval(np.polyfit(t, ms, 1), t[0])), 1)
            res["delay_ms"]["fit_end"] = round(float(np.polyval(np.polyfit(t, ms, 1), t[-1])), 1)
    if a.ref == "loopback" and (d / "played.wav").exists():
        p = read_wav(d / "played.wav")
        pp = delay_track(p, ref, max_lag_ms=500)
        if pp:
            res["played_to_loopback_ms"] = round(float(np.median([x[1] for x in pp])), 1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("t_s,delay_ms,prominence\n")
            for p in pts:
                f.write(f"{p[0]:.2f},{p[1]:.2f},{p[2]:.1f}\n")


# ---- eval ------------------------------------------------------------------

def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd))} failed:\n{r.stderr}")
    return r


def jsonl(s):
    return [json.loads(l) for l in s.splitlines() if l.strip().startswith("{")]


def cer(ref, hyp):
    keep = lambda s: [c for c in s.lower() if c.isalnum()]
    r, h = keep(ref), keep(hyp)
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def cmd_eval(a):
    d = Path(a.dir)
    out = Path(a.work) if a.work else d / "eval"
    out.mkdir(parents=True, exist_ok=True)
    mic = d / "mic.wav"
    ref = d / f"{a.ref}.wav"
    ext = ".exe" if sys.platform == "win32" else ""
    center = None
    if a.voiceprint:
        center = np.array(json.loads(Path(a.voiceprint).read_text(encoding="utf-8"))["center"], dtype=np.float64)
        center /= np.linalg.norm(center)

    variants = {"raw": (mic, True)}  # (input to the pipeline, run the agent's APM in the pipeline)
    full, lin = out / "aec.wav", out / "aec_linear.wav"
    r = run([BIN / f"aec-offline{ext}", "--mic", mic, "--ref", ref, "-o", full, "--linear", lin,
             "--stats", out / "aec_stats.jsonl"] + (["--delay-hint", a.delay_hint] if a.delay_hint is not None else []))
    cpu = json.loads(r.stderr.strip().splitlines()[-1])
    variants["aec"] = (full, False)
    variants["aec_linear"] = (lin, True)

    # ERLE where only the reference plays: mic louder than -50 dBFS, the AEC output
    # judged on the same 100 ms blocks; needs --echo-only (start,end s pairs) to be meaningful
    report = {"aec_cpu": cpu, "variants": {}}
    if a.echo_only:
        m, f, l = read_wav(mic), read_wav(full), read_wav(lin)
        spans = [tuple(map(float, s.split(":"))) for s in a.echo_only.split(",")]
        def pick(x):
            return np.concatenate([x[int(s * SR):int(e * SR)] for s, e in spans])
        report["erle_db"] = {"aec": round(db(pick(m)) - db(pick(f)), 1), "aec_linear": round(db(pick(m)) - db(pick(l)), 1)}
    stats = jsonl((out / "aec_stats.jsonl").read_text(encoding="utf-8"))
    delays = [s["delay_ms"] for s in stats if s.get("delay_ms") is not None]
    if delays:
        report["aec3_delay_ms"] = {"first": delays[0], "median": float(np.median(delays)), "last": delays[-1]}

    truth = Path(a.truth).read_text(encoding="utf-8") if a.truth else None
    for name, (wav, apm) in variants.items():
        cmd = [BIN / f"pipeline-cli{ext}", "--vad", MODELS / "fsmn-vad.gguf", "-m", MODELS / "sensevoice-small-q8_0.gguf",
               "-a", wav] + (["--apm"] if apm else [])
        utts = jsonl(run(cmd).stdout)
        if center is not None:
            spk = [BIN / f"spk-cli{ext}", "-m", MODELS / "campplus-zh-cn.gguf", "-a", wav, "--vad", MODELS / "fsmn-vad.gguf"]
            embs = jsonl(run(spk + (["--apm"] if apm else [])).stdout)
            for u in utts:
                # the spk-cli utterance overlapping this one most
                best = max(embs, key=lambda e: min(e["end_ms"], u["end_ms"]) - max(e["start_ms"], u["start_ms"]), default=None)
                if best and min(best["end_ms"], u["end_ms"]) > max(best["start_ms"], u["start_ms"]):
                    e = np.array(best["embedding"])
                    u["owner_score"] = round(float(e @ center / np.linalg.norm(e)), 3)
        text = "".join(u["text"] for u in utts)
        v = {"utterances": len(utts), "speech_s": round(sum(u["end_ms"] - u["start_ms"] for u in utts) / 1000, 1),
             "text": text, "items": utts}
        if center is not None:
            # what the agent would deliver: utterances that pass the owner gate
            v["owner_text"] = "".join(u["text"] for u in utts if u.get("owner_score", -1) >= a.threshold)
        if truth is not None:
            v["cer"] = round(cer(truth, text), 3)
            if center is not None:
                v["owner_cer"] = round(cer(truth, v["owner_text"]), 3)
        report["variants"][name] = v
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, v in report["variants"].items():
        print(f"== {name}: {v['utterances']} utterances, {v['speech_s']} s" + (f", CER {v['cer']}" if "cer" in v else "") +
              (f", CER of owner-gated text {v['owner_cer']}" if "owner_cer" in v else ""))
        for u in v["items"]:
            sc = f" owner {u['owner_score']:.3f}" if "owner_score" in u else ""
            print(f"  {u['start_ms'] / 1000:7.2f}-{u['end_ms'] / 1000:7.2f}{sc}  {u['text']}")
    extra = {k: report[k] for k in ("aec_cpu", "erle_db", "aec3_delay_ms") if k in report}
    print(json.dumps(extra, ensure_ascii=False))
    print(f"full report: {out / 'report.json'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("signal")
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--period", type=float, default=5)
    p = sub.add_parser("synth")
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delay-ms", type=int, default=120)
    p.add_argument("--gain", type=float, default=0.3)
    p.add_argument("--near")
    p.add_argument("--near-at", type=float, default=10.0, help="where the near-end speech starts, s")
    p.add_argument("--drift-ppm", type=float, default=0)
    p.add_argument("--noise-db", type=float, default=-60)
    p = sub.add_parser("delay")
    p.add_argument("dir")
    p.add_argument("--ref", default="loopback", choices=["loopback", "played"])
    p.add_argument("--csv")
    p = sub.add_parser("eval")
    p.add_argument("dir")
    p.add_argument("--ref", default="loopback", choices=["loopback", "played"])
    p.add_argument("--voiceprint")
    p.add_argument("--threshold", type=float, default=0.5, help="owner gate, as the agent's speakerThreshold")
    p.add_argument("--truth", help="what the owner said, for CER")
    p.add_argument("--echo-only", help="spans with only the reference playing, s: 2:9,40:45")
    p.add_argument("--delay-hint")
    p.add_argument("--work", help="output directory (default DIR/eval)")
    a = ap.parse_args()
    {"signal": cmd_signal, "synth": cmd_synth, "delay": cmd_delay, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
