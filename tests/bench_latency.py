#!/usr/bin/env python3
"""End-to-end latency through the agent: replays WAV files in real time and
breaks every utterance's delay down from the agent's own event fields.

  detect_lag  audio between the end of speech and the VAD closing the segment
  queue       wait for the ASR thread
  speaker     CAM++ scoring (speaker_ms)
  asr         SenseVoice (asr_ms)
  total       detect_lag + latency_ms: end of speech -> event published

Capture-device buffering (one 10 ms period) and SSE delivery are not included.

    python tests/bench_latency.py --exe build/bin/mintbot-voice-agent.exe --vad models/fsmn-vad.gguf \
        --asr models/sensevoice-small-q8_0.gguf --spk models/campplus-zh-cn.gguf \
        --voiceprint rec/owner.voiceprint.json --out bench-base.json rec/owner-1.wav rec/other-1.wav

--compare bench-base.json reports utterances whose boundaries, decision,
speaker score or text differ from an earlier run, so an optimisation can be checked for
identical output. --agent-arg passes extra flags (repeatable).
"""
import argparse
import json
import os
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


class Agent:
    def __init__(self, cmd):
        self.token = secrets.token_hex(16)
        env = dict(os.environ, MINTBOT_VOICE_TOKEN=self.token)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        line = self.p.stdout.readline()
        if not line:
            sys.exit("agent did not start: " + " ".join(cmd))
        self.base = f"http://127.0.0.1:{json.loads(line)['port']}"
        self.events = []
        self.cv = threading.Condition()
        self.connected = threading.Event()
        threading.Thread(target=self._sse, daemon=True).start()
        if not self.connected.wait(5):
            sys.exit("no SSE connection")

    def call(self, method, path):
        req = urllib.request.Request(self.base + path, method=method, headers={"Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _sse(self):
        req = urllib.request.Request(self.base + "/v1/events", headers={"Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(req) as r:
                self.connected.set()
                ev = {}
                for raw in r:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("event: "):
                        ev["type"] = line[7:]
                    elif line.startswith("data: "):
                        ev["data"] = json.loads(line[6:])
                    elif not line and ev:
                        with self.cv:
                            self.events.append(ev)
                            self.cv.notify_all()
                        ev = {}
        except OSError:
            pass

    def wait_for(self, pred, timeout):
        end = time.time() + timeout
        with self.cv:
            while not any(pred(e) for e in self.events):
                left = end - time.time()
                if left <= 0:
                    return False
                self.cv.wait(left)
        return True

    def close(self):
        try:
            self.call("POST", "/v1/shutdown")
        except OSError:
            pass
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def wav_seconds(path):
    import soundfile as sf
    return sf.info(path).duration


def cpu_seconds(pid):
    try:
        import psutil
    except ImportError:
        return None
    t = psutil.Process(pid).cpu_times()
    return t.user + t.system


def run_file(args, wav, db, cpu, busy):
    cmd = [args.exe, "--vad", args.vad, "-m", args.asr, "--source", "wav:" + wav, "--replay-speed", str(args.speed)]
    if args.spk:
        cmd += ["--spk-model", args.spk, "--speaker-db", db]
    cmd += args.agent_arg
    a = Agent(cmd)
    try:
        cpu0 = cpu_seconds(a.p.pid)
        code, body = a.call("POST", "/v1/listen/start")
        if code != 200:
            sys.exit(f"listen/start failed: {code} {body}")
        dur = wav_seconds(wav) / args.speed
        if not a.wait_for(lambda e: e["type"] == "device" and e["data"].get("state") == "ended", dur + 30):
            sys.exit("replay did not end: " + wav)
        deadline = time.time() + 60  # let the ASR thread drain
        while time.time() < deadline:
            _, st = a.call("GET", "/v1/status")
            if st.get("asr_queue", 0) == 0 and st.get("capture") == "stopped":
                time.sleep(0.5)
                _, st2 = a.call("GET", "/v1/status")
                if st2.get("asr_queue", 0) == 0:
                    break
            time.sleep(0.2)
        if cpu0 is not None:  # listening -> drained, model loading excluded
            cpu[os.path.basename(wav)] = cpu_seconds(a.p.pid) - cpu0
            print(f"  agent CPU {cpu[os.path.basename(wav)]:.1f} s for {dur:.0f} s of audio", flush=True)
        _, st = a.call("GET", "/v1/status")
        if "busy_ms" in st:
            busy[os.path.basename(wav)] = st["busy_ms"]
            print("  ASR-thread compute ms: " + ", ".join(f"{k} {v}" for k, v in st["busy_ms"].items()), flush=True)
        if "early_runs" in st:
            print(f"  early recognition runs {st['early_runs']}, used {st['early_hits']}, "
                  f"stopped early {st.get('early_aborted', 0)}", flush=True)
    finally:
        a.close()
    out = []
    for e in a.events:
        if e["type"] not in ("final", "speech_other"):
            continue
        d = e["data"]
        out.append({
            "file": os.path.basename(wav),
            "kind": e["type"],
            "start_ms": d["start_ms"], "end_ms": d["end_ms"],
            "dur_ms": d["end_ms"] - d["start_ms"],
            "detect_lag_ms": d.get("detect_lag_ms"),
            "queue_ms": d.get("queue_ms"),
            "speaker_ms": d.get("speaker_ms", 0.0),
            "asr_ms": d.get("asr_ms", 0.0),
            "latency_ms": d["latency_ms"],
            "total_ms": d.get("total_ms"),
            "score": d.get("speaker_score"),
            "speaker_windows": d.get("speaker_windows"),
            "speaker_reused": d.get("speaker_reused"),
            "early": d.get("early"),
            "text": d.get("text"),
        })
    return out


def stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "-"
    p90 = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
    return f"{statistics.median(vals):7.0f} {p90:7.0f} {vals[-1]:7.0f}"


def report(rows):
    cols = ["detect_lag_ms", "queue_ms", "speaker_ms", "asr_ms", "latency_ms", "total_ms"]
    for kind in ("final", "speech_other"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        print(f"\n{kind}: {len(rs)} utterances, duration median {statistics.median(r['dur_ms'] for r in rs):.0f} ms")
        print(f"  {'':14s} {'median':>7s} {'p90':>7s} {'max':>7s}")
        for c in cols:
            print(f"  {c:14s} {stats([r[c] for r in rs])}")
        if any(r["early"] is not None for r in rs):
            print(f"  early result used for {sum(1 for r in rs if r['early'])} of {len(rs)}")
        w = sum(r["speaker_windows"] or 0 for r in rs)
        if w:
            print(f"  speaker windows {w}, computed during speech {sum(r['speaker_reused'] or 0 for r in rs)}")


def compare(rows, base_path):
    base = json.load(open(base_path, encoding="utf-8"))["utterances"]
    key = lambda r: (r["file"], r["start_ms"], r["end_ms"])
    b = {key(r): r for r in base}
    n = {key(r): r for r in rows}
    diffs = 0
    for k in sorted(set(b) | set(n)):
        x, y = b.get(k), n.get(k)
        if x is None or y is None:
            diffs += 1
            print(f"  boundary {k}: only in {'new' if x is None else 'base'}")
        elif x["kind"] != y["kind"] or x["text"] != y["text"] or x["score"] != y["score"]:
            diffs += 1
            print(f"  {k}: {x['kind']} {x['score']} {x['text']!r} -> {y['kind']} {y['score']} {y['text']!r}")
    print(f"\ncompare with {base_path}: {len(n)} utterances, {diffs} differ")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--spk")
    ap.add_argument("--voiceprint", help="owner voiceprint; copied to a temp dir, the original is never written")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--agent-arg", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    args.exe = os.path.abspath(args.exe)

    tmp = tempfile.mkdtemp(prefix="bench-latency-")
    db = os.path.join(tmp, "owner.voiceprint.json")
    if args.voiceprint:
        shutil.copyfile(args.voiceprint, db)
    rows = []
    cpu = {}
    busy = {}
    try:
        for w in args.wavs:
            print(f"replaying {w} ...", flush=True)
            rows += run_file(args, w, db, cpu, busy)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report(rows)
    if busy:
        tot = {}
        for b in busy.values():
            for k, v in b.items():
                tot[k] = tot.get(k, 0) + v
        print("ASR-thread compute total ms: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
    if cpu:
        print(f"\nagent CPU total {sum(cpu.values()):.1f} s (user + system, all threads)")
    if args.out:
        json.dump({"agent_args": args.agent_arg, "speed": args.speed, "cpu_s": cpu, "busy_ms": busy, "utterances": rows},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if args.compare:
        sys.exit(1 if compare(rows, args.compare) else 0)


if __name__ == "__main__":
    main()
