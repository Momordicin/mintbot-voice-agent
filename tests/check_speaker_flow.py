#!/usr/bin/env python3
"""M5 end-to-end: enrollment, persistence and gating through the agent's HTTP API.

  run 1: replay --enroll-wav, POST enroll/start before listening, commit after the
         replay ends (commit before enough speech must give 409); the voiceprint
         file must exist afterwards.
  run 2: a new process with the same --speaker-db replays --test-wav; prints every
         final / speech_other event so the decisions can be checked against
         --expect (comma list of 'owner' / 'other' per utterance, in order).
  run 3: DELETE /v1/speaker removes the file.

    python tests/check_speaker_flow.py --exe build/bin/mintbot-voice-agent.exe --vad models/fsmn-vad.gguf \
        --asr models/sensevoice-small-q8_0.gguf --spk models/campplus-zh-cn.gguf \
        --enroll-wav owner3x.wav --test-wav mixed.wav --expect owner,other,other --db tmp/owner.voiceprint.json
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


class Agent:
    def __init__(self, args, wav):
        self.token = secrets.token_hex(16)
        env = dict(os.environ, MINTBOT_VOICE_TOKEN=self.token)
        self.p = subprocess.Popen([args.exe, "--vad", args.vad, "-m", args.asr, "--spk-model", args.spk,
                                   "--speaker-db", args.db, "--source", "wav:" + wav, "--replay-speed", str(args.speed)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        ready = json.loads(self.p.stdout.readline())
        self.base = f"http://127.0.0.1:{ready['port']}"
        self.events = []
        threading.Thread(target=self._sse, daemon=True).start()
        time.sleep(0.3)

    def call(self, method, path):
        req = urllib.request.Request(self.base + path, method=method, headers={"Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _sse(self):
        req = urllib.request.Request(self.base + "/v1/events", headers={"Authorization": "Bearer " + self.token})
        with urllib.request.urlopen(req) as r:
            ev = {}
            for raw in r:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    ev["type"] = line[7:]
                elif line.startswith("data: "):
                    ev["data"] = json.loads(line[6:])
                elif not line and ev:
                    self.events.append(ev)
                    ev = {}

    def wait_replay_done(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if any(e["type"] == "device" and e["data"]["state"] == "ended" for e in self.events):
                _, st = self.call("GET", "/v1/status")
                if not st["in_speech"] and st["asr_queue"] == 0 and not st["listening"]:
                    time.sleep(0.5)
                    return
            time.sleep(0.2)
        raise TimeoutError("replay did not finish")

    def close(self):
        self.call("POST", "/v1/shutdown")
        self.p.wait(10)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    for a in ("--exe", "--vad", "--asr", "--spk", "--enroll-wav", "--test-wav", "--db"):
        ap.add_argument(a, required=True)
    ap.add_argument("--expect", default="")
    ap.add_argument("--speed", type=float, default=4.0, help="replay speed factor")
    args = ap.parse_args()
    if os.path.exists(args.db):
        os.remove(args.db)
    ok = True

    a = Agent(args, args.enroll_wav)
    code, _ = a.call("POST", "/v1/speaker/enroll/commit")
    print(f"commit with no enrollment -> {code} (expect 409)"); ok &= code == 409
    print("enroll/start ->", a.call("POST", "/v1/speaker/enroll/start")[1])
    a.call("POST", "/v1/listen/start")
    time.sleep(1.5)
    code, body = a.call("POST", "/v1/speaker/enroll/commit")
    print(f"commit after {body.get('error', '')[:60]!r} -> {code} (expect 409)"); ok &= code == 409
    a.wait_replay_done()
    prog = [e["data"] for e in a.events if e["type"] == "enroll_progress"]
    print("enroll_progress:", [(p["utterances"], p["speech_ms"], p["ready"]) for p in prog])
    code, body = a.call("POST", "/v1/speaker/enroll/commit")
    print(f"commit -> {code}", body); ok &= code == 200 and body["enrolled"]
    a.close()
    print("voiceprint file exists:", os.path.exists(args.db)); ok &= os.path.exists(args.db)

    b = Agent(args, args.test_wav)
    _, sp = b.call("GET", "/v1/speaker")
    print("after restart:", {k: sp[k] for k in ("enrolled", "utterances", "speech_ms", "threshold")}); ok &= sp["enrolled"]
    b.call("POST", "/v1/listen/start")
    b.wait_replay_done()
    got = []
    for e in b.events:
        if e["type"] in ("final", "speech_other"):
            d = e["data"]
            got.append("owner" if e["type"] == "final" else "other")
            print(f"  {e['type']:12s} {d['start_ms']:6d}-{d['end_ms']:6d} score={d['speaker_score']:.3f} "
                  f"uncertain={d['speaker_uncertain']} spk_ms={d['speaker_ms']:.0f} {d.get('text', '')[:40]}")
    if args.expect:
        exp = args.expect.split(",")
        print(f"decisions {got} expected {exp} -> {'PASS' if got == exp else 'FAIL'}"); ok &= got == exp
    _, st = b.call("GET", "/v1/status")
    print("status:", {k: st[k] for k in ("utterances", "skipped_other_speaker")})
    code, body = b.call("DELETE", "/v1/speaker")
    print(f"forget -> {code}, enrolled={body['enrolled']}, file exists={os.path.exists(args.db)}")
    ok &= code == 200 and not body["enrolled"] and not os.path.exists(args.db)
    b.close()
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
