#!/usr/bin/env python3
"""The voiceprint file when it cannot be deleted or replaced (Windows: another process
holds it open without delete sharing, as Python's open() does).

  forget: DELETE /v1/speaker fails and nothing changes (still enrolled, gate still
          closed, file still there), so the agent never says "not enrolled" while
          the biometric file survives; once free, DELETE removes the file and a
          leftover owner.voiceprint.json.tmp. Error texts are valid UTF-8 JSON.
  save:   (with --enroll-wav) enroll/commit fails while the file is held: no .tmp is
          left behind, the old file and the old voiceprint stay; once free, commit
          succeeds.

    python tests/check_speaker_forget.py --exe build-p/bin/mintbot-voice-agent.exe --vad models/fsmn-vad.gguf \
        --asr models/sensevoice-small-q8_0.gguf --spk models/campplus-zh-cn.gguf --voiceprint owner.voiceprint.json \
        [--enroll-wav owner.wav]
"""
import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

failures = []
bad_bodies = []


def expect(what, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        failures.append(what)


class Agent:
    def __init__(self, args, db, extra=()):
        self.token = secrets.token_hex(16)
        self.p = subprocess.Popen([os.path.abspath(args.exe), "--vad", args.vad, "-m", args.asr, "--spk-model", args.spk,
                                   "--speaker-db", db, *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, env=dict(os.environ, MINTBOT_VOICE_TOKEN=self.token))
        self.base = f"http://127.0.0.1:{json.loads(self.p.stdout.readline())['port']}"

    def call(self, method, path):
        req = urllib.request.Request(self.base + path, method=method, headers={"Authorization": "Bearer " + self.token})
        try:
            with urllib.request.urlopen(req) as r:
                code, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, raw = e.code, e.read()
        try:
            return code, json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:  # error texts must reach the client as UTF-8 JSON
            bad_bodies.append(raw)
            return code, json.loads(raw.decode("utf-8", errors="replace"))

    def close(self):
        try:
            self.call("POST", "/v1/shutdown")
        except OSError:
            pass
        self.p.wait(10)


def check_forget(args, db):
    print("forget, file held open (delete denied):")
    a = Agent(args, db)
    try:
        _, st = a.call("GET", "/v1/speaker")
        expect("enrolled at start", st["enrolled"])
        with open(db, "rb"):
            code, body = a.call("DELETE", "/v1/speaker")
            expect(f"DELETE fails: {code} {body.get('error', '')}", code == 500)
            expect("the error body is valid UTF-8", not bad_bodies)
            _, st = a.call("GET", "/v1/speaker")
            expect("still enrolled after the failed delete", st["enrolled"])
            expect("file still there", os.path.exists(db))
        print("forget, file free (and a .tmp left by an earlier failed save):")
        with open(db + ".tmp", "w") as f:
            f.write("{}")
        code, st = a.call("DELETE", "/v1/speaker")
        expect(f"DELETE succeeds ({code})", code == 200)
        expect("not enrolled", not st["enrolled"])
        expect("file gone", not os.path.exists(db))
        expect(".tmp gone", not os.path.exists(db + ".tmp"))
        code, _ = a.call("DELETE", "/v1/speaker")
        expect(f"DELETE again, nothing to delete, succeeds ({code})", code == 200)
    finally:
        a.close()


def check_save(args, db, original):
    print("save, file held open (replace denied):")
    shutil.copyfile(original, db)
    before = open(db, "rb").read()
    a = Agent(args, db, ["--source", "wav:" + args.enroll_wav, "--replay-speed", "4"])
    try:
        _, old = a.call("GET", "/v1/speaker")
        a.call("POST", "/v1/speaker/enroll/start")
        a.call("POST", "/v1/listen/start")
        end = time.time() + 120
        while time.time() < end:
            _, st = a.call("GET", "/v1/status")
            if st.get("capture") == "stopped" and st.get("asr_queue", 0) == 0:
                break
            time.sleep(0.3)
        _, sp = a.call("GET", "/v1/speaker")
        expect(f"enrollment collected enough ({sp['enroll_utterances']} utterances)", sp["enroll_ready"])
        with open(db, "rb"):
            code, body = a.call("POST", "/v1/speaker/enroll/commit")
            expect(f"commit fails: {code} {body.get('error', '')}", code != 200)
            expect("the error body is valid UTF-8", not bad_bodies)
            expect("no .tmp left behind", not os.path.exists(db + ".tmp"))
        expect("old file unchanged", open(db, "rb").read() == before)
        _, sp = a.call("GET", "/v1/speaker")
        expect("old voiceprint still in use", sp["enrolled"] and sp["utterances"] == old["utterances"])
        code, sp = a.call("POST", "/v1/speaker/enroll/commit")
        expect(f"commit succeeds once the file is free ({code})", code == 200)
        expect("file replaced, no .tmp", open(db, "rb").read() != before and not os.path.exists(db + ".tmp"))
    finally:
        a.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--exe", "--vad", "--asr", "--spk", "--voiceprint"):
        ap.add_argument(a, required=True)
    ap.add_argument("--enroll-wav")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="forget-")
    db = os.path.join(tmp, "owner.voiceprint.json")
    try:
        shutil.copyfile(args.voiceprint, db)  # the original is never touched
        check_forget(args, db)
        if args.enroll_wav:
            check_save(args, db, args.voiceprint)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
