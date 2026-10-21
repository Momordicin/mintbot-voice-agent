// Live test of the owner-speaker gate on the real microphone: enrollment and labelled
// sessions, each recorded with its events so it doubles as calibration data.
//
//   node --experimental-strip-types examples/ts/live-speaker-test.ts --exe build/bin/mintbot-voice-agent.exe \
//        --vad models/fsmn-vad.gguf --model models/sensevoice-small-q8_0.gguf --spk models/campplus-zh-cn.gguf \
//        --db rec/owner.voiceprint.json --out rec [--threshold 0.5] [--device N] [--enroll-utterances 5] \
//        [--cross-device N]
import { existsSync, mkdirSync, readdirSync, appendFileSync } from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline/promises";
import { startVoiceAgent, type VoiceAgent } from "./voice-agent.ts";

const arg = (name: string, def?: string) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : def;
};
const exe = arg("--exe")!, vad = arg("--vad")!, model = arg("--model")!, spk = arg("--spk")!;
const db = arg("--db")!, out = arg("--out")!;
const threshold = arg("--threshold", "0.5")!;
const device = arg("--device");
const crossDevice = arg("--cross-device", device);
const enrollUtterances = Number(arg("--enroll-utterances", "5"));
const source = arg("--source");
const replaySpeed = arg("--replay-speed", "1")!;
const sourceArgs = source ? ["--source", source, "--replay-speed", replaySpeed] : [];
if (!exe || !vad || !model || !spk || !db || !out) {
  console.error("usage: live-speaker-test.ts --exe .. --vad .. --model .. --spk .. --db <voiceprint.json> --out <dir>");
  process.exit(2);
}
mkdirSync(out, { recursive: true });
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function nextFile(kind: string) {
  const used = readdirSync(out).filter((f) => f.startsWith(kind + "-") && f.endsWith(".wav")).length;
  return path.join(out, `${kind}-${used + 1}`);
}

async function session(kind: string, onEvent: (ev: any, agent: VoiceAgent) => Promise<boolean | void>, extra: string[] = []) {
  const base = nextFile(kind);
  const agent = await startVoiceAgent({
    exe, vad, model,
    args: ["--spk-model", spk, "--speaker-db", db, "--speaker-threshold", threshold, "--record", base + ".wav",
           ...sourceArgs, ...extra],
  });
  const abort = new AbortController();
  let finished = false;
  const done = (async () => {
    try {
      for await (const ev of agent.events(abort.signal)) {
        appendFileSync(base + ".events.jsonl", JSON.stringify({ type: ev.type, ...ev.data }) + "\n");
        if ((await onEvent(ev, agent)) === true) { finished = true; break; }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") throw e;
    }
  })();
  return { agent, base, abort, done, get finished() { return finished; } };
}

function line(ev: any) {
  const d = ev.data;
  const s = d.speaker_enrolled && typeof d.speaker_score === "number" ? d.speaker_score.toFixed(2) : "  - ";
  const unc = d.speaker_uncertain ? "?" : " ";
  const len = ((d.end_ms - d.start_ms) / 1000).toFixed(1).padStart(4) + "s";
  if (ev.type === "speech_other") return `  [其他人 ${s}${unc}] ${len}  (未识别)`;
  if (d.enrolling) return `  [录入中 ${s}${unc}] ${len}  ${d.text}`;
  if (!d.speaker_enrolled) return `  [未录入 ${s}${unc}] ${len}  ${d.text}`;
  return `  [${d.is_owner ? "主人  " : "其他人"} ${s}${unc}] ${len}  ${d.text}`;
}

async function enroll() {
  console.log(`\n== 录入声纹：请用平时的语气说 ${enrollUtterances} 句以上的话，每句之间停顿一下（总共至少 10 秒）`);
  const s = await session("enroll", async (ev, agent) => {
    if (ev.type === "final") console.log(line(ev));
    if (ev.type === "enroll_progress") {
      const p = ev.data;
      console.log(`  进度：${p.utterances} 句，${(p.speech_ms / 1000).toFixed(1)} 秒${p.ready ? "（已满足最低要求）" : ""}`);
      if (p.ready && p.utterances >= enrollUtterances) {
        const r = await agent.call("POST", "/v1/speaker/enroll/commit");
        console.log(`  已提交：${r.utterances} 句，${(r.speech_ms / 1000).toFixed(1)} 秒，声纹已写入 ${db}`);
        return true;
      }
    }
  });
  await s.agent.call("POST", "/v1/speaker/enroll/start");
  await s.agent.call("POST", "/v1/listen/start" + (device ? `?device=${device}` : ""));
  await s.done;
  s.abort.abort();
  await s.agent.shutdown();
}

function crossLine(ev: any) {
  const d = ev.data;
  const label = d.speaker_transcribe ? "主人  " : "其他人";
  const len = ((d.end_ms - d.start_ms) / 1000).toFixed(2) + "s";
  return `  [${label}] 分数 ${d.speaker_score.toFixed(3)}  阈值 ${d.speaker_threshold.toFixed(3)}  长度 ${len}  ${d.text}`;
}

async function testSession(kind: string) {
  const hint: Record<string, string> = {
    owner: "只有你说话：正常聊天、短指令（1–2 秒）、长句都来一些",
    other: "你不要说话：放游戏 / 视频 / 电视，或让别人说话",
    free: "随便测试（这一场不用于校准）",
    cross: "非原生设备上随便测试：每句都转写，并显示判定、分数、阈值和长度",
  };
  const cross = kind === "cross";
  console.log(`\n== ${kind} 场次：${hint[kind]}。按回车结束这一场。`);
  const stats = { owner: 0, other: 0, scores: [] as number[] };
  const s = await session(kind, async (ev) => {
    if (cross && ev.type === "final") {
      console.log(crossLine(ev));
      stats.scores.push(ev.data.speaker_score);
      ev.data.speaker_transcribe ? stats.owner++ : stats.other++;
    } else if (ev.type === "final" || ev.type === "speech_other") {
      console.log(line(ev));
      if (typeof ev.data.speaker_score === "number") stats.scores.push(ev.data.speaker_score);
      ev.type === "final" && ev.data.is_owner ? stats.owner++ : stats.other++;
    } else if (ev.type === "device" && ev.data.state !== "started") {
      console.log(`  (设备：${ev.data.state})`);
    } else if (ev.type === "error") {
      console.log(`  (错误：${ev.data.message})`);
    }
  }, cross ? ["--speaker-observe"] : []);
  const dev = cross ? crossDevice : device;
  await s.agent.call("POST", "/v1/listen/start" + (dev ? `?device=${dev}` : ""));
  if (cross) {
    await new Promise((r) => setTimeout(r, 500));
    const sp = (await s.agent.call("GET", "/v1/status")).speaker;
    if (!sp?.enrolled) console.log("  (注意：还没有声纹)");
    else if (!sp.enrolled_device) console.log("  (注意：这份声纹没有记录录入设备，判定按原生设备处理；请先 r 重新录入)");
    else if (sp.same_device) console.log(`  (注意：当前就是录入设备 ${sp.device}，不是非原生设备；用 --cross-device 选别的麦克风)`);
    else console.log(`  录入设备：${sp.enrolled_device}；当前设备：${sp.device}`);
  }
  await rl.question("");
  await s.agent.call("POST", "/v1/listen/stop");
  await new Promise((r) => setTimeout(r, 1500));
  s.abort.abort();
  await s.agent.shutdown();
  const sc = stats.scores.sort((a, b) => a - b);
  const pct = (p: number) => (sc.length ? sc[Math.min(sc.length - 1, Math.floor(p * sc.length))].toFixed(2) : "-");
  console.log(`  本场：判为主人 ${stats.owner} 句，判为其他人 ${stats.other} 句；分数 最低 ${pct(0)} 中位 ${pct(0.5)} 最高 ${pct(0.999)}`);
  console.log(`  录音：${s.base}.wav`);
}

async function main() {
  if (!existsSync(db) || (await rl.question(`已有声纹 ${db}，重新录入吗？(y/N) `)).trim().toLowerCase() === "y") await enroll();
  for (;;) {
    const c = (await rl.question("\n选择场次：1 主人  2 环境声音  3 自由测试  4 自由检测（非原生设备）  r 重新录入  q 退出 > ")).trim().toLowerCase();
    if (c === "1") await testSession("owner");
    else if (c === "2") await testSession("other");
    else if (c === "3") await testSession("free");
    else if (c === "4") await testSession("cross");
    else if (c === "r") await enroll();
    else if (c === "q") break;
  }
  const wavs = (k: string) => readdirSync(out).filter((f) => f.startsWith(k + "-") && f.endsWith(".wav")).map((f) => path.join(out, f));
  const owner = wavs("owner"), other = wavs("other");
  if (owner.length && other.length) {
    console.log("\n校准命令（在仓库根目录运行）：");
    const spkCli = path.resolve(path.dirname(exe), process.platform === "win32" ? "spk-cli.exe" : "spk-cli");
    console.log(`python tests/calibrate_speaker.py --cli ${spkCli} --model ${path.resolve(spk)} --vad ${path.resolve(vad)} ` +
                `--voiceprint ${db} --owner ${owner.join(" ")} --other ${other.join(" ")}`);
  } else {
    console.log("\n要校准阈值，至少需要一场“主人”和一场“环境声音”。");
  }
  rl.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
