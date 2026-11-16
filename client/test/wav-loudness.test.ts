// normalizeWavLoudness on WAV files built here: quiet and peaky PCM 16-bit audio scaled
// by one uniform gain, a -32768 sample, silence, an extra chunk before "data", a 32-bit float
// file, a data size larger than the file and bytes that are no WAV.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { normalizeWavLoudness } from "../src/wav-loudness.ts";

function chunk(id: string, body: Buffer): Buffer {
  const header = Buffer.alloc(8);
  header.write(id, 0, "latin1");
  header.writeUInt32LE(body.length, 4);
  return Buffer.concat([header, body, Buffer.alloc(body.length & 1)]);
}

function makeWav(samples: number[], o: { format?: number; bits?: number; before?: Buffer[] } = {}): Buffer {
  const fmt = Buffer.alloc(16);
  fmt.writeUInt16LE(o.format ?? 1, 0);
  fmt.writeUInt16LE(1, 2);
  fmt.writeUInt32LE(32000, 4);
  fmt.writeUInt32LE(32000 * 2, 8);
  fmt.writeUInt16LE(2, 12);
  fmt.writeUInt16LE(o.bits ?? 16, 14);
  const data = Buffer.alloc(samples.length * 2);
  samples.forEach((s, i) => data.writeInt16LE(s, i * 2));
  const body = Buffer.concat([Buffer.from("WAVE", "latin1"), chunk("fmt ", fmt), ...(o.before ?? []), chunk("data", data)]);
  return Buffer.concat([Buffer.from("RIFF", "latin1"), Buffer.from(Uint32Array.of(body.length).buffer), body]);
}

function pcm(wav: Buffer): number[] {
  const start = wav.indexOf("data", 12, "latin1") + 8;
  return Array.from({ length: (wav.length - start) / 2 }, (_, i) => wav.readInt16LE(start + i * 2));
}

function levels(wav: Buffer): { rmsDb: number; peakDb: number } {
  const samples = pcm(wav);
  const rms = Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0) / samples.length);
  const peak = Math.max(...samples.map(Math.abs));
  return { rmsDb: 20 * Math.log10(rms / 32768), peakDb: 20 * Math.log10(peak / 32768) };
}

const sine = (amplitude: number) => Array.from({ length: 3200 }, (_, i) => Math.round(amplitude * Math.sin((2 * Math.PI * 220 * i) / 32000)));

describe("normalizeWavLoudness", () => {
  for (const [name, samples, rmsDb, peakDb] of [
    ["quiet input is raised to -20 dBFS RMS", sine(300), -20, null],
    ["the peak cap wins when the peak would exceed -1 dBFS", [...Array.from({ length: 3200 }, (_, i) => (i % 2 ? 100 : -100)), 3000], null, -1],
    ["the peak cap wins for a -32768 sample among quiet ones", [...Array.from({ length: 3200 }, (_, i) => (i % 2 ? 100 : -100)), -32768], null, -1],
  ] as const) {
    it(name, () => {
      const input = makeWav([...samples]);
      const out = normalizeWavLoudness(input);
      const got = levels(out);
      if (rmsDb !== null) assert.ok(Math.abs(got.rmsDb - rmsDb) < 0.01, `rms ${got.rmsDb}`);
      if (peakDb !== null) assert.ok(Math.abs(got.peakDb - peakDb) < 0.01, `peak ${got.peakDb}`);
      assert.ok(got.peakDb <= -1 + 0.001, `peak ${got.peakDb}`);
      assert.ok(got.rmsDb <= -20 + 0.01, `rms ${got.rmsDb}`);
      assert.deepEqual(out.subarray(0, 44), input.subarray(0, 44));
      const [a, b] = [pcm(input), pcm(out)];
      const bounds = a.map((s, i) => [(b[i] - 0.5) / s, (b[i] + 0.5) / s].sort((x, y) => x - y));
      const [lo, hi] = [Math.max(...bounds.map(([x]) => x)), Math.min(...bounds.map(([, y]) => y))];
      assert.ok(lo <= hi + 1e-9, `no single gain within 0.5 of every sample: ${lo} > ${hi}`);
    });
  }

  it("keeps a chunk before data, with its padding byte, and scales the samples after it", () => {
    const list = chunk("LIST", Buffer.from("INFOISFT\x05\x00\x00\x00sf.5\x00", "latin1"));
    assert.equal(list.length, 8 + 17 + 1);
    const input = makeWav(sine(300), { before: [list] });
    const out = normalizeWavLoudness(input);
    const dataAt = input.indexOf("data", 12, "latin1");
    assert.deepEqual(out.subarray(0, dataAt + 8), input.subarray(0, dataAt + 8));
    assert.ok(Math.abs(levels(out).rmsDb + 20) < 0.01);
  });

  for (const [name, input] of [
    ["all-zero audio", makeWav(new Array(32000).fill(0))],
    ["a 32-bit float file", makeWav(sine(300), { format: 3, bits: 32 })],
    ["a data size larger than the file", (() => { const wav = makeWav(sine(300)); wav.writeUInt32LE(0xffffffff, 40); return wav; })()],
    ["bytes that are no RIFF/WAVE", Buffer.from("RIFF")],
  ] as const) {
    it(`returns ${name} unchanged`, () => {
      assert.deepEqual(normalizeWavLoudness(input), input);
    });
  }
});
