// Loudness normalization of the WAV bytes GPT-SoVITS returns: its output is as loud as
// its training material, often far quieter than normal speech. PCM 16-bit RIFF/WAVE only;
// anything else, an unparseable file or all-zero audio is returned unchanged.
//
//   const wav = normalizeWavLoudness(Buffer.from(await response.arrayBuffer()));

const FULL_SCALE = 32768;
const TARGET_RMS = FULL_SCALE * 10 ** (-20 / 20);
const MAX_PEAK = FULL_SCALE * 10 ** (-1 / 20);

// Scales the data chunk's samples so their RMS is -20 dBFS, with the gain capped so the
// peak is at most -1 dBFS; the same gain for every channel. Every byte outside the samples
// is kept. Returns a new buffer when it scales, otherwise the input itself.
export function normalizeWavLoudness(wav: Buffer): Buffer {
  if (wav.length < 12 || wav.toString("latin1", 0, 4) !== "RIFF" || wav.toString("latin1", 8, 12) !== "WAVE") return wav;
  let pcm16 = false;
  let fmtFound = false;
  let dataStart = -1;
  let dataSize = 0;
  for (let offset = 12; offset + 8 <= wav.length;) {
    const id = wav.toString("latin1", offset, offset + 4);
    const size = wav.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (body + size > wav.length) return wav;
    if (id === "fmt ") {
      if (size < 16) return wav;
      fmtFound = true;
      pcm16 = wav.readUInt16LE(body) === 1 && wav.readUInt16LE(body + 14) === 16;
    } else if (id === "data") {
      dataStart = body;
      dataSize = size;
    }
    offset = body + size + (size & 1);
  }
  if (!fmtFound || !pcm16 || dataStart < 0) return wav;
  const count = Math.floor(dataSize / 2);
  let sumSquares = 0;
  let peak = 0;
  for (let i = 0; i < count; i++) {
    const s = wav.readInt16LE(dataStart + i * 2);
    sumSquares += s * s;
    peak = Math.max(peak, Math.abs(s));
  }
  if (peak === 0) return wav;
  const rms = Math.sqrt(sumSquares / count);
  const gain = Math.min(TARGET_RMS / rms, MAX_PEAK / peak);
  const out = Buffer.from(wav);
  for (let i = 0; i < count; i++) {
    const at = dataStart + i * 2;
    out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(wav.readInt16LE(at) * gain))), at);
  }
  return out;
}
