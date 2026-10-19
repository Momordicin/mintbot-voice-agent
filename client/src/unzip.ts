// Minimal ZIP extraction for the release archives (tools/package_release.py writes
// them with Python's zipfile: stored or deflated entries, no ZIP64). Pure Node with
// wide-character paths: Windows' tar.exe takes its arguments in the ANSI code page,
// so it fails for a directory whose name the system code page cannot hold (e.g. a
// Chinese user name on an English system).
import { createReadStream, createWriteStream } from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { pipeline } from "node:stream/promises";
import { createInflateRaw } from "node:zlib";

interface ZipEntry {
  name: string;
  method: number; // 0 stored, 8 deflated
  compressedSize: number;
  size: number;
  localHeaderOffset: number;
}

const EOCD = 0x06054b50;
const CENTRAL = 0x02014b50;
const LOCAL = 0x04034b50;

async function readAt(fd: fs.FileHandle, position: number, length: number): Promise<Buffer> {
  const buf = Buffer.alloc(length);
  const { bytesRead } = await fd.read(buf, 0, length, position);
  return buf.subarray(0, bytesRead);
}

async function listEntries(fd: fs.FileHandle, fileSize: number): Promise<ZipEntry[]> {
  // the end-of-central-directory record is within the last 22 + 65535 (comment) bytes
  const tailLength = Math.min(fileSize, 22 + 0xffff);
  const tail = await readAt(fd, fileSize - tailLength, tailLength);
  let eocd = -1;
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail.readUInt32LE(i) === EOCD) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("not a zip archive (no end of central directory)");
  const count = tail.readUInt16LE(eocd + 10);
  const cdSize = tail.readUInt32LE(eocd + 12);
  const cdOffset = tail.readUInt32LE(eocd + 16);
  if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) throw new Error("ZIP64 archives are not supported");
  const cd = await readAt(fd, cdOffset, cdSize);
  const entries: ZipEntry[] = [];
  let p = 0;
  for (let i = 0; i < count; i++) {
    if (p + 46 > cd.length || cd.readUInt32LE(p) !== CENTRAL) throw new Error("corrupt central directory");
    const flags = cd.readUInt16LE(p + 8);
    const nameLength = cd.readUInt16LE(p + 28);
    const extraLength = cd.readUInt16LE(p + 30);
    const commentLength = cd.readUInt16LE(p + 32);
    const nameBytes = cd.subarray(p + 46, p + 46 + nameLength);
    entries.push({
      // bit 11: UTF-8 name; otherwise code page 437, which equals ASCII for our names
      name: nameBytes.toString(flags & 0x800 ? "utf8" : "latin1"),
      method: cd.readUInt16LE(p + 10),
      compressedSize: cd.readUInt32LE(p + 20),
      size: cd.readUInt32LE(p + 24),
      localHeaderOffset: cd.readUInt32LE(p + 42),
    });
    if (entries.at(-1)!.size === 0xffffffff || entries.at(-1)!.compressedSize === 0xffffffff)
      throw new Error("ZIP64 entries are not supported");
    p += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

// The entry's path under dest; refuses absolute names and ".." (zip slip).
function target(dest: string, name: string): string {
  const parts = name.split(/[\\/]/).filter((s) => s && s !== ".");
  if (/^[a-zA-Z]:/.test(name) || name.startsWith("/") || name.startsWith("\\") || parts.includes(".."))
    throw new Error(`unsafe path in archive: ${name}`);
  return path.join(dest, ...parts);
}

export async function extractZip(archive: string, dest: string, signal?: AbortSignal): Promise<void> {
  const fd = await fs.open(archive, "r");
  try {
    const { size } = await fd.stat();
    for (const e of await listEntries(fd, size)) {
      signal?.throwIfAborted();
      const out = target(dest, e.name);
      if (e.name.endsWith("/")) {
        await fs.mkdir(out, { recursive: true });
        continue;
      }
      if (e.method !== 0 && e.method !== 8) throw new Error(`${e.name}: unsupported compression method ${e.method}`);
      const local = await readAt(fd, e.localHeaderOffset, 30);
      if (local.length < 30 || local.readUInt32LE(0) !== LOCAL) throw new Error(`${e.name}: corrupt local header`);
      const dataStart = e.localHeaderOffset + 30 + local.readUInt16LE(26) + local.readUInt16LE(28);
      await fs.mkdir(path.dirname(out), { recursive: true });
      const write = createWriteStream(out);
      if (e.compressedSize === 0) {
        write.end();
        await new Promise<void>((resolve, reject) => write.on("finish", resolve).on("error", reject));
      } else {
        const read = createReadStream(archive, { start: dataStart, end: dataStart + e.compressedSize - 1 });
        if (e.method === 8) await pipeline(read, createInflateRaw(), write, { signal });
        else await pipeline(read, write, { signal });
      }
    }
  } finally {
    await fd.close();
  }
}
