// extractZip on archives built here: stored and deflated entries, Unicode names and
// destinations, empty files, and entries trying to escape the destination.
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";
import zlib from "node:zlib";

import { extractZip } from "../src/unzip.ts";

// A minimal ZIP writer (local headers, central directory, end record), UTF-8 names.
function makeZip(entries: { name: string; data: Buffer; deflate?: boolean }[]): Buffer {
  const locals: Buffer[] = [];
  const centrals: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name, "utf8");
    const body = e.deflate ? zlib.deflateRawSync(e.data) : e.data;
    const crc = zlib.crc32(e.data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x800, 6);
    local.writeUInt16LE(e.deflate ? 8 : 0, 8);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(e.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    locals.push(local, name, body);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x800, 8);
    central.writeUInt16LE(e.deflate ? 8 : 0, 10);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(body.length, 20);
    central.writeUInt32LE(e.data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);
    offset += 30 + name.length + body.length;
  }
  const cd = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cd.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, end]);
}

describe("extractZip", () => {
  let tmp: string;
  before(() => (tmp = fs.mkdtempSync(path.join(os.tmpdir(), "voice-unzip-"))));
  after(() => fs.rmSync(tmp, { recursive: true, force: true }));

  it("extracts stored, deflated and empty entries with Unicode names into a Unicode directory", async () => {
    const text = Buffer.from("模型 model モデル 모델 ".repeat(5000));
    const bin = Buffer.from(Array.from({ length: 70000 }, (_, i) => (i * 7919) % 256));
    const zip = path.join(tmp, "a.zip");
    fs.writeFileSync(zip, makeZip([
      { name: "licenses/", data: Buffer.alloc(0) },
      { name: "licenses/说明_한글_😀.txt", data: text, deflate: true },
      { name: "bin/ggml.dll", data: bin },
      { name: "empty.txt", data: Buffer.alloc(0) },
    ]));
    const dest = path.join(tmp, "用户_张三_한글", "x");
    fs.mkdirSync(dest, { recursive: true });
    await extractZip(zip, dest);
    assert.deepEqual(fs.readFileSync(path.join(dest, "licenses", "说明_한글_😀.txt")), text);
    assert.deepEqual(fs.readFileSync(path.join(dest, "bin", "ggml.dll")), bin);
    assert.equal(fs.statSync(path.join(dest, "empty.txt")).size, 0);
  });

  it("refuses entries that would land outside the destination", async () => {
    for (const name of ["../evil.txt", "a/../../evil.txt", "C:/Windows/evil.txt", "/evil.txt", "..\\evil.txt"]) {
      const zip = path.join(tmp, "slip.zip");
      fs.writeFileSync(zip, makeZip([{ name, data: Buffer.from("x") }]));
      const dest = path.join(tmp, "slip", "x");
      fs.mkdirSync(dest, { recursive: true });
      await assert.rejects(extractZip(zip, dest), /unsafe path/, name);
      assert.equal(fs.existsSync(path.join(tmp, "slip", "evil.txt")), false, name);
    }
  });

  it("rejects a file that is not a zip archive", async () => {
    const bad = path.join(tmp, "bad.zip");
    fs.writeFileSync(bad, "not a zip");
    await assert.rejects(extractZip(bad, path.join(tmp, "bad")), /not a zip archive/);
  });
});
