#!/usr/bin/env python3
"""Export 3D-Speaker / FunASR CAM++ (speech_campplus_sv_zh-cn_16k-common) to GGUF.

Offline only (needs torch + gguf). Layout written for src/spk/campplus.cpp:
  * BatchNorm -> "<prefix>.scale" / "<prefix>.shift" (inference affine, eps 1e-5)
  * Conv2d    -> unchanged torch layout (Cout, Cin, KH, KW) = ggml [KW, KH, Cin, Cout]
  * Conv1d with kernel 1 -> (Cout, Cin)
  * Conv1d with kernel K > 1 -> (K, Cout, Cin), so each tap is a contiguous [Cin, Cout] matrix
  * biases unchanged

    python tools/export_campplus_gguf.py --ckpt campplus_cn_common.bin --out models/campplus-zh-cn.gguf
"""
import argparse
import os

import gguf
import numpy as np
import torch

BN_EPS = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="campplus_cn_common.bin (state dict)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    w = gguf.GGUFWriter(args.out, "campplus")
    w.add_uint32("campplus.feat_dim", 80)
    w.add_uint32("campplus.embedding_size", 192)
    w.add_uint32("campplus.growth_rate", 32)
    w.add_uint32("campplus.bn_size", 4)
    w.add_uint32("campplus.init_channels", 128)
    w.add_array("campplus.block_layers", [12, 24, 16])
    w.add_array("campplus.block_kernel", [3, 3, 3])
    w.add_array("campplus.block_dilation", [1, 2, 2])
    w.add_string("campplus.source", "iic/speech_campplus_sv_zh-cn_16k-common (campplus_cn_common.bin)")

    done = set()
    n = 0
    for k in sd:
        if k in done:
            continue
        if k.endswith(".running_mean"):
            p = k[: -len(".running_mean")]
            mean = sd[p + ".running_mean"].double()
            var = sd[p + ".running_var"].double()
            gamma = sd[p + ".weight"].double() if p + ".weight" in sd else torch.ones_like(mean)
            beta = sd[p + ".bias"].double() if p + ".bias" in sd else torch.zeros_like(mean)
            scale = gamma / torch.sqrt(var + BN_EPS)
            shift = beta - mean * scale
            w.add_tensor(p + ".scale", scale.float().numpy())
            w.add_tensor(p + ".shift", shift.float().numpy())
            done.update(x for x in (p + s for s in (".running_mean", ".running_var", ".weight", ".bias")) if x in sd)
            n += 2
    for k, v in sd.items():
        if k in done:
            continue
        a = v.detach().float().numpy()
        if a.ndim == 3:  # Conv1d (Cout, Cin, K)
            a = a[:, :, 0] if a.shape[2] == 1 else np.ascontiguousarray(a.transpose(2, 0, 1))
        w.add_tensor(k, np.ascontiguousarray(a))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {n} tensors to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
