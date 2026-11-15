"""Ablation lab for the numeric contract (evidence for docs/quantisation-math.md).

Five quantisation schemes are run through ONE integer forward, so the only
thing that changes between rows is the scheme itself:

    ideal          free per-channel step, exact activation scale, round    <- upper bound
    hw             step = rho*2**-S, exact scale, round-half-up            <- THE DESIGN
    hw_trunc       step = rho*2**-S, exact scale, truncating shift         <- what the rounding constant buys
    pot            step = 2**-m,     power-of-two scale, truncating shift  <- naive PoT
    pot_nearest    step = 2**-round(...)  (clips the largest weights)

The integer forward here is independent of QDSCNN.forward on purpose: it is the
prototype of int_model.py, and agreement between the two is a cross-check.

  uv run python probe_shift.py                # full val split
  uv run python probe_shift.py --batches 8    # smoke
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from config import (ACT_QMAX, GAP_SHIFT, IN_CLIP_HI, IN_CLIP_LO, LOGIT_SCALE,
                    W_QMAX)
from data import AudioDataset, build_pipeline
from features import input_qparams
from model import FC_SPEC, LAYER_SPEC
from quantisation import GAP_N
from train import load_qat, pick_device

DEV = pick_device("")
torch.set_grad_enabled(False)
STRIDE = {"stem": 2, "dw1": 2, "dw2": 2}
MODES = ("ideal", "hw", "hw_trunc", "pot", "pot_nearest")


def build(mode, prm, ranges):
    """Per-layer integer constants for one scheme."""
    s_in, zp_in = (float(x) for x in input_qparams(IN_CLIP_LO, IN_CLIP_HI))
    layers = []
    for (name, kind, cin, cout, stride) in LAYER_SPEC + [FC_SPEC]:
        w = prm[name + ".w"].astype(np.float64)
        b = prm[name + ".b"].astype(np.float64).reshape(-1)
        amax = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
        ideal = amax / W_QMAX
        is_fc = name == "fc"
        # the FC's output scale is free (only argmax is used); 1/16 keeps a
        # usable logit resolution
        act_amax = ACT_QMAX * LOGIT_SCALE if is_fc else ranges[name][1]

        if mode in ("ideal", "hw", "hw_trunc"):
            s_out, j = act_amax / ACT_QMAX, None
        else:                                   # every scale forced to s_in*2**-j
            j = math.floor(math.log2(ACT_QMAX * s_in / act_amax))
            s_out = s_in * 2.0 ** (-j)

        if mode == "ideal":                     # no shift at all, a real multiply
            step, shift = ideal, None
        elif mode in ("hw", "hw_trunc"):
            rho = s_out / s_in
            shift = np.array([min(15, max(0, math.floor(math.log2(rho / a)))) for a in ideal])
            step = rho * 2.0 ** (-shift)
        else:
            rnd = math.floor if mode == "pot" else round
            m = np.array([max(0, rnd(math.log2(W_QMAX / a))) for a in amax])
            step, shift = 2.0 ** (-m), m - j

        shape = (-1,) + (1,) * (w.ndim - 1)
        q = np.clip(np.round(w / step.reshape(shape)), -W_QMAX, W_QMAX)   # clips in pot_nearest
        B = np.round(b / (step * s_in))
        if zp_in:
            B = B - zp_in * q.reshape(q.shape[0], -1).sum(axis=1)
        if mode == "hw":                        # 2**(S-1) folded into the bias
            B = B + np.power(2.0, np.maximum(shift - 1, 0))

        layers.append(dict(name=name, kind=kind, cin=cin, s_out=s_out, q=q, B=B,
                           zp_pad=int(zp_in),
                           step=step, shift=shift, ratio=step * s_in / s_out,
                           # the rounding constant already sits in B for "hw",
                           # so only "ideal" rounds here (it has no shift at all)
                           trunc=(mode != "ideal")))
        s_in, zp_in = s_out, 0
        if name == "pw4":                       # global average pool, sum >>> 6
            s_in = s_in * 2 ** GAP_SHIFT / GAP_N
            layers.append(dict(name="gap", kind="gap", s_in=s_in))
    return layers


def _requant(acc, L):
    """acc -> output code, exactly what the RTL does for this scheme."""
    if L["shift"] is None:                      # ideal: one real multiply
        ratio = torch.from_numpy(np.full(acc.shape[1], L["ratio"]))
        ratio = ratio.reshape((1, -1) + (1,) * (acc.dim() - 2)).to(acc.device)
        return torch.round(acc * ratio)
    shift = torch.from_numpy(np.atleast_1d(np.asarray(L["shift"], dtype=np.float64)))
    shift = shift.reshape((1, -1) + (1,) * (acc.dim() - 2)).to(acc.device)
    scaled = acc / torch.pow(2.0, shift)
    return torch.floor(scaled) if L["trunc"] else torch.floor(scaled + 0.5)


def forward(x_code, layers):
    """Bit-exact integer forward; float64 holds integers exactly below 2**53."""
    x = x_code.to(torch.float64)
    for L in layers:
        if L["kind"] == "gap":
            x = torch.floor(x.sum(dim=(2, 3), keepdim=True) / 2 ** GAP_SHIFT + 0.5)
            x = torch.clamp(x, 0, ACT_QMAX)
            continue
        q = torch.from_numpy(L["q"]).to(x.device)
        B = torch.from_numpy(L["B"]).reshape(1, -1).to(x.device)
        if L["kind"] == "fc":
            return _requant(F.linear(x.flatten(1), q) + B, L)
        groups = L["cin"] if L["kind"] == "dw" else 1
        q4 = q.reshape(q.shape[0], q.shape[1], 1, 1) if L["kind"] == "pw" else q
        if L["kind"] != "pw":
            # the RTL feeds the input zero point for out-of-range taps, which is
            # exactly what keeps the folded bias exact (math notes, sec. 3)
            x = F.pad(x, (1, 1, 1, 1), value=float(L["zp_pad"]))
        acc = F.conv2d(x, q4, None, STRIDE.get(L["name"], 1), 0, groups=groups)
        acc = acc + B.reshape((1, -1) + (1,) * 2)
        x = torch.clamp(_requant(acc, L), 0, ACT_QMAX)
    return None


def run(mode, prm, ranges, split, nb):
    ds = AudioDataset(split)
    n = len(ds) if nb == 0 else min(len(ds), nb * 128)
    pipe = build_pipeline(DEV).eval()
    s, zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
    layers = build(mode, prm, ranges)
    pred, gts = [], []
    for i in range(0, n, 128):
        wav = torch.from_numpy(np.array(ds.audio[i:i + 128])).to(DEV)
        xq = torch.clamp(torch.round(pipe(wav) / s) + zp, 0, ACT_QMAX)
        pred.append(forward(xq, layers).argmax(1).cpu())
        gts.append(torch.from_numpy(ds.labels[i:i + 128]))
    pred, gt = torch.cat(pred), torch.cat(gts)
    return 100.0 * (pred == gt).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    qm, ck = load_qat(DEV)
    prm, ranges = qm.float_params(), qm.act_ranges()
    print(f"QAT ckpt epoch {ck['epoch']}, fake-quant val {ck['val_acc']:.2f}%\n")
    base = None
    for mode in MODES:
        acc = run(mode, prm, ranges, a.split, a.batches)
        if base is None:
            base = acc
        print(f"  {mode:12s} {acc:6.2f}%   delta {acc - base:+6.2f}")


main()
