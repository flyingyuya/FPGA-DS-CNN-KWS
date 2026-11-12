"""Evidence probe for the shift-only numeric contract (see docs/refactor-plan.md §3).

Question: can this DS-CNN run with power-of-two per-channel weight scales, a
pure floor-shift requantiser and a 24-bit accumulator -- and what does the
power-of-two restriction cost in accuracy?

  uv run python probe_shift.py                 # full val split
  uv run python probe_shift.py --batches 8     # quick smoke

The integer forward here is bit-exact with respect to the intended hardware
(zero point folded into the bias, floor-shift, saturating ReLU) and is the
prototype of the future int_model.py / tb/reference_model.py.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ACT_QMAX, IN_CLIP_HI, IN_CLIP_LO, LABELS, NUM_CLASSES,
                    W_QMAX)
from data import AudioDataset, build_pipeline
from features import input_qparams
from model import LAYER_SPEC, FC_SPEC
from train import load_qat, pick_device

DEV = pick_device("")
torch.set_grad_enabled(False)
STRIDE = {"stem": 2, "dw1": 2, "dw2": 2}


def pot_step(amax):
    """Largest power of two <= amax/W_QMAX -> finest PoT step that still spans amax."""
    x = math.log2(W_QMAX / max(amax, 1e-12))
    return 2.0 ** (-math.floor(x))


def build_qparams(prm, ranges, mode, verbose=True):
    """mode: 'pot' (hardware) or 'float' (ideal per-channel scale)."""
    s_prev, zp_in = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
    q = {"layers": [], "mode": mode}
    for (name, kind, cin, cout, stride) in LAYER_SPEC + [FC_SPEC]:
        w = prm[name + ".w"].astype(np.float64)
        b = prm[name + ".b"].astype(np.float64).reshape(-1)
        amax = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
        if mode == "pot":
            m = np.array([math.floor(math.log2(W_QMAX / a)) for a in amax], dtype=np.int64)
            step = 2.0 ** (-m)
        else:
            step = amax / W_QMAX
            m = np.zeros_like(step, dtype=np.int64)
        wq = np.clip(np.round(w / step.reshape((-1,) + (1,) * (w.ndim - 1))), -W_QMAX, W_QMAX)
        if name == "fc":
            j, s_out = 0, s_prev
        else:
            act_amax = ranges[name][1]
            j = math.floor(math.log2(ACT_QMAX * s_prev / act_amax))
            s_out = s_prev * 2.0 ** (-j)
        # fall back to the exact float ratio when the layer scale is free (mode=float)
        ratio = step * s_prev / s_out                     # acc -> out_code factor
        if mode == "pot":
            shift = m - j                                  # out = acc >>> shift
        else:
            shift = np.log2(1.0 / ratio)                   # informational only
        bias = b / (step * s_prev)                         # accumulator domain
        if zp_in != 0:                                     # fold the input zero point
            bias = bias - zp_in * wq.reshape(wq.shape[0], -1).sum(axis=1)
        bound = (np.abs(wq.reshape(wq.shape[0], -1)).sum(axis=1) * ACT_QMAX).max() + np.abs(bias).max()
        q["layers"].append(dict(name=name, kind=kind, cin=cin, cout=cout, s_in=s_prev,
                                s_out=s_out, j=j, shift=shift, wq=wq, bias=bias,
                                step=step, ratio=ratio, bound=float(bound)))
        s_prev, zp_in = s_out, 0
        if verbose:
            print(f"{name:5s} {kind:4s} cin={cin:2d} cout={cout:2d} "
                  f"shift={int(np.min(shift))}..{int(np.max(shift))} "
                  f"j={j:2d} s_out={s_out:.5f} |acc|bound={bound:11.0f} "
                  f"({math.ceil(math.log2(max(bound,1)))+1:2d}b) "
                  f"step/ideal={float(np.mean(step/(amax/W_QMAX))):.2f}")
    q["zp_in"] = int(input_qparams(IN_CLIP_LO, IN_CLIP_HI)[1])
    return q


def int_forward(x_code, q, trace=None):
    """Bit-exact forward.  Codes are UINT8, the input zero point is folded into
    the stem bias, and every requantisation is a floor-shift (as in the RTL)."""
    x = x_code.to(torch.float64)
    zp_in = float(q["zp_in"])
    for i, L in enumerate(q["layers"]):
        name, kind = L["name"], L["kind"]
        w = torch.from_numpy(L["wq"]).to(torch.float64).to(x.device)
        bias = torch.from_numpy(L["bias"]).to(torch.float64).reshape(1, -1).to(x.device)
        if kind == "fc":
            acc = F.linear(x.flatten(1), w) + bias
            if q["mode"] == "pot":
                sh = torch.from_numpy(np.asarray(L["shift"], dtype=np.float64)).reshape(1, -1).to(x.device)
                return torch.floor(acc / torch.pow(2.0, sh)), acc
            r = torch.from_numpy(np.asarray(L["ratio"])).to(x.device).reshape(1, -1)
            return torch.round(acc * r), acc
        w4 = w.reshape(w.shape[0], w.shape[1], 1, 1) if kind == "pw" else w
        pad = 0 if kind == "pw" else 1
        groups = L["cin"] if kind == "dw" else 1
        if pad:
            # taps outside the tensor carry the layer's input zero point, which is
            # what keeps the folded bias exact
            x = F.pad(x, (1, 1, 1, 1), value=(zp_in if i == 0 else 0.0))
        acc = F.conv2d(x, w4, None, STRIDE.get(name, 1), 0, groups=groups)
        if trace is not None:
            trace.append((name, float(acc.abs().max())))
        acc = acc + bias.reshape((1, -1) + (1,) * 2)
        if q["mode"] == "pot":
            sh = torch.from_numpy(np.asarray(L["shift"], dtype=np.float64)).reshape((1, -1, 1, 1)).to(x.device)
            x = torch.clamp(torch.floor(acc / torch.pow(2.0, sh)), 0, ACT_QMAX)
        else:
            r = torch.from_numpy(np.asarray(L["ratio"])).to(x.device).reshape((1, -1, 1, 1))
            x = torch.clamp(torch.round(acc * r), 0, ACT_QMAX)
        if name == "pw4":                      # global average pool of the codes
            x = torch.floor(x.mean(dim=(2, 3), keepdim=True))
    return None, None


def evaluate(q, split, nb):
    pipe = build_pipeline(DEV).eval()
    ds = AudioDataset(split)
    n = len(ds) if nb == 0 else min(len(ds), nb * 128)
    s, zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
    preds, gts, trace = [], [], []
    for i in range(0, n, 128):
        wav = torch.from_numpy(np.array(ds.audio[i:i + 128])).to(DEV)
        xq = torch.clamp(torch.round(pipe(wav) / s) + zp, 0, ACT_QMAX)
        preds.append(int_forward(xq, q, trace)[0].argmax(1).cpu())
        gts.append(torch.from_numpy(ds.labels[i:i + 128]))
    pred, gt = torch.cat(preds), torch.cat(gts)
    conf = torch.bincount(gt * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return 100.0 * (pred == gt).float().mean().item(), conf, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=0, help="0 = all")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    qm, ck = load_qat(DEV)
    print(f"QAT ckpt: epoch {ck['epoch']} val {ck['val_acc']:.2f}% (fake-quant baseline)")
    prm, ranges = qm.float_params(), qm.act_ranges()

    print("\n-- hardware contract: PoT per-channel weights + pure right shift --")
    q_pot = build_qparams(prm, ranges, "pot")
    print("\n-- ideal: exact per-channel scale + round --")
    q_flt = build_qparams(prm, ranges, "float", verbose=False)

    print(f"\n-- {a.split} split ({len(AudioDataset(a.split))} clips) --")
    acc_f, conf_f, _ = evaluate(q_flt, a.split, a.batches)
    print(f"  exact per-channel scale + round : {acc_f:.2f}%")
    acc_p, conf_p, trace = evaluate(q_pot, a.split, a.batches)
    print(f"  PoT + shift (hardware)          : {acc_p:.2f}%   delta {acc_p - acc_f:+.2f}")

    print("\n-- accumulator peak observed on real data --")
    seen = {}
    for name, v in trace:
        seen[name] = max(seen.get(name, 0.0), v)
    for name, v in seen.items():
        print(f"  {name:5s} peak {v:10.0f}  -> {math.ceil(math.log2(max(v,1)))+1:2d} bit")

    print("\nconfusion (PoT + shift):")
    print("            " + " ".join(f"{l[:5]:>5s}" for l in LABELS))
    for i, row in enumerate(conf_p.tolist()):
        print(f"  {LABELS[i]:>9s} " + " ".join(f"{v:5d}" for v in row))


main()
