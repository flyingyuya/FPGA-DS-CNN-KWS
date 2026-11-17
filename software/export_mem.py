"""Export the trained model into every memory image the accelerator loads.

    uv run python export_mem.py               # write into artifacts/export/
    uv run python export_mem.py --install     # write into the project root

The writers live in ../sim/reference_model.py, which is the single source of
truth: it packs the ROM images, replays them with the same word/lane addressing
the RTL uses, and turns that replay into the golden vectors.  This script only
supplies the *content* (trained weights, real Speech Commands frames) and then
cross-checks three independent implementations against each other:

    QDSCNN.forward   float fake-quant, the training graph
    IntDSCNN         batched integer model, the accuracy report
    replay()         packed images + RTL addressing, the golden vectors

Produced tree (relative to the output root):
    rtl/kws_params.vh                 geometry, widths, ROM bases
    mem/weights/w_all.mem             shared weight ROM, 32 lanes x 8 bit
    mem/bias/b_all.mem                {shift[3:0], bias[23:0]} per channel
    mem/input/mfcc_<vector>.mem       14 UINT8 frames of 40 x 101
    golden/act_<layer>_<vector>.mem   per-layer activations (layer testbenches)
    golden/golden.json                logits + class for every vector
    golden/unit/*.mem                 PE-array and argmax unit vectors
    docs/quant_summary.csv            per-layer contract table
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

from config import (ACT_QMAX, IN_CLIP_HI, IN_CLIP_LO, LABELS, PRECISION,
                    SEED)
from data import AudioDataset, build_pipeline
from features import input_qparams
from int_model import IntDSCNN, accuracy
from train import load_qat, pick_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import reference_model as rm                                    # noqa: E402

STAGING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "export")


@torch.no_grad()
def quantised_split(split, pipe, s, zp, device, batch=256):
    ds = AudioDataset(split)
    xs = []
    for i in range(0, len(ds), batch):
        wav = torch.from_numpy(np.array(ds.audio[i:i + batch])).to(device)
        xs.append(torch.clamp(torch.round(pipe(wav) / s) + zp, 0, ACT_QMAX).cpu())
    return torch.cat(xs), torch.from_numpy(ds.labels)


def pick_vectors(xq, y, pred):
    """One test clip per class (the first one the integer model gets right, else
    the first one), plus the two extreme frames all-0 and all-255."""
    names, frames, src = [], [], []
    for c, lab in enumerate(LABELS):
        idx = torch.nonzero(y == c).flatten()
        if len(idx) == 0:
            raise RuntimeError(f"no test clip for class {lab}")
        ok = idx[pred[idx] == c]
        i = int(ok[0] if len(ok) else idx[0])
        names.append(f"class{c:02d}_{lab}")
        frames.append(xq[i, 0].numpy().astype(np.uint8))
        src.append(i)
    for nm, v in (("edge_all_min", 0), ("edge_all_max", 255)):
        names.append(nm)
        frames.append(np.full((rm.H_IN, rm.W_IN), v, dtype=np.uint8))
        src.append(-1)
    return names, frames, src


def cross_check(hw, model, frames, names, n=4):
    """replay() must reproduce IntDSCNN bit for bit, logits included."""
    for name, f in list(zip(names, frames))[:n]:
        g = rm.replay(hw, f)
        mine = model(torch.from_numpy(f[None, None].astype(np.float32) * 1.0)).round()[0]
        mine = mine.cpu().numpy().astype(np.int64)
        assert np.array_equal(g["logits"], mine), (
            f"replay() != IntDSCNN on {name}:\n  replay {g['logits']}\n  int    {mine}")
    print(f"  replay() == IntDSCNN bit-exact on {min(n, len(names))} vectors")


def contract_rows(hw, model):
    rows = []
    for l in rm.LAYERS:
        n = l["name"]
        if n not in hw or "B" not in hw[n]:
            continue
        q = np.asarray(hw[n]["q"])
        acc = float(np.abs(q).reshape(q.shape[0], -1).sum(axis=1).max() * ACT_QMAX
                    + np.abs(hw[n]["B"]).max())
        bits = int(np.ceil(np.log2(acc + 1))) + 1
        rows.append((n, l["kind"], l["cin"], l["cout"], l["hi"], l["wi"], l["ho"],
                     l["wo"], l["stride"], l["pad"], l["ng"], l["pixels"],
                     round(hw[n]["s_in"], 8), round(hw[n]["s_out"], 8),
                     int(np.min(hw[n]["shift"])), int(np.max(hw[n]["shift"])),
                     int(np.abs(q).max()), bits, l["wrom_base"], l["ch_base"]))
    return rows


def export(out_root, device, n_check=4):
    qm, ck = load_qat("cpu")
    qm.check_contract()                       # raises if the RTL cannot hold it
    hw = qm.hw_params()
    print(f"QAT checkpoint: epoch {ck['epoch']}, fake-quant val {ck['val_acc']:.2f}%")

    model = IntDSCNN(hw, device)
    pipe = build_pipeline(device).eval()
    s, zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)

    print("\n-- integer test-set accuracy --")
    xq, y = quantised_split("test", pipe, s, zp, device)
    acc, conf, pred = accuracy(model, xq, y)
    print(f"  bit-exact {PRECISION} accelerator: {acc:.2f}% on {len(y)} clips")

    names, frames, src = pick_vectors(xq, y, pred)
    print("\n-- cross-checks --")
    cross_check(hw, model, frames, names, n_check)

    print(f"\n-- writing images under {out_root} --")
    os.makedirs(os.path.join(out_root, "rtl"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "docs"), exist_ok=True)
    rom, bias, shift = rm.emit_roms(hw, out_root)
    rm.emit_vh(hw, os.path.join(out_root, "rtl", "kws_params.vh"))
    rm.emit_inputs(dict(zip(names, frames)), out_root)
    results = rm.emit_golden(hw, dict(zip(names, frames)), out_root)
    n_pe, n_am = rm.emit_unit_vectors(hw, np.random.default_rng(SEED), out_root)
    print(f"  weight ROM {int((rom != 0).any(axis=1).sum())} / "
          f"{rm.WROM_WORDS} words used, {rm.CH_TOTAL} bias words")
    print(f"  unit vectors: {n_pe} PE jobs, {n_am} argmax rows")
    print(f"  logits/class for {len(results)} input vectors")

    rows = contract_rows(hw, model)
    with open(os.path.join(out_root, "docs", "quant_summary.csv"), "w", newline="\n") as f:
        f.write(rm.quant_summary(rows))
    report = dict(
        precision=PRECISION, test_clips=int(len(y)), int_test_acc=acc,
        qat_val_acc=float(ck["val_acc"]), confusion=conf.tolist(), labels=LABELS,
        vectors=[dict(name=n, test_index=i, golden=results[n]["label"],
                      logits=results[n]["logits"], cls=results[n]["cls"])
                 for n, i in zip(names, src)],
        contract=[dict(zip(("layer", "kind", "cin", "cout", "hi", "wi", "ho", "wo",
                            "stride", "pad", "ng", "pixels", "s_in", "s_out",
                            "shift_min", "shift_max", "max_abs_q", "acc_bits",
                            "wrom_base", "ch_base"), r)) for r in rows],
        frontend=dict(sample_rate=16000, n_fft=512, win=480, hop=160, n_mels=40,
                      n_mfcc=40, fmin=20.0, fmax=7600.0,
                      input_scale=float(s), input_zero_point=int(zp)),
    )
    with open(os.path.join(out_root, "docs", "ml_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out_root}/docs/ml_report.json")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help=f"write straight into {ROOT} (overwrites rtl/kws_params.vh, "
                         "mem/, golden/, docs/)")
    ap.add_argument("--out", default=STAGING, help="staging output root")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    out = ROOT if a.install else a.out
    if out == STAGING and os.path.isdir(out):
        shutil.rmtree(out)                     # only ever clears our own staging dir
    rep = export(out, pick_device(a.device))
    print(f"\n{rep['precision']} test accuracy: {rep['int_test_acc']:.2f}%")


if __name__ == "__main__":
    main()
