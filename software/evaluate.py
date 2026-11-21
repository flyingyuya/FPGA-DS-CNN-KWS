"""Accuracy of every model stage on the Speech Commands test split.

  python evaluate.py            # FP32, BN-folded FP32, QAT fake-quant, bit-exact INT

The last row is the number that matters: it is produced by the same integer
arithmetic the RTL implements, from UINT8 MFCC codes to the INT16 logits.
"""

import argparse
import os

import torch

from config import CKPT_FP32, CKPT_QAT, IN_CLIP_HI, IN_CLIP_LO, LABELS, PRECISION
from features import input_qparams
from data import build_pipeline
from export_mem import quantised_split
from int_model import IntDSCNN, accuracy
from train import evaluate_model, load_fp32, load_qat, make_loader, pick_device


def print_confusion(conf):
    w = max(len(l) for l in LABELS)
    print(" " * (w + 2) + " ".join(f"{i:>5d}" for i in range(len(LABELS))))
    for i, row in enumerate(conf.tolist()):
        rec = 100.0 * row[i] / max(1, sum(row))
        print(f"{LABELS[i]:>{w}s}  " + " ".join(f"{v:5d}" for v in row) + f"   {rec:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    dev = pick_device(a.device)
    pipe = build_pipeline(dev).eval()
    dl = make_loader(a.split, False)
    rows = []
    if os.path.isfile(CKPT_FP32):
        acc, _ = evaluate_model(load_fp32(dev), dl, pipe, dev)
        rows.append(("FP32 (BatchNorm)", acc))
    if os.path.isfile(CKPT_QAT):
        qm, _ = load_qat(dev)
        qm.set_quant(False)
        rows.append(("FP32 (BN folded, QAT weights)", evaluate_model(qm, dl, pipe, dev)[0]))
        qm.set_quant(True)
        rows.append((f"QAT fake-quant {PRECISION}", evaluate_model(qm, dl, pipe, dev)[0]))
        q = qm.hw_params()
        s_in, zp_in = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
        xq, y = quantised_split(a.split, pipe, s_in, zp_in, dev)
        acc, conf, _ = accuracy(IntDSCNN(q, dev), xq, y)
        rows.append((f"bit-exact {PRECISION} accelerator", acc))
    print(f"\n{a.split} split accuracy")
    for name, acc in rows:
        print(f"  {name:34s} {acc:6.2f}%")
    if os.path.isfile(CKPT_QAT):
        print(f"\nconfusion matrix (bit-exact {PRECISION}), rows = true class")
        print_confusion(conf)


if __name__ == "__main__":
    main()
