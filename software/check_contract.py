"""does the shift-only numeric contract fit the hardware,
and what does it cost in accuracy?

  uv run python check_contract.py                    # full val split
  uv run python check_contract.py --batches 8        # smoke

It verifies, layer by layer, the four things the RTL depends on
(docs/quantisation-math.md sec. 1, 2, 7):

  1. every per-channel shift is inside the 4-bit field [0, 15]
  2. no weight code was clipped (|q_w| <= 127) -- i.e. the step is never too fine
  3. the worst-case accumulator fits the 24-bit accumulator
  4. weight_step * s_in / s_out is an exact power of two (the requantisation lemma)

and then reports the accuracy of the new contract next to the Q16.16 legacy one
that the reference RTL used.
"""

import argparse

import numpy as np
import torch

from config import ACC_BITS, SHIFT_MAX
from data import AudioDataset, build_pipeline
from train import load_qat, pick_device


@torch.no_grad()
def evaluate(forward, pipe, split, nb, device):
    ds = AudioDataset(split)
    n = len(ds) if nb == 0 else min(len(ds), nb * 128)
    pred, gts = [], []
    for i in range(0, n, 128):
        wav = torch.from_numpy(np.array(ds.audio[i:i + 128])).to(device)
        pred.append(forward(pipe(wav)).argmax(1).cpu())
        gts.append(torch.from_numpy(ds.labels[i:i + 128]))
    pred, gt = torch.cat(pred), torch.cat(gts)
    return 100.0 * (pred == gt).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dev = pick_device("")
    torch.set_grad_enabled(False)

    qm, ck = load_qat(dev)
    print(f"QAT checkpoint: epoch {ck['epoch']}, fake-quant val {ck['val_acc']:.2f}%")

    rows = qm.check_contract()          # raises if anything violates the contract
    print("\n-- hardware contract (shift-only requantisation) --")
    print(f"{'layer':6s}{'shift':>8s}{'max|q_w|':>10s}{'|B|max':>9s}"
          f"{'acc_max':>10s}{'bits':>6s}{'s_in':>10s}{'s_out':>10s}")
    for r in rows:
        print(f"{r['name']:6s}{r['shift'][0]:>4d}..{r['shift'][1]:<3d}"
              f"{r['max_abs']:>10d}{r['bias_max']:>9d}{r['acc_max']:>10.0f}"
              f"{r['acc_bits']:>4d}/{ACC_BITS}{r['s_in']:>10.5f}{r['s_out']:>10.5f}")
    print(f"  shift field 4 bit [0,{SHIFT_MAX}]        OK")
    print(f"  weight codes <= 127 (no clipping)     OK")
    print(f"  accumulator <= {ACC_BITS} bit              OK")
    print(f"  requant ratio is an exact 2**-shift   OK")

    pipe = build_pipeline(dev).eval()
    acc_new = evaluate(qm, pipe, a.split, a.batches, dev)
    acc_leg = evaluate(qm.forward_legacy, pipe, a.split, a.batches, dev)
    print(f"\n-- {a.split} split accuracy --")
    print(f"  legacy Q16.16 multiplier (reference RTL) : {acc_leg:.2f}%")
    print(f"  shift-only + 24-bit acc  (this design)   : {acc_new:.2f}%"
          f"   delta {acc_new - acc_leg:+.2f}")


main()
