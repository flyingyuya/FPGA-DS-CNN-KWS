#!/usr/bin/env python3
"""Emit the window ROM, the normalisation ROM and end-to-end MFCC golden vectors.

    python3 scripts/golden/gen_mfcc_golden.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "software"))

import mfcc_ref as R                                                   # noqa: E402
from data import AudioDataset                                          # noqa: E402


def main():
    ddir = os.path.join(ROOT, "mem", "preprocessing")
    os.makedirs(ddir, exist_ok=True)

    win = R.window()
    with open(os.path.join(ddir, "hann_q15.mem"), "w") as fh:
        fh.write("".join(f"{int(v) & 0xFFFF:04x}\n" for v in win))
    print(f"  hann window: {len(win)} taps, max {int(win.max())}")

    A, B = R.norm_table()
    with open(os.path.join(ddir, "mfcc_norm.mem"), "w") as fh:
        for c in range(len(A)):
            fh.write(f"{(int(B[c]) & 0xFFFFFFFF):08x}{(int(A[c]) & 0xFFFFFF):06x}\n")
    print(f"  norm table: |A|max {int(np.abs(A).max())} (Q?.20), "
          f"|B|max {int(np.abs(B).max())}")

    gdir = os.path.join(ROOT, "golden", "mfcc")
    os.makedirs(gdir, exist_ok=True)
    ds = AudioDataset("val")
    names = []
    for k, label in enumerate(("silence", "yes", "stop")):
        idx = int(np.nonzero(ds.labels == (0, 2, 10)[k])[0][0])
        x = np.array(ds.audio[idx]).astype(np.int64)
        codes = R.run(x)
        nm = f"{k}_{label}"
        with open(os.path.join(gdir, f"in_{nm}.mem"), "w") as fh:
            fh.write("".join(f"{int(v) & 0xFFFF:04x}\n" for v in x))
        with open(os.path.join(gdir, f"exp_{nm}.mem"), "w") as fh:
            fh.write("".join(f"{int(v):02x}\n" for v in codes))
        names.append(nm)
        print(f"  {nm:10s} codes {codes.min()}..{codes.max()} "
              f"nonzero {100.0 * (codes > 0).mean():.1f}%")
    print(f"  wrote {len(names)} end-to-end vectors to {os.path.relpath(gdir, ROOT)}")


if __name__ == "__main__":
    main()
