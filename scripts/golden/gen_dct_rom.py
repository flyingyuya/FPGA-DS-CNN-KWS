#!/usr/bin/env python3
"""Generate the 40x40 DCT-II coefficient ROM and the golden DCT vectors.

Coefficients: Q1.15 two's complement, one 16-bit word per line, row-major
(address = c*40 + k), so the engine walks one output coefficient at a time.

The RTL accumulates the *unshifted* products (Q1.15 x Q8.8 = Q9.23) in a 40-bit
accumulator and rounds once at the end:

    dct[c] = (sum_k D[c,k]*log[k] + 2^14) >>> 15        (Q9.8)

Doing the shift once at the end (instead of per term, as the mel engine does)
costs a 40-bit accumulator and buys back the truncation noise -- the mel path had
headroom, this one is feeding the classifier's normalisation.

    python3 scripts/golden/gen_dct_rom.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "software"))
from features import dct_matrix                                    # noqa: E402
from config import N_MELS                                          # noqa: E402

Q = 15
SCALE = 1 << Q
N = N_MELS      # square: N_MFCC == N_MELS by construction
F = 8                                   # log is Q8.8


def main():
    D = np.round(dct_matrix() * SCALE).astype(np.int64)
    assert np.abs(D).max() <= 32767, "coefficient does not fit Q1.15"

    out_dir = os.path.join(ROOT, "mem", "preprocessing")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dct_coef.mem")
    with open(path, "w") as fh:
        for c in range(N):
            for k in range(N):
                fh.write(f"{int(D[c, k]) & 0xFFFF:04x}\n")
    print(f"  {N*N} coefficients, |D|max={int(np.abs(D).max())}, "
          f"wrote {os.path.relpath(path, ROOT)}")

    # golden DCT for the log vectors produced by gen_log_lut.py
    gdir = os.path.join(ROOT, "golden", "fft")
    n = 0
    for name in ("impulse", "sine8", "multitone", "noise", "max_pos", "max_neg"):
        src = os.path.join(gdir, f"explog_{name}.mem")
        if not os.path.isfile(src):
            continue
        log = [int(line.strip(), 16) for line in open(src) if line.strip()]
        log = [v - 0x10000 if v >= 0x8000 else v for v in log]      # signed Q8.8
        exp = []
        for c in range(N):
            acc = int(sum(int(D[c, k]) * log[k] for k in range(N)))
            exp.append((acc + (1 << (Q - 1))) >> Q)
        with open(os.path.join(gdir, f"expdct_{name}.mem"), "w") as fh:
            assert max(abs(v) for v in exp) < (1 << 23), "dct does not fit 24 bit"
            fh.write("".join(f"{v & 0xFFFFFF:06x}\n" for v in exp))
        print(f"  {name:9s} dct range {min(exp):6d} .. {max(exp):6d}")
        n += 1
    print(f"  wrote expdct_*.mem for {n} vectors")


if __name__ == "__main__":
    main()
