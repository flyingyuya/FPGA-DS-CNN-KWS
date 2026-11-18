#!/usr/bin/env python3
"""Generate golden vectors for the 512-point FFT + power block.

The model is scripts/golden/fft_dsp.py's bit-level FFT (butterfly rounding,
per-stage halving, saturation) reading the *same* twiddle ROM the RTL uses, so
the testbench comparison is bit-exact rather than approximate.

    python3 scripts/golden/gen_fft_pow_vectors.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fft_dsp as f                                                    # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "software"))
from features import mel_filterbank                                    # noqa: E402


def mel_integer(pw):
    """Mel energies with the same integer arithmetic the RTL uses (Q1.15 unsigned)."""
    fb = np.round(mel_filterbank()[:, :len(pw)] * (1 << 15)).astype(np.int64)
    out = []
    for c in fb:
        nz = np.nonzero(c)[0]
        # the RTL truncates each product before accumulating (cheap: the mel path
        # has plenty of precision headroom), so the model must do the same
        out.append(int(sum((int(c[k]) * int(pw[k])) >> 15 for k in nz)) if len(nz) else 0)
    return out

N, LOG2N, DW = 512, 9, 24
OUT = os.path.join(ROOT, "golden", "fft")


def q23(v):
    return int(np.clip(np.round(v * (1 << 23)), -(1 << 23), (1 << 23) - 1))


def vectors():
    n = np.arange(N)
    rng = np.random.default_rng(20261120)
    return {
        "impulse":  [q23(0.9 if k == 0 else 0.0) for k in n],
        "sine8":    [q23(0.5 * np.sin(2 * np.pi * 8 * k / N)) for k in n],
        "multitone": [q23(0.3 * np.sin(2 * np.pi * 8 * k / N)
                          + 0.2 * np.sin(2 * np.pi * 71 * k / N)
                          + 0.1 * np.cos(2 * np.pi * 200 * k / N)) for k in n],
        "noise":    [q23(0.2 * rng.standard_normal()) for k in n],
        "max_pos":  [(1 << 23) - 1] * N,
        "max_neg":  [-(1 << 23)] * N,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, x in vectors().items():
        pw = f.fft_core_model(x, log2n=LOG2N, power=True)
        assert len(pw) == N // 2
        assert min(pw) >= 0
        with open(os.path.join(OUT, f"in_{name}.mem"), "w") as fh:
            fh.write("".join(f"{v & 0xFFFFFF:06x}\n" for v in x))
        with open(os.path.join(OUT, f"exp_{name}.mem"), "w") as fh:
            fh.write("".join(f"{v & 0x1FFFFFFFFFFFF:013x}\n" for v in pw))
        mel = mel_integer(pw)
        with open(os.path.join(OUT, f"expmel_{name}.mem"), "w") as fh:
            fh.write("".join(f"{v & 0x3FFFFFFFFFFFFF:014x}\n" for v in mel))
        print(f"  {name:9s} peak bin {int(np.argmax(pw)):3d}  max power {max(pw)/2**46:.5f}")
    print(f"wrote {len(vectors())} vector pairs to {OUT}")


if __name__ == "__main__":
    main()
