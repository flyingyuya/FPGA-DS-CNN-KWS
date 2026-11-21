"""Integer reference model for the whole on-chip MFCC chain.

This composes the per-block models that the individual testbenches already verify
(fft_dsp.py's bit-level FFT, the integer mel engine, the 8-bit log table, the DCT
with its single rounded shift) and adds the framing and the per-coefficient
normalisation, so tb_mfcc_top can compare the final 4040-byte map bit-exactly.

Framing matches features.py: the clip is reflect-padded by 256 samples, frame f
covers [f*160 - 256, f*160 + 256), the first 480 taps are windowed and the last 32
are zero.  Address of (coefficient c, frame f) in the output map is c*101 + f.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "software"))

import fft_dsp as F                                                   # noqa: E402
import gen_fft_pow_vectors as GV                                      # noqa: E402
import gen_log_lut as GL                                              # noqa: E402
from features import dct_matrix                                       # noqa: E402
import config as _C                                                   # noqa: E402

# geometry from the training front end, never literals (this file hardcoded
# 40x101 and broke as soon as N_MELS changed)
N_FFT, WIN, HOP = _C.N_FFT, _C.WIN_LENGTH, _C.HOP_LENGTH
NFRAME = _C.N_FRAMES_USED
N_MELS, N_MFCC = _C.N_MELS, _C.N_MFCC
LOG_FRAC, NORM_FRAC = 8, 20
CLIP = 16000

_pw = None


def _lut():
    global _pw
    if _pw is None:
        _pw = GL.lut()
        GL.LUT = _pw
    return _pw


def window():
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN) / WIN)
    return np.round(w * (1 << 15)).astype(np.int64)


def frame_q23(samples, f, win):
    """One windowed frame in Q1.23, exactly as the RTL builds it."""
    x = np.zeros(N_FFT, dtype=np.int64)
    for t in range(WIN):
        i = f * HOP + t - N_FFT // 2
        if i < 0:
            i = -i
        elif i >= CLIP:
            i = 2 * CLIP - 2 - i
        prod = int(samples[i]) * int(win[t])            # Q2.30
        x[t] = (prod + (1 << 6)) >> 7                    # Q1.23, round-half-up
    return [int(v) for v in x]


def norm_table():
    from data import load_feature_stats
    from features import input_qparams
    from config import IN_CLIP_HI, IN_CLIP_LO
    mean, std = load_feature_stats()
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    s, zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
    A = np.round((1.0 / (std * s * (1 << LOG_FRAC))) * (1 << NORM_FRAC)).astype(np.int64)
    B = np.round((zp - mean / (std * s)) * (1 << NORM_FRAC)).astype(np.int64)
    assert np.abs(A).max() < (1 << 23), "A does not fit 24 bit"
    assert np.abs(B).max() < (1 << 31) - 1, "B does not fit 32 bit"
    return A, B


def run(samples):
    """16000 int16 samples -> 4040 UINT8 codes (c*101 + f order)."""
    _lut()
    win = window()
    D = np.round(dct_matrix() * (1 << 15)).astype(np.int64)
    A, B = norm_table()
    out = np.zeros(N_MFCC * NFRAME, dtype=np.uint8)
    for f in range(NFRAME):
        pw = F.fft_core_model(frame_q23(samples, f, win), log2n=9, power=True)
        mel = GV.mel_integer(pw)
        log = [GL.log_q8(v) for v in mel]
        for c in range(N_MFCC):
            acc = int(sum(int(D[c, k]) * log[k] for k in range(N_MELS)))
            dct = (acc + (1 << 14)) >> 15
            code = (int(A[c]) * dct + int(B[c]) + (1 << (NORM_FRAC - 1))) >> NORM_FRAC
            out[c * NFRAME + f] = max(0, min(255, code))
    return out
