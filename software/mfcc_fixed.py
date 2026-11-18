"""Bit-true fixed-point model of the on-chip MFCC front end.

The accelerator was trained on float MFCCs (features.py).  Before writing RTL we
need to know what the fixed-point formats cost, because the answer decides how
many bits each stage gets.

Chain modelled here (each arrow is an explicit rounding point):

    int16 PCM ──► Hann window Q1.15 ──► x·w in Q2.30, >>7 ──► Q1.23
        ──► 512-point FFT (radix-2 DIT, twiddle Q1.15, ÷2 per stage)
        ──► re,im in Q1.23 ──► power = re²+im²  (Q2.46, int64)
        ──► mel filterbank, coefficients Q1.15 ──► mel energy (Q2.46)
        ──► log: exponent + 12-bit mantissa LUT, Q8.8
        ──► DCT-II 40x40, coefficients Q1.15 ──► Q8.8
        ──► per-coefficient affine normalisation + UINT8 quantisation

The FFT is taken from numpy rather than re-implemented: the RTL core this project
will use (Sonic-Speed-Parallelism rtl/fft/fft_core.v) is already verified bit-exact
against numpy for N=1024, and its per-stage truncation (0.5 LSB x 9 stages) sits
below the dominant error terms modelled here.  Once the RTL exists, this file
becomes the golden model and the per-stage arithmetic must be mirrored exactly.

  uv run python mfcc_fixed.py --batches 4      # quick signal
  uv run python mfcc_fixed.py                  # full val split
"""

import argparse
import math

import numpy as np
import torch

from config import (CLIP_SAMPLES, F_MAX, F_MIN, HOP_LENGTH, IN_CLIP_HI,
                    IN_CLIP_LO, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE,
                    WIN_LENGTH)
from data import AudioDataset, build_pipeline
from features import input_qparams, mel_filterbank, dct_matrix
from int_model import IntDSCNN
from train import load_qat

# --- fixed-point formats -----------------------------------------------------
WIN_FRAC = 15          # Hann window, Q1.15
DATA_FRAC = 23         # FFT data, Q1.23 (matches fft_core's DW=24)
TW_FRAC = 15           # twiddle factors, Q1.15
MEL_FRAC = 15          # mel filterbank coefficients, Q1.15
DCT_FRAC = 15          # DCT-II matrix, Q1.15
LOG_FRAC = 8           # log output, Q8.8
LOG_LUT_BITS = 8      # mantissa resolution used by the log lookup


def hann_q15(n=WIN_LENGTH):
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)     # periodic (torch default)
    q = np.round(w * (1 << WIN_FRAC)).astype(np.int64)
    return np.clip(q, 0, (1 << WIN_FRAC) - 1)


def frames_q23(samples):
    """(B, 16000) int16 -> (B, 101, 512) int64 windowed frames in Q1.23.

    Mirrors torch.stft(center=True, reflect padding): the clip is reflect-padded
    by N_FFT//2 on both sides, then frame i covers [i*hop, i*hop+512) with only
    the first WIN_LENGTH samples windowed (the rest is the zero pad).
    """
    B, N = samples.shape
    pad = N_FFT // 2
    # np.pad(mode="reflect") is exactly torch's reflect padding; doing the index
    # arithmetic by hand is off by one on the right edge and quietly corrupts the
    # last frames.
    padded = np.pad(samples, ((0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + N // HOP_LENGTH
    start = np.arange(n_frames) * HOP_LENGTH
    out = np.zeros((B, n_frames, N_FFT), dtype=np.int64)
    win = hann_q15()
    for k in range(n_frames):
        seg = padded[:, start[k]:start[k] + WIN_LENGTH].astype(np.int64)
        prod = seg * win                                        # Q2.30
        # Q1.15 x Q1.15 = Q2.30; Q1.23 keeps 23 fractional bits -> drop 7
        out[:, k, :WIN_LENGTH] = np.round(prod / (1 << (2 * WIN_FRAC - DATA_FRAC)))
    return out


def fft_q23(frames):
    """Real FFT with the hardware's ÷2-per-stage scaling, output quantised."""
    B, nf, n = frames.shape
    # frames are ALREADY integers in Q1.23, so their FFT is already scaled by
    # 2**23 -- multiplying by 2**23 again saturates the whole spectrum.
    # The hardware divides by 2 per stage (N = 512 -> 9 stages).
    scaled = frames.astype(np.float64) / (1 << int(math.log2(n)))
    spec = np.fft.rfft(scaled, axis=2)                          # (B, nf, 257)
    re = np.round(spec.real).astype(np.int64)
    im = np.round(spec.imag).astype(np.int64)
    lim = (1 << (DATA_FRAC)) - 1
    return np.clip(re, -lim - 1, lim), np.clip(im, -lim - 1, lim)


def power_mel(frames, ideal_fft=False, ideal_mel=False):
    """Q1.23 spectrum -> (B, nf, 40) mel energies in Q2.46 (int64)."""
    if ideal_fft:
        sp = np.fft.rfft(frames.astype(np.float64) / 512.0 ** 2, axis=2)
        power = np.round(sp.real ** 2 + sp.imag ** 2).astype(np.int64) * 0 + 0
        power = np.round((sp.real ** 2 + sp.imag ** 2) * (1 << 46)).astype(np.int64)
    else:
        re, im = fft_q23(frames)
        power = re * re + im * im                               # Q2.46
    fb = mel_filterbank()
    fbq = (fb * (1 << MEL_FRAC) if ideal_mel
           else np.round(fb * (1 << MEL_FRAC))).astype(np.int64)   # (40, 257) Q1.15
    # (B,nf,257) x (257,40) in integer arithmetic, >> 15 keeps Q2.46
    mel = np.einsum("bfp,kp->bfk", power.astype(np.float64), fbq.astype(np.float64))
    return np.round(mel / (1 << MEL_FRAC)).astype(np.int64)


def log_q8(mel, eps_shift=30, fft_scale_bits=18, ideal=False):
    """ln(mel + eps) with an exponent + 12-bit mantissa LUT, output Q8.8.

    Two scaling facts have to be undone here:

    * the FFT divides by N (fft_core halves once per stage: 9 stages -> /512), so
      the hardware mel energy is the software one / 512**2 = /2**18.  That rescales
      the +1e-6 epsilon too, and getting it wrong made the epsilon dominate every
      bin (every log came out as ln(1e-6)).
    * the mel values are Q2.46 and get shifted right by `eps_shift` so the integer
      epsilon stays inside int64.

    Both corrections are a constant: hardware pays for them with one constant in
    the exponent adder, not with extra hardware.
    """
    v = mel / float(1 << eps_shift)
    eps = 1e-6 * (1 << (46 - eps_shift)) / float(1 << fft_scale_bits)
    v = np.maximum(v, 0) + eps
    e = np.floor(np.log2(v)).astype(np.int64)
    m = v / np.power(2.0, e) - 1.0                              # [0, 1)
    step = 1.0 / (1 << LOG_LUT_BITS)
    q = np.round(m * (1 << LOG_LUT_BITS)).astype(np.int64)       # LUT index
    ln_mant = np.log1p(q * step)                                # what the LUT holds
    # v carries a factor 2**(46-eps_shift) relative to the float mel value, so the
    # log must give back that constant.  In hardware it is a constant added in the
    # exponent path -- free -- but forgetting it biases c0 by 11 and wrecks the
    # classifier, so it is spelled out here.
    ln = (e * math.log(2.0) + ln_mant
          - (46 - eps_shift) * math.log(2.0) + fft_scale_bits * math.log(2.0))
    if ideal:
        sw = mel.astype(np.float64) / (1 << 46) * (1 << fft_scale_bits) + 1e-6
        return np.round(np.log(sw) * (1 << LOG_FRAC)).astype(np.int64)
    return np.round(ln * (1 << LOG_FRAC)).astype(np.int64)


def dct_q8(logmel, ideal=False):
    """40x40 DCT-II with Q1.15 coefficients -> Q8.8."""
    d = ((dct_matrix() * (1 << DCT_FRAC)) if ideal
         else np.round(dct_matrix() * (1 << DCT_FRAC))).astype(np.int64)
    acc = np.einsum("ck,bfk->bfc", d.astype(np.float64), logmel.astype(np.float64))
    return np.round(acc / (1 << DCT_FRAC)).astype(np.int64)


def quantise(mfcc_q8, mean, std, s, zp):
    """Per-coefficient normalisation + UINT8 quantisation, exactly as features.py."""
    f = mfcc_q8 / float(1 << LOG_FRAC)
    x = (f - mean[None, :, None]) / std[None, :, None]
    x = np.clip(x, IN_CLIP_LO, IN_CLIP_HI)
    code = np.round(x / s) + zp
    return np.clip(code, 0, 255).astype(np.uint8)


@torch.no_grad()
def run(split, nb, device, ideal=""):
    qm, ck = load_qat("cpu")
    hw = qm.hw_params()
    model = IntDSCNN(hw, device)
    mean, std = None, None
    from data import load_feature_stats
    mean, std = load_feature_stats()
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    s, zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)

    ds = AudioDataset(split)
    n = len(ds) if nb == 0 else min(len(ds), nb * 128)
    preds, gts = [], []
    for i in range(0, n, 128):
        wav = np.array(ds.audio[i:i + 128]).astype(np.int64)     # int16
        fr = frames_q23(wav)
        mel = power_mel(fr, ideal_fft="fft" in ideal, ideal_mel="mel" in ideal)
        lm = log_q8(mel, ideal="log" in ideal)
        mf = np.transpose(dct_q8(lm, ideal="dct" in ideal), (0, 2, 1))      # (B, frames, coeff) -> (B, coeff, frames)
        xq = quantise(mf, mean, std, s, zp)[:, None]
        preds.append(model(torch.from_numpy(xq).float()).argmax(1).cpu())
        gts.append(torch.from_numpy(ds.labels[i:i + len(wav)]))
        print(f"  {min(i + 128, n):5d}/{n}", end="\r", flush=True)
    pred, gt = torch.cat(preds), torch.cat(gts)
    print()
    return 100.0 * (pred == gt).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ideal", default="", help="comma list: fft,mel,log,dct")
    a = ap.parse_args()
    dev = torch.device("cpu")
    acc = run(a.split, a.batches, dev, a.ideal)
    print(f"ideal[{a.ideal or '-'}] -> fixed-point MFCC + int8: {acc:.2f}% on {a.split}")


if __name__ == "__main__":
    main()
