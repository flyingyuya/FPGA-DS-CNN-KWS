"""MFCC front-end and input quantiser.

Implemented directly in PyTorch (STFT -> HTK mel filterbank -> log -> DCT-II)
instead of through torchaudio so the exact maths is visible, has no I/O
backend dependency, and can be ported to a host CPU / DSP that feeds the
accelerator.  Output geometry is (40 coefficients x 101 frames), matching
KWS_H_IN x KWS_W_IN in rtl/kws_params.vh.  Memory image address of element
(coefficient y, frame x) is y*101 + x.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import (ACT_QMAX, F_MAX, F_MIN, HOP_LENGTH, IN_CLIP_HI,
                    IN_CLIP_LO, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE, WIN_LENGTH)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sr=SAMPLE_RATE,
                   fmin=F_MIN, fmax=F_MAX):
    """ (n_mels, n_fft//2+1) triangular HTK mel filterbank. """

    n_bins = n_fft // 2 + 1
    fft_hz = np.linspace(0.0, sr / 2.0, n_bins)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (fft_hz - lo) / max(ce - lo, 1e-9)
        dn = (hi - fft_hz) / max(hi - ce, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, dn))
    return fb.astype(np.float32)


def dct_matrix(n_mfcc=N_MFCC, n_mels=N_MELS):
    """ Orthonormal DCT-II, (n_mfcc, n_mels). """

    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    d = np.cos(math.pi / n_mels * (n + 0.5) * k) * math.sqrt(2.0 / n_mels)
    d[0] *= 1.0 / math.sqrt(2.0)
    return d.astype(np.float32)


class MFCC(nn.Module):
    """ Batched MFCC: (B, 16000) float waveform in [-1, 1] -> (B, 40, 101). """

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))
        self.register_buffer("dct", torch.from_numpy(dct_matrix()))

    def forward(self, wav):
        spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH, window=self.window,
                          center=True, pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2              # (B, 257, 101)
        mel = torch.matmul(self.fb, power)                   # (B, 40, 101)
        logmel = torch.log(mel + 1e-6)
        return torch.matmul(self.dct, logmel)                # (B, 40, 101)


# -----------------------------------------------------------------------------
# Normalisation + UINT8 input quantisation
# -----------------------------------------------------------------------------
def input_qparams(lo=IN_CLIP_LO, hi=IN_CLIP_HI):
    """ Asymmetric UINT8 quantiser for the normalised MFCC map.  Same formula as
    asym() in sim/reference_model.py, so the exporter reproduces it exactly.
    """

    lo = min(lo, 0.0)
    s = (hi - lo) / ACT_QMAX
    zp = int(np.clip(round(-lo / s), 0, ACT_QMAX))
    return s, zp


def normalise(mfcc, mean, std):
    """ Per-coefficient standardisation, then clip to the quantiser range.
    mean/std are (40,) tensors broadcast over frames.
    """

    x = (mfcc - mean[:, None]) / std[:, None]
    return x.clamp(IN_CLIP_LO, IN_CLIP_HI)


def quantise_input(x, s, zp):
    """ Normalised float map -> UINT8 codes (what the accelerator stores). """

    if isinstance(x, torch.Tensor):
        return torch.clamp(torch.round(x / s) + zp, 0, ACT_QMAX).to(torch.uint8)
    return np.clip(np.round(x / s) + zp, 0, ACT_QMAX).astype(np.uint8)


def dequantise_input(xq, s, zp):
    if isinstance(xq, torch.Tensor):
        return (xq.float() - zp) * s
    return (xq.astype(np.float64) - zp) * s
