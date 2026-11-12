"""Quantisation-aware training model that mirrors the accelerator numerics.

QDSCNN works on the BN-folded network and inserts fake quantisers exactly
where the hardware quantises:

  input    UINT8 asymmetric, fixed (scale, zp) from features.input_qparams()
  weights  signed W_BITS, symmetric per output channel,
           s_w[oc] = max|w[oc]| / W_QMAX              (W_QMAX = 7 or 127)
  bias     INT32 in the accumulator domain, step s_w[oc] * s_in
  acts     UINT8 after every conv + ReLU, zp = 0, s = hi / 255 per tensor
  gap      integer mean of UINT8 codes (keeps the pw4 scale)
  logits   signed INT16 with LSB = 1/16 natural-log unit (softmax LUT domain)

Rounding uses a straight-through estimator.  The remaining hardware effects
(Q16.16 multiplier rounding, round-half-up instead of half-to-even) are below
one LSB and are only modelled by the bit-exact int_model.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (ACT_EMA, ACT_PERCENTILE, ACT_QMAX, CALIB_BATCHES,
                    IN_CLIP_HI, IN_CLIP_LO, W_QMAX)
from features import input_qparams
from model import CONV_NAMES, FC_SPEC, LAYER_SPEC

LOGIT_FRAC = 16
LOGIT_MIN, LOGIT_MAX = -32768, 32767


def round_ste(x):
    """Round in the forward pass, identity gradient in the backward pass."""
    return (torch.round(x) - x).detach() + x


def weight_scale(w):
    """Per-output-channel symmetric scale, detached (it is not trained)."""
    amax = w.detach().abs().reshape(w.shape[0], -1).amax(dim=1).clamp_min(1e-12)
    return amax / W_QMAX


def fake_quant_weight(w):
    s = weight_scale(w)
    shape = (-1,) + (1,) * (w.dim() - 1)
    q = torch.clamp(round_ste(w / s.view(shape)), -W_QMAX, W_QMAX)
    return q * s.view(shape), s


def fake_quant_bias(b, s_w, s_in):
    step = (s_w * s_in).detach()
    return round_ste(b / step) * step


class ActQuant(nn.Module):
    """Unsigned 8-bit post-ReLU activation quantiser with an EMA range."""

    def __init__(self):
        super().__init__()
        self.register_buffer("hi", torch.tensor(1.0))
        self.register_buffer("initialised", torch.tensor(False))
        self.observe = False      # update the range from live batches
        self.enabled = True

    def scale(self):
        return self.hi.clamp_min(1e-9) / ACT_QMAX

    def forward(self, x):
        if self.observe and self.training:
            with torch.no_grad():
                m = x.detach().amax()
                if not bool(self.initialised):
                    self.hi.copy_(m)
                    self.initialised.fill_(True)
                else:
                    self.hi.mul_(ACT_EMA).add_((1 - ACT_EMA) * m)
        if not self.enabled:
            return x
        s = self.scale()
        return torch.clamp(round_ste(x / s), 0, ACT_QMAX) * s


class QDSCNN(nn.Module):
    """BN-folded DS-CNN with hardware-faithful fake quantisation."""

    def __init__(self, folded):
        super().__init__()
        self.kinds = {n: k for n, k, *_ in LAYER_SPEC}
        self.strides = {n: s for n, _, _, _, s in LAYER_SPEC}
        self.groups = {n: (ci if k == "dw" else 1) for n, k, ci, _, _ in LAYER_SPEC}
        self.w = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for n in CONV_NAMES + [FC_SPEC[0]]:
            w, b = folded[n]
            self.w[n] = nn.Parameter(w.clone().float())
            self.b[n] = nn.Parameter(b.clone().float())
        self.aq = nn.ModuleDict({n: ActQuant() for n in CONV_NAMES})
        self.in_s, self.in_zp = input_qparams(IN_CLIP_LO, IN_CLIP_HI)
        self.quant = True

    # ---- range control -------------------------------------------------------
    def set_observe(self, flag):
        for q in self.aq.values():
            q.observe = flag

    def set_quant(self, flag):
        self.quant = flag
        for q in self.aq.values():
            q.enabled = flag

    @torch.no_grad()
    def calibrate(self, batches):
        """Initialise every activation range from a high percentile of the
        float activations (robust to single outliers), then hand over to EMA."""
        was_quant = self.quant
        self.set_quant(False)
        self.eval()
        acts = {n: [] for n in CONV_NAMES}
        hooks = [self.aq[n].register_forward_hook(
            lambda m, i, o, n=n: acts[n].append(
                i[0].flatten()[torch.randperm(i[0].numel(), device=i[0].device)[:200000]]))
            for n in CONV_NAMES]
        for i, x in enumerate(batches):
            if i >= CALIB_BATCHES:
                break
            self(x)
        for h in hooks:
            h.remove()
        for n in CONV_NAMES:
            v = torch.cat(acts[n]).float()
            hi = torch.quantile(v[:2_000_000].cpu(), ACT_PERCENTILE / 100.0)
            self.aq[n].hi.fill_(max(float(hi), 1e-6))
            self.aq[n].initialised.fill_(True)
        self.set_quant(was_quant)

    # ---- forward -------------------------------------------------------------
    def forward(self, x):
        if self.quant:                       # input: UINT8 codes, dequantised
            s, zp = self.in_s, self.in_zp
            x = (torch.clamp(round_ste(x / s) + zp, 0, ACT_QMAX) - zp) * s
        s_in = self.in_s
        for n in CONV_NAMES:
            w, b = self.w[n], self.b[n]
            if self.quant:
                w, s_w = fake_quant_weight(w)
                b = fake_quant_bias(b, s_w, s_in)
            pad = 0 if self.kinds[n] == "pw" else 1
            x = F.conv2d(x, w, b, self.strides[n], pad, groups=self.groups[n])
            x = self.aq[n](F.relu(x))
            s_in = self.aq[n].scale()
        x = x.mean(dim=(2, 3))
        if self.quant:                       # GAP: integer mean in code domain
            x = round_ste(x / s_in) * s_in
        w, b = self.w["fc"], self.b["fc"]
        if self.quant:
            w, s_w = fake_quant_weight(w)
            b = fake_quant_bias(b, s_w, s_in)
        y = F.linear(x, w, b)
        if self.quant:                       # INT16 logits, LSB = 1/16
            y = torch.clamp(round_ste(y * LOGIT_FRAC), LOGIT_MIN, LOGIT_MAX) / LOGIT_FRAC
        return y

    # ---- export --------------------------------------------------------------
    @torch.no_grad()
    def act_ranges(self):
        r = {"input": (float(IN_CLIP_LO), float(IN_CLIP_HI))}
        for n in CONV_NAMES:
            r[n] = (0.0, float(self.aq[n].hi))
        return r

    @torch.no_grad()
    def float_params(self):
        """FP32 BN-folded parameters in the shapes tb/reference_model.py uses:
        std/dw (cout, cin|1, 3, 3), pw/fc (cout, cin)."""
        prm = {}
        for n in CONV_NAMES + ["fc"]:
            w = self.w[n].detach().cpu().double().numpy()
            if self.kinds.get(n) == "pw":
                w = w[:, :, 0, 0]
            prm[n + ".w"] = np.ascontiguousarray(w)
            prm[n + ".b"] = self.b[n].detach().cpu().double().numpy()
        return prm
