"""Quantisation-aware training model that mirrors the accelerator numerics.

-----------------------------------------------------------------------------
What changed, and why (see docs/quantisation-math.md for the full derivation)
-----------------------------------------------------------------------------
The accelerator this project targets has **no requantisation multiplier**.
Every layer produces its 8-bit output with a per-channel arithmetic right
shift, so the hardware is exactly

    A_oc  = bias[oc] + sum_t w_code[oc,t] * act_code[t]        (24-bit signed)
    code  = clamp( A_oc >>> shift[oc], 0, 255 )

For that to equal "the real output quantised with scale s_out" we need

    weight_step[oc] * s_in / s_out == 2 ** -shift[oc]                    (T1)

which is satisfied by

    weight_step[oc] = rho * 2 ** -shift[oc],   rho = s_out / s_in         (T2)

so a layer has ONE free real constant (rho) and its channels may only differ by
powers of two.  Two further tricks remove hardware:

  * the input zero point is folded into the bias (`- zp_in * sum(w_code)`), which
    is why the MAC has no subtractor and the multiplier is 8x8 instead of 8x9;
  * the rounding constant 2**(shift-1) is folded into the bias too, which turns
    the truncating shift into round-half-up for free.

Only the weight *steps* are restricted; the activation scale stays exact
(s_out = hi/255), so the 8-bit activation range is fully used.

-----------------------------------------------------------------------------
Q16.16 legacy path (kept for comparison)
-----------------------------------------------------------------------------
`forward_legacy()` replays the original contract of the reference RTL: a
per-channel Q16.16 multiplier, round-half-up, INT16 logits and softmax-friendly
outputs.  It is never used for training or export; it exists so the accuracy
cost of the shift-only contract can be measured side by side.  The helper
functions it needs are prefixed `*_legacy`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (ACC_BITS, ACT_EMA, ACT_PERCENTILE, ACT_QMAX, CALIB_BATCHES,
                    GAP_SHIFT, H_IN, IN_CLIP_HI, IN_CLIP_LO, LOGIT_SCALE,
                    SHIFT_MAX, W_QMAX, W_IN)
from features import input_qparams
from model import CONV_NAMES, FC_SPEC, LAYER_SPEC

# legacy Q16.16 logit domain, kept only for forward_legacy()
LOGIT_FRAC = 16
LOGIT_MIN, LOGIT_MAX = -32768, 32767


# =============================================================================
# Rounding with a straight-through gradient
# =============================================================================
def round_ste(x):
    """Round in the forward pass, identity gradient in the backward pass."""
    return (torch.round(x) - x).detach() + x


def floor_ste(x):
    """floor in the forward pass (== an arithmetic right shift), identity gradient.

    Truncation carries a systematic -0.5 LSB bias, which is exactly why it has
    to appear during training: the network then learns to lift its own bias by
    half a step instead of us patching it afterwards.
    """
    return (torch.floor(x) - x).detach() + x


# =============================================================================
# Geometric constant: how many pixels the global average pool has to average
# =============================================================================
def gap_shape():
    """(H, W) of the tensor entering GAP, walked through LAYER_SPEC."""
    h, w = H_IN, W_IN
    for _, kind, _, _, stride in LAYER_SPEC:
        if kind != "pw":
            h, w = (h + 2 - 3) // stride + 1, (w + 2 - 3) // stride + 1
    return h, w


GAP_H, GAP_W = gap_shape()
GAP_N = GAP_H * GAP_W               # 5 * 13 = 65


# =============================================================================
# LEGACY: exact per-channel scale + Q16.16 requantisation multiplier
# -----------------------------------------------------------------------------
# Kept verbatim from the reference implementation so the two contracts can be
# compared line by line.  Do not use for training or export.
# =============================================================================
def weight_scale_legacy(w):
    """Per-output-channel symmetric scale, detached (it is not trained)."""
    amax = w.detach().abs().reshape(w.shape[0], -1).amax(dim=1).clamp_min(1e-12)
    return amax / W_QMAX


def fake_quant_weight_legacy(w):
    s = weight_scale_legacy(w)
    shape = (-1,) + (1,) * (w.dim() - 1)
    q = torch.clamp(round_ste(w / s.view(shape)), -W_QMAX, W_QMAX)
    return q * s.view(shape), s


def fake_quant_bias_legacy(b, s_w, s_in):
    step = (s_w * s_in).detach()
    return round_ste(b / step) * step


# =============================================================================
# NEW: the accelerator's contract
# =============================================================================
def weight_step(w, s_in, s_out):
    """Per-channel weight step delta_oc = rho * 2**-shift_oc  (eq. T2).

    rho is the layer's single free real constant.  The shift is chosen with
    floor(), i.e. the step is rounded *up* towards the ideal M/127, so that
    127 * delta always covers max|W| and no weight ever clips.  Rounding to the
    nearest power of two instead clips the largest weights and cost 1.6 points
    of accuracy in measurement -- see docs/quantisation-math.md sec. 2.
    """
    rho = (s_out / s_in).detach()
    ideal = w.detach().abs().reshape(w.shape[0], -1).amax(dim=1).clamp_min(1e-12) / W_QMAX
    shift = torch.floor(torch.log2(rho / ideal)).clamp(0, SHIFT_MAX)
    return rho * torch.pow(2.0, -shift), shift


def fake_quant_weight(w, s_in, s_out):
    """Quantise one layer's weights.  Returns (dequantised, codes, step, shift)."""
    step, shift = weight_step(w, s_in, s_out)
    shape = (-1,) + (1,) * (w.dim() - 1)
    q = torch.clamp(round_ste(w / step.view(shape)), -W_QMAX, W_QMAX)
    return q * step.view(shape), q, step, shift


def fake_quant_bias(b, q, step, s_in, zp_in, shift):
    """Bias, in both domains it is needed in.

    The hardware ROM holds (`B_rom`, integer accumulator domain):

        B_rom[oc] = round(b[oc] / (step*s_in))          bias quantisation
                    - zp_in * sum_t q[oc,t]             zero-point fold
                    + 2**(shift[oc]-1)                  round-half-up

    and the float forward adds `b_deq = round(b/(step*s_in)) * step*s_in`
    (bias quantisation only).  The two are linked by an identity that is worth
    stating once, because it is what keeps training and RTL in sync:

        activations entering the layer are dequantised WITH the zero point
        removed, and out-of-range taps contribute exactly 0.  Then

            sum_t q_w*code + B_rom  =  y/(step*s_in) + 2**(shift-1)

        so   floor( (...)/2**shift )  ==  floor( y/s_out + 1/2 )

    i.e. the float forward must requantise with round-half-up and needs no
    knowledge of the zero point at all.  Equivalently: the ROM bias is only
    exact when the RTL feeds `zp_in` for out-of-range taps (sec. 3 of the math
    notes); a float `F.pad(..., 0)` on already-dequantised activations is the
    same thing.

    Returns (bias for the float forward, integer ROM content).
    """
    b_q = round_ste(b / (step * s_in))
    b_deq = b_q * (step * s_in)
    B_rom = b_q + torch.pow(2.0, (shift - 1).clamp_min(0))
    if zp_in:
        B_rom = B_rom - zp_in * q.reshape(q.shape[0], -1).sum(dim=1)
    return b_deq, B_rom


# =============================================================================
# Activation quantiser
# =============================================================================
class ActQuant(nn.Module):
    """Unsigned 8-bit post-ReLU activation quantiser with an EMA range.

    scale() = hi/255 is already the exact s_out required by eq. T1: the
    activation scale must NOT be restricted to a power of two, only the weight
    steps are.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("hi", torch.tensor(1.0))
        self.register_buffer("initialised", torch.tensor(False))
        self.observe = False      # update the range from live batches
        self.enabled = True
        # True  : floor(v/s + 1/2) -- matches the 2**(shift-1) folded into B_rom
        # False : round_ste        -- the legacy Q16.16 contract, used by
        #                             forward_legacy() for the comparison only
        self.hw_round = True

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
        if self.hw_round:
            return torch.clamp(floor_ste(x / s + 0.5), 0, ACT_QMAX) * s
        return torch.clamp(round_ste(x / s), 0, ACT_QMAX) * s      # legacy


# =============================================================================
# Network
# =============================================================================
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
        """Hardware-faithful forward: shift-only requantisation."""
        if self.quant:                       # input: UINT8 codes, dequantised
            s, zp = self.in_s, self.in_zp
            x = (torch.clamp(round_ste(x / s) + zp, 0, ACT_QMAX) - zp) * s
        s_in, zp_in = self.in_s, self.in_zp
        for n in CONV_NAMES:
            w, b = self.w[n], self.b[n]
            s_out = self.aq[n].scale()       # this layer's output scale = next s_in
            if self.quant:
                w, q, step, shift = fake_quant_weight(w, s_in, s_out)
                b, _ = fake_quant_bias(b, q, step, s_in, zp_in, shift)
            pad = 0 if self.kinds[n] == "pw" else 1
            x = F.conv2d(x, w, b, self.strides[n], pad, groups=self.groups[n])
            x = self.aq[n](F.relu(x))
            s_in, zp_in = s_out, 0
        if self.quant:
            # GAP in the code domain: integer sum, round-half-up, >> GAP_SHIFT.
            # 1/N is NOT a power of two, but 1/2**GAP_SHIFT is, and the residual
            # 2**GAP_SHIFT/N factor is carried by s_gap into the FC layer.
            codes = torch.round(x / s_in)
            gap = torch.clamp(torch.floor(
                (codes.sum(dim=(2, 3), keepdim=True) + 2 ** (GAP_SHIFT - 1))
                / 2 ** GAP_SHIFT), 0, ACT_QMAX)
            s_in = s_in * 2 ** GAP_SHIFT / GAP_N          # s_gap = s_in * 64/65
            x = gap * s_in
        else:
            x = x.mean(dim=(2, 3))
        w, b = self.w["fc"], self.b["fc"]
        if self.quant:
            w, q, step, shift = fake_quant_weight(w, s_in, LOGIT_SCALE)
            b, _ = fake_quant_bias(b, q, step, s_in, 0, shift)
        y = F.linear(x.flatten(1), w, b)
        if not self.quant:
            return y
        # Multiplying the code back by LOGIT_SCALE keeps the value in natural units, 
        # exactly like forward_legacy() does with LOGIT_FRAC.
        return torch.floor(y / LOGIT_SCALE + 0.5) * LOGIT_SCALE

    def forward_legacy(self, x):
        """Reference-RTL contract: Q16.16 multiplier + INT16 logits.

        Not used for training or export.  Call it to measure the accuracy that
        the shift-only contract costs (see docs/quantisation-math.md sec. 8).
        """
        for q in self.aq.values():          # plain round(), as the legacy RTL does
            q.hw_round = False
        try:
            return self._forward_legacy(x)
        finally:
            for q in self.aq.values():
                q.hw_round = True

    def _forward_legacy(self, x):
        if self.quant:
            s, zp = self.in_s, self.in_zp
            x = (torch.clamp(round_ste(x / s) + zp, 0, ACT_QMAX) - zp) * s
        s_in = self.in_s
        for n in CONV_NAMES:
            w, b = self.w[n], self.b[n]
            if self.quant:
                w, s_w = fake_quant_weight_legacy(w)
                b = fake_quant_bias_legacy(b, s_w, s_in)
            pad = 0 if self.kinds[n] == "pw" else 1
            x = F.conv2d(x, w, b, self.strides[n], pad, groups=self.groups[n])
            x = self.aq[n](F.relu(x))
            s_in = self.aq[n].scale()
        x = x.mean(dim=(2, 3))
        if self.quant:
            x = round_ste(x / s_in) * s_in
        w, b = self.w["fc"], self.b["fc"]
        if self.quant:
            w, s_w = fake_quant_weight_legacy(w)
            b = fake_quant_bias_legacy(b, s_w, s_in)
        y = F.linear(x, w, b)
        if self.quant:
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

    @torch.no_grad()
    def hw_params(self):
        """Every constant the accelerator needs, in execution order.

        This is the single interface between the ML pipeline and the RTL:
        int_model.py, export_mem.py and tb/reference_model.py must read the
        network ONLY through this dict, so all three are bit-identical.

            q     weight codes, W_QMAX-scaled integers in [-127, 127]
            B     bias ROM content, 24-bit signed accumulator domain
                  (holds the zero-point fold and the 2**(shift-1) rounding)
            shift per-channel right shift, 4-bit field
            step  weight step (float; the hardware does not need it)
            s_in  activation scale entering the layer
            s_out activation scale leaving the layer
        """
        out, s_in, zp_in = {}, self.in_s, self.in_zp
        for n in CONV_NAMES:
            s_out = self.aq[n].scale()
            _, q, step, shift = fake_quant_weight(self.w[n], s_in, s_out)
            _, B = fake_quant_bias(self.b[n], q, step, s_in, zp_in, shift)
            if self.kinds.get(n) == "pw":
                q = q.reshape(q.shape[0], q.shape[1])
            out[n] = dict(q=q.cpu().numpy(), B=B.cpu().numpy(),
                          shift=shift.cpu().numpy().astype(np.int64),
                          step=step.cpu().numpy(), s_in=float(s_in),
                          s_out=float(s_out), zp_in=int(zp_in))
            s_in, zp_in = s_out, 0
        # global average pool: sum >>> GAP_SHIFT, s_gap = s_in * 2**G / N
        s_gap = s_in * 2 ** GAP_SHIFT / GAP_N
        out["gap"] = dict(N=GAP_N, shift=GAP_SHIFT, s_in=float(s_in),
                          s_out=float(s_gap), zp_in=0)
        _, q, step, shift = fake_quant_weight(self.w["fc"], s_gap, LOGIT_SCALE)
        _, B = fake_quant_bias(self.b["fc"], q, step, s_gap, 0, shift)
        out["fc"] = dict(q=q.cpu().numpy(), B=B.cpu().numpy(),
                         shift=shift.cpu().numpy().astype(np.int64),
                         step=step.cpu().numpy(), s_in=float(s_gap),
                         s_out=float(LOGIT_SCALE), zp_in=0)
        return out

    @torch.no_grad()
    def check_contract(self):
        """Assert the exported constants fit the hardware, return a report.

        Raises AssertionError on any violation, so export_mem.py fails loudly
        instead of silently producing ROM images that overflow.
        """
        hw = self.hw_params()
        rows, bad = [], []
        for n in CONV_NAMES + ["fc"]:
            h = hw[n]
            q, B, shift = h["q"], h["B"], h["shift"]
            max_abs = int(np.abs(q).max())
            bias_max = int(np.abs(B).max())
            # worst-case accumulator: every activation code at 255
            acc_max = float(np.abs(q).reshape(q.shape[0], -1).sum(axis=1).max() * ACT_QMAX
                            + bias_max)
            acc_bits = int(np.ceil(np.log2(acc_max + 1))) + 1
            # the shift must be the exact requantisation ratio (eq. T1)
            ratio = h["step"] * h["s_in"] / h["s_out"]
            err = float(np.abs(ratio - 2.0 ** -shift).max() / (2.0 ** -shift).max())
            rows.append(dict(name=n, shift=(int(shift.min()), int(shift.max())),
                             max_abs=max_abs, bias_max=bias_max,
                             acc_max=acc_max, acc_bits=acc_bits,
                             s_in=h["s_in"], s_out=h["s_out"], ratio_err=err))
            if max_abs > W_QMAX:
                bad.append(f"{n}: weight code {max_abs} > {W_QMAX} (clipped)")
            if not (0 <= shift.min() and shift.max() <= SHIFT_MAX):
                bad.append(f"{n}: shift {shift.min()}..{shift.max()} outside [0,{SHIFT_MAX}]")
            if acc_bits > ACC_BITS:
                bad.append(f"{n}: accumulator needs {acc_bits} bit > {ACC_BITS}")
            if err > 1e-6:
                bad.append(f"{n}: requant ratio is not a power of two (err {err:.2e})")
        assert not bad, "hardware contract violated:\n  " + "\n  ".join(bad)
        return rows
