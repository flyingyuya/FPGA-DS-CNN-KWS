"""Bit-exact integer model of the accelerator (batched, torch, fast).

Scope: this file answers "what accuracy will the deployed hardware get".  It
consumes only `QDSCNN.hw_params()`, so it cannot drift away from the training
graph or the exporter.

Two independent references exist on purpose:

  * `IntDSCNN` (here)      batched tensor maths, used for test-set accuracy
  * `sim/reference_model.py` single sample, word/lane addressing identical to the
                            RTL, used to generate the golden vectors

They must agree bit for bit; `export_mem.py` asserts that on every clip it
exports.

The arithmetic is exactly what the RTL does:

    acc      = B[oc] + sum_t q_w[oc,t] * code[t]        (24-bit signed, no overflow)
    code_out = clamp( acc >>> shift[oc], 0, 255)        (shift carries +2**(S-1))
    gap      = clamp( (sum_pix code + 2**(G-1)) >>> G, 0, 255 )
    logits   = acc >>> shift[oc]                        (signed, no clamp)

Everything is carried in float64 whose 53-bit mantissa holds the integers
exactly (|acc| < 2**24), so the "float" conv2d below is an exact integer
convolution.
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import ACT_QMAX, GAP_SHIFT, LOGIT_SCALE
from model import CONV_NAMES, FC_SPEC, LAYER_SPEC

ORDER = list(CONV_NAMES) + ["gap", "fc"]
STRIDE = {n: s for n, _, _, _, s in LAYER_SPEC}
STRIDE["fc"] = 1


class IntDSCNN:
    """Batched integer forward driven by the exported hardware constants."""

    def __init__(self, hw, device="cpu"):
        self.hw = hw
        self.device = torch.device(device)
        self.t = {}
        kinds = {s[0]: s[1] for s in LAYER_SPEC}
        for n in CONV_NAMES + ["fc"]:
            h = hw[n]
            w = np.asarray(h["q"], dtype=np.float64)
            self.t[n] = dict(
                w=torch.from_numpy(w).to(self.device),
                B=torch.from_numpy(np.asarray(h["B"], dtype=np.float64)).to(self.device),
                shift=torch.from_numpy(np.asarray(h["shift"], dtype=np.float64)).to(self.device),
                kind=kinds.get(n, "fc"),
                zp_in=float(h.get("zp_in", 0)),
            )
        self.gap_shift = float(hw["gap"]["shift"])
        self.gap_n = float(hw["gap"]["N"])

    # -- one layer ------------------------------------------------------------
    def _shift(self, acc, shift, width):
        s = shift.reshape((1, -1) + (1,) * (width - 2))
        return torch.floor(acc / torch.pow(2.0, s))

    def forward(self, xq, trace=None):
        """xq: (B, 40, 101) UINT8 codes -> (B, 12) integer logit codes."""
        x = xq.to(self.device).to(torch.float64)
        for n in CONV_NAMES:
            t = self.t[n]
            kind = t["kind"]
            w = t["w"]
            if kind == "pw":                                   # (cout, cin)
                w4 = w.reshape(w.shape[0], w.shape[1], 1, 1)
                groups = 1
            else:
                w4 = w
                groups = w.shape[0] if kind == "dw" else 1
            if kind != "pw":
                # the RTL sources the layer's input zero point for out-of-range
                # taps; that is what keeps the folded bias exact (math notes 3)
                x = F.pad(x, (1, 1, 1, 1), value=t["zp_in"])
            acc = F.conv2d(x, w4, None, STRIDE[n], 0, groups=groups)
            acc = acc + t["B"].reshape((1, -1) + (1,) * 2)
            if trace is not None:
                trace.append((n, float(acc.abs().max())))
            x = torch.clamp(self._shift(acc, t["shift"], 4), 0, ACT_QMAX)
        codes = x
        gap = torch.clamp(torch.floor(codes.sum(dim=(2, 3), keepdim=True)
                                     / 2 ** self.gap_shift + 0.5), 0, ACT_QMAX)
        t = self.t["fc"]
        acc = F.linear(gap.flatten(1), t["w"]) + t["B"].reshape(1, -1)
        # B already carries the 2**(S-1) rounding constant, so the shift itself
        # is the rounding step -- adding 0.5 here would round twice
        return self._shift(acc, t["shift"], 2)

    def __call__(self, xq, trace=None):
        return self.forward(xq, trace)


@torch.no_grad()
def accuracy(model, xq, y, batch=256):
    """Top-1 accuracy (%) and the confusion matrix of the integer model."""
    from config import NUM_CLASSES
    pred = torch.cat([model(xq[i:i + batch]).argmax(1).cpu()
                      for i in range(0, len(y), batch)])
    conf = torch.bincount(y * NUM_CLASSES + pred,
                          minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return 100.0 * (pred == y).float().mean().item(), conf, pred


@torch.no_grad()
def per_layer_peaks(model, xq):
    """Worst-case |accumulator| actually reached per layer on real data."""
    trace = []
    model(xq, trace=trace)
    return {n: v for n, v in ((r[0], r[1]) for r in trace)}
