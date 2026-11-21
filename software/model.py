"""DS-CNN for 20x101 MFCC keyword spotting, matching the accelerator layer table.

    input  1 x 20 x 101 (UINT8 MFCC codes in hardware)
    stem   3x3 conv, 32 ch, stride 2      -> 32 x 10 x 51
    dw1    3x3 depthwise, stride 2        -> 32 x 5  x 26
    pw1    1x1 conv, 32 ch                -> 32 x 5  x 26
    dw2    3x3 depthwise, stride 2        -> 32 x 3  x 13
    pw2    1x1 conv, 64 ch                -> 64 x 3  x 13
    dw3/pw3, dw4/pw4  (stride 1, 64 ch)   -> 64 x 3  x 13
    gap    global average pool            -> 64
    fc     64 -> 12 logits

Every conv is followed by BatchNorm + ReLU during training.  BatchNorm is then
folded into the conv weight and bias (fold_bn) because the hardware has no BN
unit: it is a per-channel affine map, so it collapses into the per-channel
weight scale and the INT32 bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT, NUM_CLASSES

# (name, kind, cin, cout, stride) -- must equal LAYERS in sim/reference_model.py
LAYER_SPEC = [
    ("stem", "std", 1, 32, 2),
    ("dw1", "dw", 32, 32, 2),
    ("pw1", "pw", 32, 32, 1),
    ("dw2", "dw", 32, 32, 2),
    ("pw2", "pw", 32, 64, 1),
    ("dw3", "dw", 64, 64, 1),
    ("pw3", "pw", 64, 64, 1),
    ("dw4", "dw", 64, 64, 1),
    ("pw4", "pw", 64, 64, 1),
]
FC_SPEC = ("fc", "fc", 64, NUM_CLASSES, 1)
CONV_NAMES = [s[0] for s in LAYER_SPEC]

def make_conv(kind: str, cin, cout, stride, bias):
    if kind == "std":
        return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias)
    if kind == "dw":
        return nn.Conv2d(cin, cin, kernel_size=3, stride=stride, padding=1,
                         groups=cin, bias=bias)
    if kind == "pw":
        return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

class DSCNN(nn.Module):
    """FP32 training model (conv -> BN -> ReLU)."""

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        for name, kind, cin, cout, stride in LAYER_SPEC:
            self.convs[name] = make_conv(kind, cin, cout, stride, bias=False)
            self.bns[name] = nn.BatchNorm2d(cout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(FC_SPEC[2], FC_SPEC[3])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for name in CONV_NAMES:
            # Conv -> BatchNorm -> ReLU
            x = F.relu(self.bns[name](self.convs[name](x)))
        x = x.mean(dim=(2, 3))
        return self.fc(self.dropout(x))

@torch.no_grad()
def fold_bn(model):
    """Return {name: (weight, bias)} with BatchNorm folded into each conv.

        y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta
          = conv'(x) + b'   with  w' = w * g,  b' = beta - mu * g,
                                  g  = gamma / sqrt(var + eps)
    """
    out = {}
    for name in CONV_NAMES:
        conv, bn = model.convs[name], model.bns[name]
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * g.view(-1, 1, 1, 1)
        b = bn.bias - bn.running_mean * g
        if conv.bias is not None:
            b = b + conv.bias * g
        out[name] = (w.detach().clone(), b.detach().clone())
    out["fc"] = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs():
    """Multiply-accumulates per inference for the fixed 40x101 input."""
    h, w, macs = 40, 101, 0
    for _, kind, cin, cout, stride in LAYER_SPEC:
        if kind != "pw":
            h, w = (h + 2 - 3) // stride + 1, (w + 2 - 3) // stride + 1
        k = 9 if kind == "std" else 9 if kind == "dw" else 1
        macs += h * w * cout * (k * cin if kind == "std" else k if kind == "dw" else cin)
    return macs + FC_SPEC[2] * FC_SPEC[3]