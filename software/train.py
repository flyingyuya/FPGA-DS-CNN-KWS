"""Two-stage training: FP32 DS-CNN, then quantisation-aware fine-tuning.

  python train.py --stage fp32        # train DS-CNN with BatchNorm
  python train.py --stage qat         # fold BN, fake-quantise, fine-tune
  python train.py --stage all         # both (default)

Checkpoints land in artifacts/.  The QAT checkpoint holds everything the
exporter needs: folded weights and calibrated activation ranges.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (ARTIFACTS, BATCH_SIZE, CKPT_FP32, CKPT_QAT, DEVICE,
                    FP32_EPOCHS, FP32_LR, LABEL_SMOOTHING, NUM_WORKERS,
                    PRECISION, QAT_EPOCHS, QAT_FREEZE_RANGES_EPOCH, QAT_LR, SEED,
                    W_BITS, WEIGHT_DECAY)
from data import AudioDataset, build_pipeline
from model import CONV_NAMES, DSCNN, count_macs, count_params, fold_bn
from quantisation import QDSCNN


def pick_device(name=DEVICE):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_loader(split, shuffle, batch_size=BATCH_SIZE, workers=NUM_WORKERS):
    return DataLoader(AudioDataset(split), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate_model(model, loader, pipe, device, max_batches=None):
    """Top-1 accuracy (%) and confusion matrix of a float / fake-quant model."""
    model.eval()
    conf = torch.zeros(12, 12, dtype=torch.long)
    for i, (wav, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        pred = model(pipe(wav.to(device))).argmax(1).cpu()
        conf += torch.bincount(y * 12 + pred, minlength=144).view(12, 12)
    return 100.0 * conf.diag().sum().item() / max(1, conf.sum().item()), conf


def _cosine(opt, base_lr, step, total, warmup):
    lr = base_lr * (step / warmup if step < warmup else
                    0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def fit(model, pipe, device, epochs, lr, ckpt_path, ckpt_extra, max_batches=None,
        on_epoch_start=None):
    train_dl, val_dl = make_loader("train", True), make_loader("val", False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    steps = epochs * (min(len(train_dl), max_batches) if max_batches else len(train_dl))
    warmup = max(1, steps // 20)
    step, best = 0, -1.0
    for ep in range(epochs):
        if on_epoch_start:
            on_epoch_start(ep)
        model.train()
        t0, tot, correct, seen = time.time(), 0.0, 0, 0
        for i, (wav, y) in enumerate(train_dl):
            if max_batches and i >= max_batches:
                break
            wav, y = wav.to(device), y.to(device)
            cur = _cosine(opt, lr, step, steps, warmup)
            logits = model(pipe(wav, train=True))
            loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            tot += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        val_acc, _ = evaluate_model(model, val_dl, pipe, device, max_batches)
        print(f"  epoch {ep + 1:3d}/{epochs}  loss {tot / max(1, seen):.4f}  "
              f"train {100.0 * correct / max(1, seen):6.2f}%  val {val_acc:6.2f}%  "
              f"lr {cur:.2e}  {time.time() - t0:5.1f}s", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(dict(state_dict=model.state_dict(), val_acc=val_acc,
                            epoch=ep + 1, **ckpt_extra(model)), ckpt_path)
    print(f"  best val {best:.2f}% -> {ckpt_path}")
    return best


def train_fp32(device, epochs=FP32_EPOCHS, max_batches=None):
    seed_all()
    pipe = build_pipeline(device)
    model = DSCNN().to(device)
    print(f"DS-CNN: {count_params(model)} parameters, {count_macs()} MACs / inference")
    return fit(model, pipe, device, epochs, FP32_LR, CKPT_FP32,
               lambda m: {}, max_batches)


def load_fp32(device):
    model = DSCNN()
    model.load_state_dict(torch.load(CKPT_FP32, map_location="cpu", weights_only=True)["state_dict"])
    return model.to(device).eval()


def build_qat_model(device):
    """Fold BN of the best FP32 checkpoint and wrap it in QDSCNN."""
    folded = fold_bn(load_fp32("cpu"))
    return QDSCNN(folded).to(device)


def train_qat(device, epochs=QAT_EPOCHS, max_batches=None):
    seed_all(SEED + 1)
    pipe = build_pipeline(device)
    qm = build_qat_model(device)
    calib = (pipe(w.to(device)) for w, _ in make_loader("train", True))
    qm.calibrate(calib)
    val_dl = make_loader("val", False)
    qm.set_quant(False)
    fp_acc, _ = evaluate_model(qm, val_dl, pipe, device, max_batches)
    qm.set_quant(True)
    ptq_acc, _ = evaluate_model(qm, val_dl, pipe, device, max_batches)
    print(f"BN-folded FP32 val {fp_acc:.2f}%   post-training {PRECISION} val {ptq_acc:.2f}%")

    def on_epoch(ep):
        qm.set_observe(ep < QAT_FREEZE_RANGES_EPOCH)

    def extra(m):
        return dict(ranges=m.act_ranges(), w_bits=W_BITS, ptq_val_acc=ptq_acc,
                    fp32_val_acc=fp_acc)
    print(f"QAT ({PRECISION} weights, UINT8 activations), {epochs} epochs")
    return fit(qm, pipe, device, epochs, QAT_LR, CKPT_QAT, extra, max_batches, on_epoch)


def load_qat(device):
    ck = torch.load(CKPT_QAT, map_location="cpu", weights_only=True)
    if ck.get("w_bits", W_BITS) != W_BITS:
        raise ValueError(f"{CKPT_QAT} was trained for INT{ck['w_bits']}, config says INT{W_BITS}")
    sd = ck["state_dict"]
    qm = QDSCNN({n: (sd[f"w.{n}"], sd[f"b.{n}"]) for n in CONV_NAMES + ["fc"]})
    qm.load_state_dict(sd)
    return qm.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fp32", "qat", "all"], default="all")
    ap.add_argument("--fp32-epochs", type=int, default=FP32_EPOCHS)
    ap.add_argument("--qat-epochs", type=int, default=QAT_EPOCHS)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per epoch (smoke tests)")
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    os.makedirs(ARTIFACTS, exist_ok=True)
    dev = pick_device(a.device)
    print(f"device: {dev}   precision: {PRECISION}")
    if a.stage in ("fp32", "all"):
        train_fp32(dev, a.fp32_epochs, a.max_batches)
    if a.stage in ("qat", "all"):
        train_qat(dev, a.qat_epochs, a.max_batches)


if __name__ == "__main__":
    main()