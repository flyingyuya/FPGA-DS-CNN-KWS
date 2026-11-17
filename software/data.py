"""Google Speech Commands v0.02 loading for the 12-class KWS task.

Pipeline:
  1. download_speech_commands()  fetch + extract the 2.3 GB archive (once)
  2. build_index(split)          list (wav path, label) for train/val/test,
                                 using the official validation/testing lists,
                                 plus sampled _unknown_ words and synthetic
                                 _silence_ clips cut from _background_noise_
  3. cache_split(split)          decode every clip once into an int16 memmap
                                 (N, 16000) so epochs never touch the WAVs
  4. AudioDataset                memmap -> (int16 waveform, label)
  5. FeaturePipeline             batched augmentation + MFCC + normalisation

Raw audio is cached rather than MFCCs so time-shift and noise augmentation
can still run per batch.
"""

import glob
import json
import os
import tarfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import (N_FRAMES_USED,ARTIFACTS, BG_NOISE_PROB, BG_NOISE_VOL, CACHE_DIR,
                    CLIP_SAMPLES, DATA_ROOT, LABELS, NUM_CLASSES, SAMPLE_RATE,
                    GSCD_DIR, GSCD_URL, SEED, SILENCE_IDX, SILENCE_PCT, SPEC_AUG,
                    TIME_SHIFT_MS, UNKNOWN_IDX, UNKNOWN_PCT, WANTED_WORDS)
from features import MFCC, normalise

SPLITS = ("train", "val", "test")
STATS_JSON = os.path.join(ARTIFACTS, "feature_stats.json")


# -----------------------------------------------------------------------------
# Download / WAV I/O
# -----------------------------------------------------------------------------
def download_gscd(url=GSCD_URL, dest=GSCD_DIR):
    """ Download and extract Speech Commands v0.02 into dest (idempotent). """
    if os.path.isfile(os.path.join(dest, "testing_list.txt")):
        return dest
    os.makedirs(dest, exist_ok=True)
    tgz = os.path.join(DATA_ROOT, os.path.basename(url))
    if not os.path.isfile(tgz):
        print(f"downloading {url} -> {tgz} (~2.3 GB)")

        def hook(blocks, bs, total):
            if total > 0 and blocks % 2000 == 0:
                print(f"  {100.0 * blocks * bs / total:5.1f}%", flush=True)
        urllib.request.urlretrieve(url, tgz + ".part", hook)
        os.replace(tgz + ".part", tgz)
    print(f"extracting {tgz}")
    with tarfile.open(tgz, "r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:                       # Python < 3.12
            tf.extractall(dest)
    return dest


def read_wav(path):
    """ Read a *.wav file, 16-bit PCM mono WAV -> float32 in [-1, 1]. """
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0]
    return x.astype(np.float32) / 32768.0


def fit_length(x, n=CLIP_SAMPLES):
    """ Audio length alignment, more cuts and fewer compenstations. """
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def load_background_noise(root=GSCD_DIR):
    """ Load all _background_noise_/*.wav clips as float32 in [-1, 1]. """
    files = sorted(glob.glob(os.path.join(root, "_background_noise_", "*.wav")))
    return [read_wav(f) for f in files]


# -----------------------------------------------------------------------------
# Split index
# -----------------------------------------------------------------------------
def _read_list(root, name):
    with open(os.path.join(root, name)) as f:
        return set(ln.strip() for ln in f if ln.strip())


def build_index(split, root=GSCD_DIR, seed=SEED):
    """Return a list of (relative wav path or None, label index).

    None marks a _silence_ clip, cut from background noise at cache time.
    _unknown_ and _silence_ are each sampled to UNKNOWN_PCT / SILENCE_PCT of
    the known-word count, the standard Speech Commands 12-class recipe."""
    val, test = _read_list(root, "validation_list.txt"), _read_list(root, "testing_list.txt")
    rng = np.random.default_rng(seed + SPLITS.index(split))
    known, unknown = [], []
    for wav in sorted(glob.glob(os.path.join(root, "*", "*.wav"))):
        word = os.path.basename(os.path.dirname(wav))
        if word.startswith("_"):
            continue
        rel = f"{word}/{os.path.basename(wav)}"
        s = "val" if rel in val else "test" if rel in test else "train"
        if s != split:
            continue
        if word in WANTED_WORDS:
            known.append((rel, LABELS.index(word)))
        else:
            unknown.append((rel, UNKNOWN_IDX))
    n_unk = int(round(len(known) * UNKNOWN_PCT / 100.0))
    n_sil = int(round(len(known) * SILENCE_PCT / 100.0))
    pick = rng.choice(len(unknown), size=min(n_unk, len(unknown)), replace=False)
    items = known + [unknown[i] for i in sorted(pick)] + [(None, SILENCE_IDX)] * n_sil
    order = rng.permutation(len(items))
    return [items[i] for i in order]


# -----------------------------------------------------------------------------
# Raw-audio cache
# -----------------------------------------------------------------------------
def cache_paths(split):
    return (os.path.join(CACHE_DIR, f"{split}_audio.npy"),
            os.path.join(CACHE_DIR, f"{split}_labels.npy"))


def cache_split(split, root=GSCD_DIR, seed=SEED):
    """ Decode a split into an int16 (N, 16000) memmap plus a label vector. """
    a_path, l_path = cache_paths(split)
    if os.path.isfile(a_path) and os.path.isfile(l_path):
        return a_path, l_path
    os.makedirs(CACHE_DIR, exist_ok=True)
    items = build_index(split, root, seed)
    noise = load_background_noise(root)
    rng = np.random.default_rng(seed + 100 + SPLITS.index(split))
    audio = np.lib.format.open_memmap(a_path + ".part", mode="w+", dtype=np.int16,
                                      shape=(len(items), CLIP_SAMPLES))
    labels = np.zeros(len(items), dtype=np.int64)
    for i, (rel, lab) in enumerate(items):
        if rel is None:
            # _silence_: a random 1 s crop of background noise at a random
            # volume (including near-zero, i.e. true silence)
            n = noise[rng.integers(len(noise))]
            off = rng.integers(0, len(n) - CLIP_SAMPLES)
            x = n[off:off + CLIP_SAMPLES] * rng.uniform(0.0, 0.5)
        else:
            x = fit_length(read_wav(os.path.join(root, rel)))
        audio[i] = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
        labels[i] = lab
        if i % 5000 == 0:
            print(f"  {split}: {i}/{len(items)}", flush=True)
    audio.flush()
    del audio
    os.replace(a_path + ".part", a_path)
    np.save(l_path, labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    print(f"  {split}: {len(items)} clips, per class {counts.tolist()}")
    return a_path, l_path


def cache_noise(root=GSCD_DIR):
    path = os.path.join(CACHE_DIR, "background_noise.npy")
    if not os.path.isfile(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, np.concatenate(load_background_noise(root)).astype(np.float32))
    return path


def make_synthetic_cache(n_per_class=40, seed=SEED):
    """Tiny tone-based stand-in dataset for smoke-testing the pipeline without
    the 2.3 GB download.  Accuracy numbers from it mean nothing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(CLIP_SAMPLES) / SAMPLE_RATE
    for split in SPLITS:
        n = n_per_class if split == "train" else max(4, n_per_class // 4)
        xs, ys = [], []
        for c in range(NUM_CLASSES):
            for _ in range(n):
                f0 = 200.0 + 180.0 * c
                on = rng.uniform(0.1, 0.4)
                env = ((t > on) & (t < on + 0.5)).astype(np.float32)
                x = 0.3 * env * np.sin(2 * np.pi * f0 * t) * (c != SILENCE_IDX)
                x = x + 0.01 * rng.normal(size=CLIP_SAMPLES)
                xs.append(np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16))
                ys.append(c)
        order = rng.permutation(len(xs))
        a_path, l_path = cache_paths(split)
        np.save(a_path, np.stack(xs)[order])
        np.save(l_path, np.array(ys, dtype=np.int64)[order])
    np.save(os.path.join(CACHE_DIR, "background_noise.npy"),
            (0.05 * rng.normal(size=SAMPLE_RATE * 10)).astype(np.float32))
    print(f"synthetic cache written to {CACHE_DIR}")


def prepare_all(synthetic=False):
    if synthetic:
        make_synthetic_cache()
        return
    download_gscd()
    for s in SPLITS:
        cache_split(s)
    cache_noise()


# -----------------------------------------------------------------------------
# Dataset + batched feature pipeline
# -----------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, split: str):
        a_path, l_path = cache_paths(split)
        if not os.path.isfile(a_path):
            raise FileNotFoundError(f"{a_path} missing -- run prepare_data.py first")
        self.audio = np.load(a_path, mmap_mode="r")
        self.labels = np.load(l_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.audio[i])), int(self.labels[i])


class FeaturePipeline(nn.Module):
    """int16 waveform batch -> normalised MFCC batch (B, 1, 40, 101).

    With train=True it also applies random time shift, background-noise mixing
    and light SpecAugment.  The eval path is the deployed front-end."""

    def __init__(self, mean, std, noise=None):
        super().__init__()
        self.mfcc = MFCC()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        noise = torch.zeros(CLIP_SAMPLES * 2) if noise is None else torch.as_tensor(noise)
        self.register_buffer("noise", noise.float())

    def _augment(self, x):
        B, N = x.shape
        dev = x.device
        # random time shift with zero fill
        sh = int(SAMPLE_RATE * TIME_SHIFT_MS / 1000)
        shift = torch.randint(-sh, sh + 1, (B, 1), device=dev)
        idx = torch.arange(N, device=dev)[None, :] - shift
        valid = (idx >= 0) & (idx < N)
        x = torch.gather(x, 1, idx.clamp(0, N - 1)) * valid
        # background noise mix
        if self.noise.numel() > N:
            off = torch.randint(0, self.noise.numel() - N, (B, 1), device=dev)
            seg = self.noise[off + torch.arange(N, device=dev)[None, :]]
            vol = torch.rand(B, 1, device=dev) * BG_NOISE_VOL
            vol = vol * (torch.rand(B, 1, device=dev) < BG_NOISE_PROB)
            x = x + vol * seg
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _spec_augment(f):
        B, H, W = f.shape
        dev = f.device
        out = f.clone()
        for _ in range(2):                                   # time masks
            w = torch.randint(0, 11, (B, 1), device=dev)
            t0 = (torch.rand(B, 1, device=dev) * (W - w)).long()
            t = torch.arange(W, device=dev)[None, :]
            out = out.masked_fill(((t >= t0) & (t < t0 + w))[:, None, :], 0.0)
        h = torch.randint(0, 5, (B, 1), device=dev)          # one freq mask
        h0 = (torch.rand(B, 1, device=dev) * (H - h)).long()
        r = torch.arange(H, device=dev)[None, :]
        return out.masked_fill(((r >= h0) & (r < h0 + h))[:, :, None], 0.0)

    def forward(self, wav16, train=False):
        x = wav16.float() / 32768.0
        if train:
            x = self._augment(x)                            # B, 16000
        f = normalise(self.mfcc(x), self.mean, self.std)    # B, 40, 101
        if N_FRAMES_USED < f.shape[2]:                      # centred context crop
            off = (f.shape[2] - N_FRAMES_USED) // 2
            f = f[:, :, off:off + N_FRAMES_USED]
        if train and SPEC_AUG:
            f = self._spec_augment(f)
        return f.unsqueeze(1)


@torch.no_grad()
def compute_feature_stats(batch_size=512, path=STATS_JSON):
    """Per-coefficient MFCC mean / std over the (un-augmented) training set."""
    ds = AudioDataset("train")
    mfcc = MFCC()
    s1 = torch.zeros(40, dtype=torch.float64)
    s2 = torch.zeros(40, dtype=torch.float64)
    n = 0
    for i in range(0, len(ds), batch_size):
        x = torch.from_numpy(np.array(ds.audio[i:i + batch_size])).float() / 32768.0
        m = mfcc(x).double()                                  # (B, 40, 101)
        s1 += m.sum(dim=(0, 2))
        s2 += (m ** 2).sum(dim=(0, 2))
        n += m.shape[0] * m.shape[2]
    mean = s1 / n
    std = torch.sqrt(torch.clamp(s2 / n - mean ** 2, min=1e-8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(mean=mean.tolist(), std=std.tolist()), f, indent=1)
    return mean.float(), std.float()


def load_feature_stats(path=STATS_JSON):
    if not os.path.isfile(path):
        return compute_feature_stats(path=path)
    with open(path) as f:
        d = json.load(f)
    return torch.tensor(d["mean"]), torch.tensor(d["std"])


def build_pipeline(device="cpu"):
    mean, std = load_feature_stats()
    noise_path = os.path.join(CACHE_DIR, "background_noise.npy")
    noise = np.load(noise_path) if os.path.isfile(noise_path) else None
    return FeaturePipeline(mean, std, noise).to(device)
