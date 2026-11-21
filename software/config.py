"""Central configuration for the KWS ML pipeline (INT8 weight variant).

Everything that has to agree with the hardware (layer table, bit widths,
label order, MFCC geometry) is either defined here or imported from
../sim/reference_model.py, which stays the single source of truth for the
accelerator numerics.  The only line that differs between int4/ml and
int8/ml is W_BITS.
"""

import os
import sys

# -----------------------------------------------------------------------------
# Precision -- the one knob that separates the two hardware variants
# -----------------------------------------------------------------------------
W_BITS = 8                          # signed weight width used by the RTL
W_QMAX = (1 << (W_BITS - 1)) - 1    # 7 for INT4, 127 for INT8 (symmetric)
W_QMIN = -(1 << (W_BITS - 1))       # -8 / -128 (never produced, see quant.py)
ACT_BITS = 8                        # UINT8 activations in both variants
ACT_QMAX = (1 << ACT_BITS) - 1
PRECISION = f"int{W_BITS}"

# -----------------------------------------------------------------------------
# Hardware numeric contract (see docs/quantisation-math.md)
# -----------------------------------------------------------------------------
# The accelerator has no requantisation multiplier: every layer's output is
# produced by a per-channel arithmetic right shift.  That is only possible when
#     weight_step[oc] * s_in / s_out == 2**-shift[oc]
# i.e. weight_step[oc] = rho * 2**-shift[oc] with ONE free real constant rho per
# layer.  quantisation.py enforces exactly this; nothing else in the design may.
SHIFT_MAX = 15                      # shift is a 4-bit field in the RTL
ACC_BITS = 24                       # signed accumulator / bias width
GAP_SHIFT = 6                       # GAP: sum >>> 6, so s_gap = s_in * 64/65
LOGIT_SCALE = 1.0 / 16.0            # FC output LSB (argmax invariant, so it is free)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# __file__ is undefined inside the notebook, which must be started from software/
ML_DIR = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
          else os.getcwd())
VARIANT_ROOT = os.path.dirname(ML_DIR)                 # int4/ or int8/
REFERENCE_MODEL = os.path.join(VARIANT_ROOT, "sim", "reference_model.py")

DATA_ROOT = os.environ.get("KWS_DATA_ROOT", os.path.join(ML_DIR, "data"))
GSCD_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GSCD_DIR = os.path.join(DATA_ROOT, "speech_commands_v0.02")
CACHE_DIR = os.path.join(DATA_ROOT, "cache")           # raw-audio memmaps

ARTIFACTS = os.path.join(ML_DIR, "artifacts")
CKPT_FP32 = os.path.join(ARTIFACTS, "dscnn_fp32.pt")
CKPT_QAT = os.path.join(ARTIFACTS, f"dscnn_{PRECISION}_qat.pt")
EXPORT_DIR = os.path.join(ARTIFACTS, "export")         # staging tree

# -----------------------------------------------------------------------------
# Dataset -- Google Speech Commands v0.02, 12-class task (Warden 2018)
# Label order MUST match LABELS in sim/reference_model.py and golden.json.
# -----------------------------------------------------------------------------
LABELS = ["_silence_", "_unknown_", "yes", "no", "up", "down",
          "left", "right", "on", "off", "stop", "go"]
WANTED_WORDS = LABELS[2:]
SILENCE_IDX, UNKNOWN_IDX = 0, 1
NUM_CLASSES = len(LABELS)
SILENCE_PCT = 10.0       # silence clips, % of the known-word clip count
UNKNOWN_PCT = 10.0       # unknown clips, % of the known-word clip count

# -----------------------------------------------------------------------------
# MFCC front-end.  40 coefficients x 101 frames == KWS_H_IN x KWS_W_IN.
# 1 s @ 16 kHz, 30 ms window, 10 ms hop, centred frames -> 16000/160 + 1 = 101
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 160
N_MELS = 20  # chosen by the feature-dimension ablation (docs §44)
N_MFCC = 20  # = N_MELS: keep every mel band as a cepstral coefficient
F_MIN, F_MAX = 20.0, 7600.0
# Context actually fed to the network, in MFCC frames, so this is the one knob
# that changes the accelerator's input width (and therefore the activation-bank size).
N_FRAMES_USED = CLIP_SAMPLES // HOP_LENGTH + 1         # 101 = full 1 s
H_IN, W_IN = N_MFCC, N_FRAMES_USED                     # 40, 101

# Input quantiser: after per-coefficient standardisation the feature map is
# clipped to [IN_CLIP_LO, IN_CLIP_HI] sigma and mapped to UINT8 with one
# per-tensor (scale, zero point) -- exactly what the stem's zp_in expects.
IN_CLIP_LO, IN_CLIP_HI = -3.0, 5.0

# -----------------------------------------------------------------------------
# Augmentation (training only)
# -----------------------------------------------------------------------------
TIME_SHIFT_MS = 100
BG_NOISE_PROB = 0.8
BG_NOISE_VOL = 0.1
SPEC_AUG = True          # light time / frequency masking on the MFCC map

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
SEED = 20261120
BATCH_SIZE = 128
# worker processes need fork(); spawn platforms (Windows, macOS) cannot pickle
# notebook-defined classes, so they load data in the main process
NUM_WORKERS = 4 if sys.platform.startswith("linux") else 0
FP32_EPOCHS = 60
FP32_LR = 3e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
DROPOUT = 0.1

# Quantisation-aware fine-tuning.  INT4 needs noticeably more QAT than INT8.
QAT_EPOCHS = 20 if W_BITS <= 4 else 8
QAT_LR = 5e-4
QAT_FREEZE_RANGES_EPOCH = max(1, QAT_EPOCHS - 3)   # stop updating act ranges
CALIB_BATCHES = 32
ACT_EMA = 0.99
ACT_PERCENTILE = 99.99   # initial activation range, robust to outliers

DEVICE = os.environ.get("KWS_DEVICE", "")             # "" = auto
