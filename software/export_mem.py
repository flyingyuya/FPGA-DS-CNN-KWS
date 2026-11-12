"""Export the trained QAT model into every memory image the accelerator loads.

  python export_mem.py              # write into artifacts/export/ (staging)
  python export_mem.py --install    # write into the variant root (int4/ or int8/)

The writers are imported from ../tb/reference_model.py, so the file formats,
packing, unified-ROM addressing and golden-vector formats are byte-identical
to what the RTL and testbenches already expect.  This script only replaces the
*content*: trained weights instead of synthetic ones, and real Speech Commands
test clips instead of synthetic MFCC blobs.

Produced tree (relative to the output root):
  mem/weights_int{4,8}/w_<layer>.mem   packed weight ROMs
  mem/bias/b_<layer>.mem, b_all.mem    INT32 biases
  mem/scales/m_*.mem, sw_*.mem, m_all.mem, zero_points.mem, gap_cfg.mem
  mem/lut/exp_lut.mem                  softmax exp LUT
  mem/input/mfcc_<vector>.mem          14 UINT8 MFCC test frames
  golden/*.mem, golden/golden.json     layer + end-to-end golden results
  rtl/kws_params.vh                    regenerated header (geometry unchanged)
  docs/quant_summary.csv, docs/ml_report.json
"""

import argparse
import importlib.util
import json
import os
import shutil

import numpy as np
import torch

from config import (ARTIFACTS, EXPORT_DIR, LABELS, PRECISION, REFERENCE_MODEL,
                    SEED, VARIANT_ROOT, W_BITS)
from data import AudioDataset, build_pipeline, load_feature_stats
from features import quantise_input
from int_model import IntDSCNN, build_hw_qparams
from model import LAYER_SPEC
from train import load_qat, pick_device


def load_reference_model(path=REFERENCE_MODEL):
    spec = importlib.util.spec_from_file_location("reference_model", path)
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    return rm


def check_contract(rm):
    """Fail early if the ML model and the hardware reference disagree."""
    assert rm.W_BITS == W_BITS, f"reference model is INT{rm.W_BITS}, ML config INT{W_BITS}"
    assert list(rm.LABELS) == list(LABELS), "label order differs from reference model"
    hw = [(l["name"], l["kind"], l["cin"], l["cout"], l["stride"])
          for l in rm.LAYERS if l["kind"] in ("std", "dw", "pw")]
    assert hw == [tuple(s) for s in LAYER_SPEC], f"layer table mismatch:\n{hw}\n{LAYER_SPEC}"
    assert (rm.H_IN, rm.W_IN) == (40, 101)


@torch.no_grad()
def quantised_split(split, pipe, s, zp, device, batch=512):
    ds = AudioDataset(split)
    xs = []
    for i in range(0, len(ds), batch):
        wav = torch.from_numpy(np.array(ds.audio[i:i + batch])).to(device)
        xs.append(quantise_input(pipe(wav), s, zp).cpu())
    return torch.cat(xs), torch.from_numpy(ds.labels)


@torch.no_grad()
def int_accuracy(imodel, xq, y, batch=512):
    pred = torch.cat([imodel(xq[i:i + batch]).argmax(1).cpu() for i in range(0, len(y), batch)])
    conf = torch.bincount(y * 12 + pred, minlength=144).view(12, 12)
    return 100.0 * (pred == y).float().mean().item(), conf, pred


def pick_vectors(xq, y, pred):
    """One test clip per class (first one the integer model gets right, else
    the first one), then the two edge frames all-0 and all-255."""
    names, frames, src = [], [], []
    for c, lab in enumerate(LABELS):
        idx = torch.nonzero(y == c).flatten()
        if len(idx) == 0:
            raise RuntimeError(f"no test clip for class {lab}")
        ok = idx[pred[idx] == c]
        i = int(ok[0] if len(ok) else idx[0])
        names.append(f"class{c:02d}_{lab}")
        frames.append(xq[i, 0].numpy().astype(np.uint8)[None])
        src.append(i)
    for nm, v in (("edge_all_min", 0), ("edge_all_max", 255)):
        names.append(nm)
        frames.append(np.full((1, 40, 101), v, dtype=np.uint8))
        src.append(-1)
    return names, frames, src


def write_quant_summary(rm, q, path):
    """Same CSV that reference_model.main() writes."""
    with open(path, "w", newline="\n") as f:
        f.write("layer,kind,cin,cout,hi,wi,ho,wo,stride,"
                "act_scale,zero_point,Mq_min,Mq_max,M_rel_err_pct\n")
        f.write(f"input,-,1,1,{rm.H_IN},{rm.W_IN},{rm.H_IN},{rm.W_IN},1,"
                f"{q['sa']['input']:.8f},{q['zp']['input']},,,\n")
        for l in rm.LAYERS:
            n = l["name"]
            sa, zp = q["sa"].get(n, float("nan")), q["zp"].get(n, 0)
            row = (f"{n},{l['kind']},{l['cin']},{l['cout']},{l['hi']},{l['wi']},"
                   f"{l['ho']},{l['wo']},{l['stride']},{sa:.8f},{zp},")
            if n in q["M"]:
                row += f"{int(q['M'][n].min())},{int(q['M'][n].max())},{q['M_err'][n] * 100:.4f}"
            else:
                row += ",,"
            f.write(row + "\n")


def export(out_root, device, verify_n=4):
    rm = load_reference_model()
    check_contract(rm)
    qm, ck = load_qat("cpu")
    prm = qm.float_params()
    ranges = qm.act_ranges()
    print(f"QAT checkpoint: epoch {ck['epoch']}, val {ck['val_acc']:.2f}% (fake-quant)")

    print("\n-- hardware quantisation parameters --")
    q = build_hw_qparams(prm, ranges)
    s_in, zp_in = q["sa"]["input"], q["zp"]["input"]
    print(f"  input: s={s_in:.6f} zp={zp_in}")
    imodel = IntDSCNN(q, device)
    lut = rm.build_exp_lut()

    print("\n-- integer test-set accuracy --")
    pipe = build_pipeline(device).eval()
    xq, y = quantised_split("test", pipe, s_in, zp_in, device)
    acc, conf, pred = int_accuracy(imodel, xq, y)
    print(f"  {PRECISION} bit-exact accelerator model: {acc:.2f}% on {len(y)} test clips")

    # Cross-check the batched integer model against the reference golden model.
    for i in range(min(verify_n, len(y))):
        r = rm.int_forward(xq[i].numpy().astype(np.uint8), q, lut)
        mine = imodel(xq[i:i + 1])[0].cpu().numpy()
        assert np.array_equal(r["logits"], mine), \
            f"int_model / reference_model mismatch on clip {i}: {mine} vs {r['logits']}"
    print(f"  batched int model == reference_model.int_forward on {verify_n} clips")

    names, xqs, src = pick_vectors(xq, y, pred)

    print(f"\n-- writing memory images under {out_root} --")
    rm.ROOT = out_root                          # every writer resolves paths via rm.p()
    for sub in ("mem", "golden", "rtl", "docs"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    info = rm.emit_memories(q, lut, None, names, xqs)
    depth = rm.emit_vh(q, info)
    rng = np.random.default_rng(SEED)
    n1 = rm.emit_requant_vectors(q, rng)
    n2 = rm.emit_mac_vectors(rng)
    n3 = rm.emit_gap_vectors(rng)
    n4 = rm.emit_softmax_vectors(q, lut, rng)
    print(f"  unit vectors: requant {n1}, mac {n2}, gap {n3}, softmax {n4}")
    print(f"  activation bank depth {depth} words")
    print("\n-- golden inference results --")
    summary = rm.emit_golden(q, lut, names, xqs)
    write_quant_summary(rm, q, os.path.join(out_root, "docs", "quant_summary.csv"))

    # Host-side artefacts: folded FP32 model (accepted by reference_model.py
    # --from-npz) and the front-end constants a host needs to feed the chip.
    os.makedirs(ARTIFACTS, exist_ok=True)
    np.savez(os.path.join(ARTIFACTS, f"dscnn_{PRECISION}_folded.npz"), **prm)
    mean, std = load_feature_stats()
    report = dict(
        precision=PRECISION, test_clips=int(len(y)), int_test_acc=acc,
        qat_val_acc=float(ck["val_acc"]), ptq_val_acc=float(ck.get("ptq_val_acc", -1)),
        fp32_val_acc=float(ck.get("fp32_val_acc", -1)),
        confusion=conf.tolist(), labels=LABELS,
        vectors=[dict(name=n, test_index=i, golden=s["label"], conf_q16=s["conf"])
                 for n, i, s in zip(names, src, summary)],
        frontend=dict(sample_rate=16000, n_fft=512, win=480, hop=160, n_mels=40,
                      n_mfcc=40, fmin=20.0, fmax=7600.0, mean=mean.tolist(),
                      std=std.tolist(), input_scale=s_in, input_zero_point=zp_in),
        act_ranges={k: list(v) for k, v in ranges.items()},
    )
    with open(os.path.join(out_root, "docs", "ml_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out_root}/docs/ml_report.json")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help=f"write straight into {VARIANT_ROOT} (overwrites mem/, golden/, "
                         "rtl/kws_params.vh, docs/quant_summary.csv)")
    ap.add_argument("--out", default=EXPORT_DIR, help="staging output root")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    out = VARIANT_ROOT if a.install else a.out
    if out == EXPORT_DIR and os.path.isdir(out):
        shutil.rmtree(out)                      # only ever clears our own staging dir
    export(out, pick_device(a.device))


if __name__ == "__main__":
    main()
