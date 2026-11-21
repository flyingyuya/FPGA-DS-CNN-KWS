#!/usr/bin/env python3
"""Regenerate every preprocessing ROM and golden vector, in dependency order.

    uv run --project software python scripts/golden/gen_all.py

Why a driver instead of "just run them": the generators form a chain --

    fft twiddle ROM                     (depends on N_FFT only)
    gen_fft_pow_vectors  -> golden/fft/expfft_*, expmel_*      (mel band vectors)
    gen_log_lut          -> golden/log/explog_*                 reads expmel_*
    gen_dct_rom          -> mem/preprocessing/dct_coef.mem,
                            golden/dct/expdct_*                  reads explog_*
    gen_mel_rom          -> rtl/preprocessing/mel_rom.v          (filter spans)
    gen_mfcc_golden      -> mem/preprocessing/hann_q15.mem,
                            mem/preprocessing/mfcc_norm.mem,
                            golden/mfcc/in_*, exp_*, pw_*         reads all of the above

Run them out of order and you silently get golden vectors computed from the
*previous* configuration -- which is exactly what happened when the mel-band
count changed from 40 to 20 and ``gen_log_lut`` ran before
``gen_fft_pow_vectors``: LOG_Q8 failed across the board with no clue why.

After running this, rebuild nothing in the accelerator: ``software/export_mem.py
--install`` owns ``rtl/kws_params.vh``, ``mem/`` for the CNN and ``golden/`` for
the CNN vectors.  Then run ``bash sim/run.sh all``.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# (script, what it produces, does anything downstream read it?)
STEPS = [
    ("gen_fft_twiddle.py",      "rtl/preprocessing/fft_twiddle_rom.v", False),
    ("gen_fft_pow_vectors.py",  "golden/fft/ (expfft_*, expmel_*)",    False),
    ("gen_log_lut.py",          "rtl/preprocessing/log_lut.v, golden/log/", True),
    ("gen_dct_rom.py",          "mem/preprocessing/dct_coef.mem, golden/dct/", True),
    ("gen_mel_rom.py",          "rtl/preprocessing/mel_rom.v",         False),
    ("gen_mfcc_golden.py",      "mem/preprocessing/{hann,norm}, golden/mfcc/", True),
]


def main():
    py = sys.executable
    for script, produces, chained in STEPS:
        path = os.path.join(HERE, script)
        if not os.path.isfile(path):
            print(f"  MISSING {script}")
            return 1
        tag = "chain" if chained else "     "
        print(f"  [{tag}] {script:24s} -> {produces}")
        r = subprocess.run([py, path], cwd=ROOT)
        if r.returncode != 0:
            print(f"\n  FAILED at {script} (exit {r.returncode}).")
            print("  Note the order: a failure here usually means an EARLIER step "
                  "did not run, not that this one is broken.")
            return r.returncode
    print("\n  all preprocessing ROMs and golden vectors regenerated.")
    print("  next: bash sim/run.sh all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
