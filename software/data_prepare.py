"""Download Speech Commands v0.02, build the 12-class splits, cache raw audio
and compute MFCC normalisation statistics.

  python prepare_data.py               # real dataset (~2.3 GB download, once)
  python prepare_data.py --synthetic   # tiny tone dataset, pipeline smoke test only

Set KWS_DATA_ROOT to keep the dataset outside the repository, for example to
share one download between int4/ml and int8/ml.
"""

import argparse

from config import CACHE_DIR
from data import compute_feature_stats, prepare_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    prepare_all(synthetic=a.synthetic)
    mean, std = compute_feature_stats()
    print(f"cache: {CACHE_DIR}")
    print(f"MFCC mean range [{mean.min():.2f}, {mean.max():.2f}], "
          f"std range [{std.min():.2f}, {std.max():.2f}]")


if __name__ == "__main__":
    main()
