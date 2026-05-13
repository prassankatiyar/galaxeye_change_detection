from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out",  default="eda_outputs")
    p.add_argument("--max_per_split", type=int, default=200,
                   help="Sample at most N files per split for fast statistics.")
    return p.parse_args()


def explore_split(root: Path, split: str, max_n: int):
    pre_dir    = root / split / "pre-event"
    post_dir   = root / split / "post-event"
    target_dir = root / split / "target"
    if not pre_dir.exists():
        print(f"[skip] {split}: not found")
        return None

    files = sorted(pre_dir.glob("*.tif"))
    print(f"[{split}] total files: {len(files)}")
    files = files[:max_n]

    eo_shapes, sar_shapes, mask_shapes = [], [], []
    eo_dtypes, sar_dtypes              = Counter(), Counter()
    eo_minmax, sar_minmax              = [], []
    class_counts                       = Counter()
    pos_fractions                      = []

    for f in tqdm(files, desc=split):
        try:
            eo  = tifffile.imread(str(pre_dir / f.name))
            sar = tifffile.imread(str(post_dir / f.name))
            tgt = tifffile.imread(str(target_dir / f.name))
        except Exception as e:
            print(f"  ! could not read {f.name}: {e}")
            continue

        eo_shapes.append(eo.shape);  eo_dtypes[str(eo.dtype)]  += 1
        sar_shapes.append(sar.shape); sar_dtypes[str(sar.dtype)] += 1
        mask_shapes.append(tgt.shape)
        eo_minmax.append((float(eo.min()),  float(eo.max())))
        sar_minmax.append((float(sar.min()), float(sar.max())))

        for c in range(4):
            class_counts[c] += int((tgt == c).sum())


        bin_mask = (tgt >= 2)
        pos_fractions.append(float(bin_mask.mean()))

    return {
        "split": split,
        "n_files":              len(files),
        "eo_shape_examples":    list(set(map(tuple, eo_shapes)))[:5],
        "sar_shape_examples":   list(set(map(tuple, sar_shapes)))[:5],
        "mask_shape_examples":  list(set(map(tuple, mask_shapes)))[:5],
        "eo_dtypes":            dict(eo_dtypes),
        "sar_dtypes":           dict(sar_dtypes),
        "eo_value_range":       (min(v[0] for v in eo_minmax),  max(v[1] for v in eo_minmax)),
        "sar_value_range":      (min(v[0] for v in sar_minmax), max(v[1] for v in sar_minmax)),
        "class_pixel_counts":   {int(k): int(v) for k, v in class_counts.items()},
        "binary_positive_fraction_mean":   float(np.mean(pos_fractions)),
        "binary_positive_fraction_median": float(np.median(pos_fractions)),
        "binary_positive_fraction_p95":    float(np.percentile(pos_fractions, 95)),
        "pos_fractions": pos_fractions,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in ("train", "val", "test"):
        s = explore_split(root, split, args.max_per_split)
        if s is not None:
            summaries.append(s)

    serial = []
    for s in summaries:
        s2 = {k: v for k, v in s.items() if k != "pos_fractions"}
        s2["binary_positive_fraction_distribution_quantiles"] = {
            "p10": float(np.percentile(s["pos_fractions"], 10)),
            "p50": float(np.percentile(s["pos_fractions"], 50)),
            "p90": float(np.percentile(s["pos_fractions"], 90)),
        }
        serial.append(s2)
    json.dump(serial, open(out / "summary.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in summaries:
        ax.hist(s["pos_fractions"], bins=40, alpha=0.5, label=s["split"])
    ax.set_xlabel("Fraction of 'change' pixels per image")
    ax.set_ylabel("Image count")
    ax.set_title("Class imbalance: distribution of positive-class fraction per image")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "positive_fraction_hist.png", dpi=130)
    plt.close(fig)

    print("\n" + "=" * 72)
    print("  EXPLORATION SUMMARY")
    print("=" * 72)
    for s in summaries:
        print(f"\n[{s['split']}]   files sampled: {s['n_files']}")
        print(f"  EO  shapes: {s['eo_shape_examples']}    dtypes: {s['eo_dtypes']}    range: {s['eo_value_range']}")
        print(f"  SAR shapes: {s['sar_shape_examples']}   dtypes: {s['sar_dtypes']}   range: {s['sar_value_range']}")
        print(f"  Mask shapes: {s['mask_shape_examples']}")
        total = sum(s['class_pixel_counts'].values())
        print("  Class pixel fractions (4-class, original):")
        for k in sorted(s['class_pixel_counts']):
            print(f"     class {k}: {s['class_pixel_counts'][k] / total * 100:6.3f} %")
        print(f"  After binary remap, mean positive fraction = "
              f"{s['binary_positive_fraction_mean'] * 100:.3f} %")
        print(f"  Median = {s['binary_positive_fraction_median'] * 100:.3f} %  "
              f"P95 = {s['binary_positive_fraction_p95'] * 100:.3f} %")
    print(f"\n[done] full summary: {out / 'summary.json'}")
    print(f"[done] histogram:    {out / 'positive_fraction_hist.png'}")


if __name__ == "__main__":
    main()
