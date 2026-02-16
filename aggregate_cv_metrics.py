#!/usr/bin/env python3
"""
Aggregate per-fold metrics written by `run_boredom_cv.py` into a summary CSV and Markdown.
Writes: runs/boredom_cv/summary_metrics.csv and summary_metrics.md
"""
import json
import glob
import os
from statistics import mean, stdev
from math import isnan

OUT_DIR = os.path.join("runs", "boredom_cv")
GLOB = os.path.join(OUT_DIR, "fold*", "fold_metrics.json")

METRIC_KEYS = [
    "val_acc", "val_roc_auc", "val_pr_auc", "val_precision", "val_recall", "val_f1", "val_specificity",
    "test_acc", "test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_specificity",
]

files = sorted(glob.glob(GLOB))
if not files:
    print("No fold metrics found at:", GLOB)
    raise SystemExit(1)

rows = []
for p in files:
    with open(p) as f:
        d = json.load(f)
    row = {k: d.get(k) for k in METRIC_KEYS}
    row["fold"] = d.get("fold")
    rows.append(row)

# write CSV
csv_path = os.path.join(OUT_DIR, "summary_metrics.csv")
with open(csv_path, "w") as f:
    header = ["fold"] + METRIC_KEYS
    f.write(",".join(header) + "\n")
    for r in rows:
        vals = [str(r.get(c, "")) if r.get(c) is not None else "" for c in header]
        f.write(",".join(vals) + "\n")

# compute mean and std for each metric (ignore None)
agg = {}
for k in METRIC_KEYS:
    vals = [r[k] for r in rows if r.get(k) is not None]
    vals_num = [float(v) for v in vals if v is not None]
    if not vals_num:
        agg[k] = (None, None)
    elif len(vals_num) == 1:
        agg[k] = (mean(vals_num), 0.0)
    else:
        agg[k] = (mean(vals_num), stdev(vals_num))

# write Markdown summary table
md_path = os.path.join(OUT_DIR, "summary_metrics.md")
with open(md_path, "w") as f:
    # header
    f.write("| Metric | ")
    for r in rows:
        f.write(f"Fold {int(r['fold'])} | ")
    f.write("Mean ± Std |\n")
    # separator
    f.write("|---" * (2 + len(rows)) + "|\n")
    # for each metric
    for k in METRIC_KEYS:
        f.write(f"| {k} |")
        for r in rows:
            v = r.get(k)
            f.write(f" {'' if v is None else (f'{v:.4f}' if isinstance(v, (int, float)) else v)} |")
        mean_val, std_val = agg[k]
        if mean_val is None:
            f.write(" - |\n")
        else:
            f.write(f" {mean_val:.4f} ± {std_val:.4f} |\n")

print("Wrote:", csv_path, md_path)
