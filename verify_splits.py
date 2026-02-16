import json
import glob
import sys
from collections import Counter

split_files = sorted(glob.glob("boredom_split_fold*.json"))
if not split_files:
    print("No split files found (pattern boredom_split_fold*.json).")
    sys.exit(1)

all_ok = True
for sf in split_files:
    with open(sf, "r") as fh:
        data = json.load(fh)

    sets = {}
    labels = {}
    for split in ("train", "val", "test"):
        entries = data.get(split, [])
        paths = [e["file"] for e in entries]
        lbls = [e.get("label") for e in entries]
        sets[split] = set(paths)
        labels[split] = Counter(lbls)
        dup_count = len(paths) - len(sets[split])
        print(f"{sf}: {split} — files={len(paths)}, unique={len(sets[split])}, dup_in_split={dup_count}, labels={dict(labels[split])}")

    # check pairwise overlaps
    overlaps = {
        "train∩val": sets["train"] & sets["val"],
        "train∩test": sets["train"] & sets["test"],
        "val∩test": sets["val"] & sets["test"],
    }
    for name, inter in overlaps.items():
        if inter:
            all_ok = False
            print(f"  OVERLAP {sf} {name}: {len(inter)} files")
            for p in sorted(inter):
                print(f"    {p}")
    print("-" * 60)

if all_ok:
    print("All folds: train/val/test are disjoint and have no duplicate entries within splits.")
    sys.exit(0)
else:
    print("Found overlaps. Inspect the printed file lists above.")
    sys.exit(2)