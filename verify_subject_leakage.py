import json
import glob
import sys
import re

split_files = sorted(glob.glob("boredom_split_fold*.json"))
if not split_files:
    print("No split files found (pattern boredom_split_fold*.json).")
    sys.exit(1)

def get_subject_id(filename):
    m = re.match(r"S(\d+)_", filename.split("/")[-1])
    if m:
        return m.group(1)
    return None

all_ok = True
for sf in split_files:
    with open(sf, "r") as fh:
        data = json.load(fh)

    sets = {}
    subjects = {}
    for split in ("train", "val", "test"):
        entries = data.get(split, [])
        paths = [e["file"] for e in entries]
        subjs = set([get_subject_id(p) for p in paths if get_subject_id(p)])
        
        sets[split] = set(paths)
        subjects[split] = subjs
        print(f"{sf} {split}: {len(paths)} files, {len(subjs)} subjects")

    # check subject overlaps
    overlaps = {
        "train∩val": subjects["train"] & subjects["val"],
        "train∩test": subjects["train"] & subjects["test"],
        "val∩test": subjects["val"] & subjects["test"],
    }
    for name, inter in overlaps.items():
        if inter:
            all_ok = False
            print(f"  SUBJECT LEAKAGE {sf} {name}: {len(inter)} subjects")
            print(f"    {inter}")
    print("-" * 60)

if all_ok:
    print("SUCCESS: No subject leakage detected across all folds.")
else:
    print("FAILURE: Subject leakage detected.")
    sys.exit(1)
