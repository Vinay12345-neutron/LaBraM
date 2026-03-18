#!/usr/bin/env python3
"""
Sanity Check Script for the Boredom Classification Pipeline.
Performs:
  1. Data integrity check (all H5 files readable, correct shape, no NaN/Inf)
  2. Label correctness (filename matches label in split JSON)
  3. Split integrity (subject disjointness across train/val/test)
  4. Permutation test (shuffle labels → retrain → accuracy should drop to ~50%)

Usage:
    python sanity_check.py [--skip_permutation]
"""
import json, os, re, argparse
import numpy as np
from pathlib import Path
import h5py


def check_data_integrity(boredom_dir, neutral_dir):
    """Check all H5 files are readable with correct shapes."""
    print("\n" + "="*60)
    print("1. DATA INTEGRITY CHECK")
    print("="*60)

    issues = []
    total_files = 0

    for label_name, directory in [("Boredom", boredom_dir), ("Neutral", neutral_dir)]:
        files = sorted(Path(directory).glob("*.h5"))
        print(f"  {label_name}: {len(files)} files")
        total_files += len(files)

        for fpath in files:
            try:
                with h5py.File(fpath, 'r') as f:
                    keys = list(f.keys())
                    if not keys:
                        issues.append(f"  EMPTY: {fpath}")
                        continue

                    obj = f[keys[0]]
                    if isinstance(obj, h5py.Group):
                        if 'eeg' in obj:
                            data = obj['eeg'][:]
                        else:
                            subkeys = list(obj.keys())
                            if subkeys:
                                data = obj[subkeys[0]][:]
                            else:
                                issues.append(f"  NO DATA: {fpath}")
                                continue
                    else:
                        data = obj[:]

                    # Check for NaN/Inf
                    if np.isnan(data).any():
                        issues.append(f"  NaN found: {fpath}")
                    if np.isinf(data).any():
                        issues.append(f"  Inf found: {fpath}")

                    # Check shape
                    if data.ndim != 2:
                        issues.append(f"  Wrong dims ({data.ndim}D): {fpath}")
                    elif min(data.shape) < 10:
                        issues.append(f"  Suspicious shape {data.shape}: {fpath}")

            except Exception as e:
                issues.append(f"  UNREADABLE: {fpath} ({e})")

    if issues:
        print(f"\n  ⚠ {len(issues)} issues found:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✓ All {total_files} files OK")

    return len(issues) == 0


def check_label_correctness(split_file):
    """Verify filenames match their assigned labels."""
    print("\n" + "="*60)
    print("2. LABEL CORRECTNESS CHECK")
    print("="*60)

    with open(split_file) as f:
        split = json.load(f)

    issues = []
    for split_name, items in split.items():
        if split_name not in ['train', 'val', 'test']:
            continue
        for item in items:
            fname = os.path.basename(item['file'])
            label = item['label']

            if 'Boredom' in fname and label != 1:
                issues.append(f"  {fname} labeled as {label} (should be 1)")
            elif 'Neutral' in fname and label != 0:
                issues.append(f"  {fname} labeled as {label} (should be 0)")

    if issues:
        print(f"\n  ⚠ {len(issues)} label mismatches:")
        for issue in issues:
            print(f"    {issue}")
    else:
        total = sum(len(split.get(s, [])) for s in ['train', 'val', 'test'])
        print(f"  ✓ All {total} labels correct (Boredom=1, Neutral=0)")

    return len(issues) == 0


def check_split_integrity():
    """Verify subject disjointness across all folds."""
    print("\n" + "="*60)
    print("3. SPLIT INTEGRITY CHECK (Subject Disjointness)")
    print("="*60)

    issues = []

    for fold in range(5):
        split_file = f"boredom_split_fold{fold}.json"
        if not os.path.exists(split_file):
            print(f"  Fold {fold}: Split file not found, skipping")
            continue

        with open(split_file) as f:
            split = json.load(f)

        def get_subjects(items):
            subjects = set()
            for item in items:
                m = re.search(r'S(\d+)_', os.path.basename(item['file']))
                if m:
                    subjects.add(int(m.group(1)))
            return subjects

        train_subjs = get_subjects(split.get('train', []))
        val_subjs = get_subjects(split.get('val', []))
        test_subjs = get_subjects(split.get('test', []))

        train_test_overlap = train_subjs & test_subjs
        train_val_overlap = train_subjs & val_subjs
        val_test_overlap = val_subjs & test_subjs

        if train_test_overlap:
            issues.append(f"  Fold {fold}: Train-Test overlap: {train_test_overlap}")
        if train_val_overlap:
            issues.append(f"  Fold {fold}: Train-Val overlap: {train_val_overlap}")
        if val_test_overlap:
            issues.append(f"  Fold {fold}: Val-Test overlap: {val_test_overlap}")

        n_total = len(train_subjs | val_subjs | test_subjs)
        print(f"  Fold {fold}: Train={len(train_subjs)}, Val={len(val_subjs)}, Test={len(test_subjs)}, Total={n_total} subjects")
        
        # Class balance check
        train_labels = [item['label'] for item in split.get('train', [])]
        test_labels = [item['label'] for item in split.get('test', [])]
        print(f"         Train: {train_labels.count(1)} Boredom / {train_labels.count(0)} Neutral")
        print(f"         Test:  {test_labels.count(1)} Boredom / {test_labels.count(0)} Neutral")

    if issues:
        print(f"\n  ⚠ {len(issues)} overlap issues:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✓ All folds have disjoint train/val/test subjects")

    return len(issues) == 0


def check_permutation(split_file):
    """Shuffle labels and verify accuracy drops (conceptual — prints command to run)."""
    print("\n" + "="*60)
    print("4. PERMUTATION TEST (Label Shuffle)")
    print("="*60)

    with open(split_file) as f:
        split = json.load(f)

    # Create shuffled version
    import copy
    shuffled = copy.deepcopy(split)
    rng = np.random.RandomState(999)

    for key in ['train', 'val', 'test']:
        if key in shuffled:
            labels = [item['label'] for item in shuffled[key]]
            rng.shuffle(labels)
            for i, item in enumerate(shuffled[key]):
                item['label'] = int(labels[i])

    perm_file = "boredom_split_permuted.json"
    with open(perm_file, "w") as f:
        json.dump(shuffled, f, indent=2)

    print(f"  Created shuffled split: {perm_file}")
    print(f"  To run permutation test, execute:")
    print(f"")
    print(f"    OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_class_finetuning.py \\")
    print(f"      --output_dir runs/permutation_test \\")
    print(f"      --log_dir log/permutation_test \\")
    print(f"      --model labram_base_patch200_200 \\")
    print(f"      --finetune ./checkpoints/labram-base.pth \\")
    print(f"      --dataset BOREDOM \\")
    print(f"      --input_size 512 \\")
    print(f"      --batch_size 16 \\")
    print(f"      --lr 5e-4 \\")
    print(f"      --epochs 3 \\")
    print(f"      --seed 12345 \\")
    print(f"      --boredom_split {perm_file}")
    print(f"")
    print(f"  Expected: Accuracy should be ~50% (random chance for binary).")
    print(f"  If accuracy is still high, there's likely a data leakage issue.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_permutation", action="store_true")
    parser.add_argument("--boredom_dir", default="boredom_hdf5")
    parser.add_argument("--neutral_dir", default="neutral_hdf5")
    parser.add_argument("--split", default="boredom_split_fold0.json")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║       BOREDOM CLASSIFICATION SANITY CHECK        ║")
    print("╚══════════════════════════════════════════════════╝")

    ok1 = check_data_integrity(args.boredom_dir, args.neutral_dir)
    ok2 = check_label_correctness(args.split)
    ok3 = check_split_integrity()

    if not args.skip_permutation:
        check_permutation(args.split)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    checks = [("Data Integrity", ok1), ("Label Correctness", ok2), ("Split Integrity", ok3)]
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")

    if all(ok for _, ok in checks):
        print("\n  ✓ ALL CHECKS PASSED")
    else:
        print("\n  ⚠ SOME CHECKS FAILED — review issues above")


if __name__ == "__main__":
    main()
