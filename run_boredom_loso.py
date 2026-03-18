#!/usr/bin/env python3
"""
Leave-One-Subject-Out (LOSO) Cross-Validation for boredom vs neutral.
Each iteration: 1 subject for test, remaining for train (10% of train subjects used as val).
Writes JSON splits and launches run_class_finetuning.py per subject.

Usage:
    python run_boredom_loso.py [--epochs 5] [--dry_run]
"""
import json, os, subprocess, argparse, time
from pathlib import Path
import numpy as np
import re
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL = "labram_base_patch200_200"
FINETUNE = "./checkpoints/labram-base.pth"
INPUT_SIZE = 512
BATCH = 16
SEED = 12345
WORKDIR = Path.cwd()
BOREDOM_DIR = WORKDIR / "boredom_hdf5"
NEUTRAL_DIR = WORKDIR / "neutral_hdf5"
OUT_ROOT = WORKDIR / "runs" / "boredom_loso"
LOG_ROOT = WORKDIR / "log" / "boredom_loso"


def get_subject_id(path_str):
    name = Path(path_str).name
    m = re.match(r"S(\d+)_", name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs per LOSO fold")
    parser.add_argument("--dry_run", action="store_true", help="Only create splits, don't train")
    parser.add_argument("--start_from", type=int, default=0, help="Skip subjects before this index")
    args = parser.parse_args()

    # Collect all files
    boredom_files = sorted([str(p) for p in BOREDOM_DIR.glob("*.h5")])
    neutral_files = sorted([str(p) for p in NEUTRAL_DIR.glob("*.h5")])
    files = np.array(boredom_files + neutral_files)
    labels = np.array([1] * len(boredom_files) + [0] * len(neutral_files))
    groups = np.array([get_subject_id(f) for f in files])

    unique_subjects = sorted(set(groups))
    n_subjects = len(unique_subjects)
    print(f"Total files: {len(files)} ({len(boredom_files)} Boredom + {len(neutral_files)} Neutral)")
    print(f"Total subjects: {n_subjects}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Subject IDs: {unique_subjects}")

    logo = LeaveOneGroupOut()
    all_metrics = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(logo.split(files, labels, groups)):
        test_subject = groups[test_idx[0]]

        if fold_idx < args.start_from:
            print(f"Skipping fold {fold_idx} (subject S{test_subject})")
            continue

        print(f"\n{'='*60}")
        print(f"LOSO Fold {fold_idx}/{n_subjects-1}: Test Subject = S{test_subject}")
        print(f"{'='*60}")

        # Split remaining into train/val (by subject)
        train_val_files = files[train_val_idx]
        train_val_labels = labels[train_val_idx]
        train_val_groups = groups[train_val_idx]

        # Use 10% of remaining subjects as validation
        rng = np.random.RandomState(SEED + fold_idx)
        remaining_subjects = sorted(set(train_val_groups))
        n_val_subjects = max(1, int(0.1 * len(remaining_subjects)))
        val_subjects = set(rng.choice(remaining_subjects, size=n_val_subjects, replace=False))

        train_mask = np.array([g not in val_subjects for g in train_val_groups])
        val_mask = ~train_mask

        train_items = [{"file": str(train_val_files[i]), "label": int(train_val_labels[i])}
                       for i in range(len(train_val_files)) if train_mask[i]]
        val_items = [{"file": str(train_val_files[i]), "label": int(train_val_labels[i])}
                     for i in range(len(train_val_files)) if val_mask[i]]
        test_items = [{"file": str(files[i]), "label": int(labels[i])} for i in test_idx]

        print(f"  Train: {len(train_items)} files, Val: {len(val_items)} files, Test: {len(test_items)} files")
        print(f"  Val subjects: {sorted(val_subjects)}")

        # Write split JSON
        split = {"train": train_items, "val": val_items, "test": test_items}
        splits_dir = WORKDIR / "loso_splits"
        splits_dir.mkdir(exist_ok=True)
        split_path = splits_dir / f"loso_split_S{test_subject}.json"
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

        if args.dry_run:
            print(f"  [DRY RUN] Split saved to {split_path}")
            continue

        # Create output dirs
        out_dir = OUT_ROOT / f"S{test_subject}"
        log_dir = LOG_ROOT / f"S{test_subject}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Use unique port per fold to avoid EADDRINUSE (starting at 40000 to avoid defaults)
        master_port = 40000 + fold_idx
        cmd = [
            "OMP_NUM_THREADS=1", "torchrun",
            f"--master_port={master_port}",
            "--nproc_per_node=1", "run_class_finetuning.py",
            "--output_dir", str(out_dir),
            "--log_dir", str(log_dir),
            "--model", MODEL,
            "--finetune", FINETUNE,
            "--dataset", "BOREDOM",
            "--input_size", str(INPUT_SIZE),
            "--batch_size", str(BATCH),
            "--lr", "5e-4",
            "--epochs", str(args.epochs),
            "--seed", str(SEED),
            "--boredom_split", str(split_path)
        ]
        shell_cmd = " ".join(cmd)
        print(f"  CMD: {shell_cmd}")
        try:
            subprocess.run(shell_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Fold {fold_idx} (S{test_subject}) failed: {e}")
            time.sleep(3)
            continue

        # Brief delay to ensure port is fully released
        time.sleep(3)

        # Read metrics from log
        log_file = out_dir / "log.txt"
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
            if lines:
                try:
                    last = json.loads(lines[-1].strip())
                    fold_metrics = {
                        "fold": fold_idx,
                        "test_subject": int(test_subject),
                        "test_acc": last.get("test_accuracy"),
                        "test_roc_auc": last.get("test_roc_auc"),
                        "test_pr_auc": last.get("test_pr_auc"),
                        "test_precision": last.get("test_precision"),
                        "test_recall": last.get("test_recall"),
                        "test_f1": last.get("test_f1"),
                        "test_specificity": last.get("test_specificity"),
                    }
                    all_metrics.append(fold_metrics)
                    print(f"  Test Acc: {fold_metrics['test_acc']:.4f}, ROC AUC: {fold_metrics['test_roc_auc']:.4f}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Warning: Could not parse metrics for S{test_subject}: {e}")

    # Summary
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"LOSO SUMMARY ({len(all_metrics)} subjects completed)")
        print(f"{'='*60}")
        accs = [m["test_acc"] for m in all_metrics if m["test_acc"] is not None]
        aucs = [m["test_roc_auc"] for m in all_metrics if m["test_roc_auc"] is not None]
        f1s = [m["test_f1"] for m in all_metrics if m["test_f1"] is not None]

        if accs:
            print(f"  Accuracy:  {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        if aucs:
            print(f"  ROC AUC:   {np.mean(aucs)*100:.2f}% ± {np.std(aucs)*100:.2f}%")
        if f1s:
            print(f"  F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # Save summary
        summary_path = OUT_ROOT / "loso_summary.json"
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

        # Also save as CSV
        import csv
        csv_path = OUT_ROOT / "loso_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"  CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
