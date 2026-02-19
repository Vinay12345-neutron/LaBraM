#!/usr/bin/env python3
"""
Orchestrate 5-fold subject-wise CV for boredom vs neutral.
Writes JSON splits and launches run_class_finetuning.py (torchrun) per fold.
"""
import json, os, subprocess
from pathlib import Path
import numpy as np
import re
from sklearn.model_selection import GroupKFold

# configuration
BATCH = 16
EPOCHS = 2
MODEL = "labram_base_patch200_200"
FINETUNE = "./checkpoints/labram-base.pth"
INPUT_SIZE = 512
WORKDIR = Path.cwd()
BOREDOM_DIR = WORKDIR / "boredom_hdf5"
NEUTRAL_DIR = WORKDIR / "neutral_hdf5"
OUT_ROOT = WORKDIR / "runs" / "boredom_cv"
LOG_ROOT = WORKDIR / "log" / "boredom_cv"
CV_FOLDS = 5
SEED = 12345

boredom_files = sorted([str(p) for p in BOREDOM_DIR.glob("*.h5")])
neutral_files = sorted([str(p) for p in NEUTRAL_DIR.glob("*.h5")])
files = np.array(boredom_files + neutral_files)
labels = np.array([1]*len(boredom_files) + [0]*len(neutral_files))

# Extract groups (Subject IDs)
# Assumes format "S<digits>_..." e.g. S11_Boredom.h5
def get_subject_id(path_str):
    name = Path(path_str).name
    m = re.match(r"S(\d+)_", name)
    if m:
        return int(m.group(1))
    return -1

groups = np.array([get_subject_id(f) for f in files])

# Use GroupKFold to strictly keep subjects together
gkf = GroupKFold(n_splits=CV_FOLDS)

# Build splits by enumerating folds
# GroupKFold.split(X, y, groups) yields (train_idx, test_idx)
folds = list(gkf.split(files, labels, groups=groups))

for fold_id in range(CV_FOLDS):
    # test = fold_id's test indices
    _, test_idx = folds[fold_id]
    
    # val = next fold's test indices (circularly)
    # Since GroupKFold creates non-overlapping test sets (groups), 
    # taking the next fold's test set as validation ensures it is also disjoint from current test set.
    _, val_idx = folds[(fold_id + 1) % CV_FOLDS]
    
    # train = all remaining indices
    # We remove both test_idx and val_idx from the full set
    all_test_val = set(list(test_idx) + list(val_idx))
    train_idx = [i for i in range(len(files)) if i not in all_test_val]

    def build_pairs(idxs):
        return [{"file": str(files[i]), "label": int(labels[i])} for i in idxs]

    split = {"train": build_pairs(train_idx), "val": build_pairs(val_idx), "test": build_pairs(test_idx)}
    split_path = WORKDIR / f"boredom_split_fold{fold_id}.json"
    
    print(f"Computed split for fold {fold_id}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    out_dir = OUT_ROOT / f"fold{fold_id}"
    log_dir = LOG_ROOT / f"fold{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "OMP_NUM_THREADS=1", "torchrun", "--nproc_per_node=1", "run_class_finetuning.py",
        "--output_dir", str(out_dir),
        "--log_dir", str(log_dir),
        "--model", MODEL,
        "--finetune", FINETUNE,
        "--dataset", "BOREDOM",
        "--input_size", str(INPUT_SIZE),
        "--batch_size", str(BATCH),
        "--lr", "5e-4",
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--boredom_split", str(split_path)
    ]
    # Join into a single shell command so env var applies
    shell_cmd = " ".join(cmd)
    print(f"Launching fold {fold_id}: {shell_cmd}")
    subprocess.run(shell_cmd, shell=True, check=True)
    
    # read best checkpoint log
    log_file = out_dir / "log.txt"
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            # Handle empty log or incomplete lines logic if needed
            if lines and lines[-1].strip():
                try:
                    last = json.loads(lines[-1])
                    fold_metrics = {
                        "fold": fold_id,
                        "val_acc": last.get("val_accuracy"),
                        "val_roc_auc": last.get("val_roc_auc"),
                        "val_pr_auc": last.get("val_pr_auc"),
                        "val_precision": last.get("val_precision"),
                        "val_recall": last.get("val_recall"),
                        "val_f1": last.get("val_f1"),
                        "val_specificity": last.get("val_specificity"),
                        "test_acc": last.get("test_accuracy"),
                        "test_roc_auc": last.get("test_roc_auc"),
                        "test_pr_auc": last.get("test_pr_auc"),
                        "test_precision": last.get("test_precision"),
                        "test_recall": last.get("test_recall"),
                        "test_f1": last.get("test_f1"),
                        "test_specificity": last.get("test_specificity"),
                    }
                    with open(out_dir / "fold_metrics.json", "w") as f:
                        json.dump(fold_metrics, f, indent=2)

                    # Print the fold metrics to stdout for quick inspection
                    print(f"Fold {fold_id} metrics:")
                    try:
                        print(json.dumps(fold_metrics, indent=2))
                    except Exception:
                        print(fold_metrics)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode last line of log for fold {fold_id}")
