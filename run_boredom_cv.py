#!/usr/bin/env python3
"""
Orchestrate 5-fold subject-wise CV for boredom vs neutral.
Writes JSON splits and launches run_class_finetuning.py (torchrun) per fold.
"""
import json, os, subprocess
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

fold_id = 0
for test_idx, _ in skf.split(files, labels):
    # choose test fold = indices in test_idx for this iteration (we get n_splits rounds)
    # but sklearn's generator yields (train_idx, test_idx). We need to iterate differently:
    break

# Build splits by enumerating folds
folds = list(skf.split(files, labels))
for fold_id in range(CV_FOLDS):
    # test = fold_id's test indices
    _, test_idx = folds[fold_id]
    # val = next fold's test indices
    _, val_idx = folds[(fold_id + 1) % CV_FOLDS]
    # train = all remaining indices
    all_test_val = set(list(test_idx) + list(val_idx))
    train_idx = [i for i in range(len(files)) if i not in all_test_val]

    def build_pairs(idxs):
        return [{"file": str(files[i]), "label": int(labels[i])} for i in idxs]

    split = {"train": build_pairs(train_idx), "val": build_pairs(val_idx), "test": build_pairs(test_idx)}
    split_path = WORKDIR / f"boredom_split_fold{fold_id}.json"
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
            last = json.loads(lines[-1])
        with open(out_dir / "fold_metrics.json", "w") as f:
            json.dump({
                "fold": fold_id,
                "val_acc": last.get("val_accuracy"),
                "val_roc_auc": last.get("val_roc_auc"),
                "val_pr_auc": last.get("val_pr_auc"),
                "test_acc": last.get("test_accuracy"),
                "test_roc_auc": last.get("test_roc_auc"),
                "test_pr_auc": last.get("test_pr_auc"),
            }, f, indent=2)
