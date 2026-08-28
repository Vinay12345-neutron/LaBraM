#!/usr/bin/env python3
"""
Ablation 1: Pretraining vs. Random Initialization (From Scratch)
With intelligent resume:
- Skips fully completed folds (Fold 1 & Fold 2).
- Cleans up partial directories.
- Uses dynamic master port (29525) to avoid any port-binding collisions.
"""

import json
import os
import shutil
import random
import subprocess
from pathlib import Path
import numpy as np

# Configuration - STRICTLY IDENTICAL to Main Experiment
BATCH = 16
EPOCHS = 24
MODEL = "labram_base_patch200_200"
INPUT_SIZE = 512
WORKDIR = Path.cwd()
OUT_ROOT = WORKDIR / "runs" / "ablation_scratch"
LOG_ROOT = WORKDIR / "log" / "ablation_scratch"
CV_FOLDS = 5
SEED = 12345

def fold_is_complete(fold_id):
    log_file = OUT_ROOT / f"fold{fold_id}" / "log.txt"
    if not log_file.exists():
        return False
    with open(log_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) >= EPOCHS:
        try:
            last = json.loads(lines[-1])
            if last.get("epoch") == EPOCHS - 1:
                return True
        except Exception:
            return False
    return False

def main():
    print("=" * 80)
    print("RESUMING ABLATION 1: TRAINING LABRAM FROM RANDOM INITIALIZATION (FROM SCRATCH)")
    print("=" * 80)
    print(f"Model Architecture       : {MODEL} (~5.8M params)")
    print(f"Pretraining Checkpoint   : NONE (Random Initialization from Scratch)")
    print(f"Cross-Validation         : 5-Fold Subject-Disjoint (73 subjects)")
    print(f"Epochs per Fold          : {EPOCHS}")
    print(f"Batch Size               : {BATCH}")
    print(f"Learning Rate            : 5e-4 (Cosine Annealing with Layer Decay 0.9)")
    print(f"Random Seed              : {SEED}")
    print(f"Output Directory         : {OUT_ROOT}")
    print("=" * 80)

    for fold_id in range(CV_FOLDS):
        split_path = WORKDIR / f"boredom_split_fold{fold_id}.json"
        assert split_path.exists(), f"Split file not found: {split_path}"

        if fold_is_complete(fold_id):
            print(f"\n[SKIP] Fold {fold_id + 1} / {CV_FOLDS} is ALREADY FULLY COMPLETED (all {EPOCHS} epochs verified).")
            continue

        out_dir = OUT_ROOT / f"fold{fold_id}"
        log_dir = LOG_ROOT / f"fold{fold_id}"

        # Clean up any partial/interrupted files from previous crash so it starts cleanly
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(split_path, "r") as f:
            split = json.load(f)

        print(f"\n>>> Launching Fold {fold_id + 1} / {CV_FOLDS} (Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])})")

        port = random.randint(29510, 29990)
        cmd = [
            "OMP_NUM_THREADS=1", "torchrun", "--nproc_per_node=1", f"--master_port={port}", "run_class_finetuning.py",
            "--output_dir", str(out_dir),
            "--log_dir", str(log_dir),
            "--model", MODEL,
            "--dataset", "BOREDOM",
            "--input_size", str(INPUT_SIZE),
            "--batch_size", str(BATCH),
            "--lr", "5e-4",
            "--epochs", str(EPOCHS),
            "--seed", str(SEED),
            "--boredom_split", str(split_path),
            "--no_auto_resume"
        ]

        shell_cmd = " ".join(cmd)
        print(f"Executing: {shell_cmd}")
        subprocess.run(shell_cmd, shell=True, check=True)

    print("\n" + "=" * 80)
    print("ABLATION 1 TRAINING COMPLETE ACROSS ALL 5 FOLDS!")
    print(f"All outputs and checkpoints saved in: {OUT_ROOT}")
    print("Run `python evaluate_ablation_scratch.py` to compute full metrics and comparison.")
    print("=" * 80)

if __name__ == "__main__":
    main()
