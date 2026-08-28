#!/usr/bin/env python3
"""
Ablation 2: Frozen Pretrained LaBraM Backbone + Linear Probe (Linear Evaluation)
- Loads official pretrained checkpoint: checkpoints/labram-base.pth
- Freezes all 5.8M backbone parameters (all 12 Transformer blocks, patch embed, norms, positional biases)
- Trains ONLY the binary linear classification head (201 parameters: head.weight + head.bias)
- Runs identical 5-fold subject-disjoint GroupKFold CV across 73 subjects (18,396 test windows)
- Evaluates on held-out test subjects selected ONLY via minimum validation loss.
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
OUT_ROOT = WORKDIR / "runs" / "ablation_frozen"
LOG_ROOT = WORKDIR / "log" / "ablation_frozen"
FINETUNE_CKPT = WORKDIR / "checkpoints" / "labram-base.pth"
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
    print("=" * 85)
    print("STARTING ABLATION 2: FROZEN LABRAM BACKBONE + LINEAR PROBE (LINEAR EVALUATION)")
    print("=" * 85)
    print(f"Model Architecture       : {MODEL} (~5.8M params)")
    print(f"Pretraining Checkpoint   : {FINETUNE_CKPT}")
    print(f"Backbone Status          : FROZEN (requires_grad = False for all 12 blocks)")
    print(f"Trainable Head Status    : TRAINABLE (Linear(200, 1) -> 201 parameters)")
    print(f"Cross-Validation         : 5-Fold Subject-Disjoint (73 subjects, 18,396 test windows)")
    print(f"Epochs per Fold          : {EPOCHS}")
    print(f"Batch Size               : {BATCH}")
    print(f"Learning Rate            : 5e-4 (AdamW, Cosine Annealing)")
    print(f"Random Seed              : {SEED}")
    print(f"Output Directory         : {OUT_ROOT}")
    print("=" * 85)

    assert FINETUNE_CKPT.exists(), f"Pretrained checkpoint not found at: {FINETUNE_CKPT}"

    for fold_id in range(CV_FOLDS):
        split_path = WORKDIR / f"boredom_split_fold{fold_id}.json"
        assert split_path.exists(), f"Split file not found: {split_path}"

        if fold_is_complete(fold_id):
            print(f"\n[SKIP] Fold {fold_id + 1} / {CV_FOLDS} is ALREADY FULLY COMPLETED (all {EPOCHS} epochs logged).")
            continue

        out_dir = OUT_ROOT / f"fold{fold_id}"
        log_dir = LOG_ROOT / f"fold{fold_id}"

        # Clean partial run if interrupted
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(split_path, "r") as f:
            split = json.load(f)

        print(f"\n>>> Launching Fold {fold_id + 1} / {CV_FOLDS} (Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])})")

        port = random.randint(29530, 29990)
        cmd = [
            "OMP_NUM_THREADS=1", "torchrun", "--nproc_per_node=1", f"--master_port={port}", "run_class_finetuning.py",
            "--output_dir", str(out_dir),
            "--log_dir", str(log_dir),
            "--model", MODEL,
            "--finetune", str(FINETUNE_CKPT),
            "--freeze_backbone",
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

    print("\n" + "=" * 85)
    print("ABLATION 2 TRAINING COMPLETE ACROSS ALL 5 FOLDS!")
    print(f"All outputs and checkpoints saved in: {OUT_ROOT}")
    print("Run `python evaluate_ablation_frozen.py` to compute full metrics and comparison.")
    print("=" * 85)

if __name__ == "__main__":
    main()
