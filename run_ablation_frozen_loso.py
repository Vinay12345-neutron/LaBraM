#!/usr/bin/env python3
"""
Ablation 2 under Leave-One-Subject-Out (LOSO) Cross-Validation (73 Subjects)
- Frozen Pretrained LaBraM Backbone (5,798,936 frozen params) + Linear Probe (201 trainable params)
- Evaluates across all 73 subjects (1 subject held-out per fold)
- Zero subject-level leakage (65 train subjects, 7 validation subjects, 1 test subject per fold)
- Checkpoints selected strictly on validation loss (checkpoint-best.pth)
"""

import json
import csv
import re
import os
import time
import shutil
import random
import argparse
import subprocess
from pathlib import Path
import numpy as np

# Global Configuration
WORKDIR = Path.cwd()
SPLITS_DIR = WORKDIR / "loso_splits"
CKPT_PATH = WORKDIR / "checkpoints" / "labram-base.pth"
OUT_ROOT = WORKDIR / "runs" / "ablation_frozen_loso"
LOG_ROOT = WORKDIR / "log" / "ablation_frozen_loso"
MODEL = "labram_base_patch200_200"
INPUT_SIZE = 512
BATCH = 16
LR = 5e-4
SEED = 12345

def get_subject_id(path_str):
    name = Path(path_str).name
    m = re.match(r"S(\d+)_", name)
    return int(m.group(1)) if m else -1

def fold_is_complete(sub_id, epochs):
    out_dir = OUT_ROOT / f"S{sub_id}"
    log_file = out_dir / "log.txt"
    ckpt_best = out_dir / "checkpoint-best.pth"
    if not (log_file.exists() and ckpt_best.exists()):
        return False
    with open(log_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) >= epochs:
        try:
            last = json.loads(lines[-1])
            if last.get("epoch") == epochs - 1:
                return True
        except Exception:
            return False
    return False

def main():
    parser = argparse.ArgumentParser(description="Run Ablation 2 (Frozen Backbone Linear Probe) under 73-Fold LOSO CV")
    parser.add_argument("--epochs", type=int, default=24, help="Epochs per LOSO fold (default: 24)")
    parser.add_argument("--dry_run", action="store_true", help="Perform single dry-run fold check without full training")
    parser.add_argument("--start_from", type=int, default=0, help="Skip folds before this index (0-72)")
    parser.add_argument("--single_subject", type=int, default=-1, help="Run only for a single subject ID (e.g. 1)")
    args = parser.parse_args()

    print("=" * 95)
    print("STARTING ABLATION 2: FROZEN PRETRAINED LABRAM BACKBONE UNDER 73-FOLD LOSO CV")
    print("=" * 95)
    print(f"Model Architecture       : {MODEL} (~5.8M params)")
    print(f"Pretraining Checkpoint   : {CKPT_PATH}")
    print(f"Trainable Parameters     : 201 (head.weight: 200, head.bias: 1)")
    print(f"Frozen Parameters        : 5,798,936 (12 Transformer blocks + patch embeddings)")
    print(f"Evaluation Protocol      : Leave-One-Subject-Out (LOSO) across 73 Subjects")
    print(f"Epochs per Fold          : {args.epochs}")
    print(f"Batch Size               : {BATCH}")
    print(f"Learning Rate            : {LR} (AdamW, Cosine Annealing)")
    print(f"Random Seed              : {SEED}")
    print(f"Output Directory         : {OUT_ROOT}")
    print("=" * 95)

    assert CKPT_PATH.exists(), f"Pretrained checkpoint not found at: {CKPT_PATH}"
    assert SPLITS_DIR.exists(), f"Splits directory not found at: {SPLITS_DIR}"

    split_files = sorted(SPLITS_DIR.glob("loso_split_S*.json"), key=lambda p: int(re.search(r"S(\d+)", p.name).group(1)))
    assert len(split_files) == 73, f"Expected 73 split files, got {len(split_files)}"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for fold_idx, split_path in enumerate(split_files):
        sub_id = int(re.search(r"S(\d+)", split_path.name).group(1))

        if args.single_subject != -1 and sub_id != args.single_subject:
            continue

        if fold_idx < args.start_from and args.single_subject == -1:
            print(f"[SKIP] Fold {fold_idx + 1}/73 (Subject S{sub_id}) skipped by --start_from {args.start_from}")
            continue

        if fold_is_complete(sub_id, args.epochs):
            print(f"[SKIP] Fold {fold_idx + 1}/73 (Subject S{sub_id}) is ALREADY COMPLETED.")
            continue

        out_dir = OUT_ROOT / f"S{sub_id}"
        log_dir = LOG_ROOT / f"S{sub_id}"

        # Clean prior partial output
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(split_path) as f:
            split_data = json.load(f)

        print(f"\n>>> Launching LOSO Fold {fold_idx + 1}/73: Test Subject = S{sub_id}")
        print(f"    Train: {len(split_data['train'])} files (65 subjects) | Val: {len(split_data['val'])} files (7 subjects) | Test: {len(split_data['test'])} files (S{sub_id})")

        port = random.randint(31000, 35000)
        cmd = [
            "OMP_NUM_THREADS=1", "torchrun", "--nproc_per_node=1", f"--master_port={port}", "run_class_finetuning.py",
            "--output_dir", str(out_dir),
            "--log_dir", str(log_dir),
            "--model", MODEL,
            "--finetune", str(CKPT_PATH),
            "--freeze_backbone",
            "--dataset", "BOREDOM",
            "--input_size", str(INPUT_SIZE),
            "--batch_size", str(BATCH),
            "--lr", str(LR),
            "--epochs", str(args.epochs),
            "--seed", str(SEED),
            "--boredom_split", str(split_path),
            "--no_auto_resume"
        ]

        shell_cmd = " ".join(cmd)
        if args.dry_run:
            print(f"  [DRY RUN COMMAND]: {shell_cmd}")
            if fold_idx == 0:
                print("\n  Executing 1-step dry run for Fold 1 (S1)...")
                subprocess.run(shell_cmd, shell=True, check=True)
                print("\n  ✓ 1-Fold Dry Run Completed Successfully!")
                return
            continue

        print(f"Executing: {shell_cmd}")
        subprocess.run(shell_cmd, shell=True, check=True)
        time.sleep(2)

    print("\n" + "=" * 95)
    print("ALL 73 LOSO FOLDS COMPLETED FOR ABLATION 2!")
    print(f"Checkpoints and outputs saved in: {OUT_ROOT}")
    print("Run `python evaluate_ablation_frozen_loso.py` to extract full per-subject and pooled metrics.")
    print("=" * 95)

if __name__ == "__main__":
    main()
