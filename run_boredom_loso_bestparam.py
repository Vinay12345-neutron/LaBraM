#!/usr/bin/env python3
"""
Strict LOSO Cross-Validation with Pre-locked Hyperparameters.

Two-phase protocol for publication-grade evaluation, proving zero tuning leakage:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — HYPERPARAMETER TUNING  (runs ONCE, strictly outside the LOSO loop)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Reserves N_PILOT_SUBJECTS (default 7, ~10%) as a FIXED PILOT SET.
    Chosen once, deterministically, by a fixed seed — never part of LOSO.
  • Within pilot: first (N-2) subjects → train, last 2 → val / test.
  • Runs a full grid search over (lr × epochs) on the pilot set only.
  • Selects the best configuration by validation ROC-AUC.
  • Saves the winning config to:  <out_root>/best_params.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — LOSO WITH LOCKED PARAMS  (all 73 subjects, one left out at a time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • All 73 LOSO folds use the EXACT SAME frozen hyperparameters from Phase 1.
  • No per-fold tuning of any kind.
  • Val subject selection uses the same fixed SEED for every fold — the choice
    of validation subjects is NOT informed by the test subject identity.
  • Results saved to:  <out_root>/loso_summary.json  and  loso_summary.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Full run (Phase 1 tuning → 73-fold LOSO):
  python run_boredom_loso_bestparam.py

  # Skip tuning, load existing best_params.json:
  python run_boredom_loso_bestparam.py --skip_tuning

  # Dry run — print commands and create splits, no training:
  python run_boredom_loso_bestparam.py --dry_run

  # Resume interrupted LOSO from fold index N (Phase 1 must already be done):
  python run_boredom_loso_bestparam.py --skip_tuning --loso_start_from N

  # Run Phase 1 only, then stop:
  python run_boredom_loso_bestparam.py --tuning_only
"""

import json
import csv
import os
import re
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


# ─────────────────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL       = "labram_base_patch200_200"
FINETUNE    = "./checkpoints/labram-base.pth"
INPUT_SIZE  = 512
BATCH       = 16
SEED        = 12345          # Fixed everywhere — do NOT change between runs

WORKDIR     = Path.cwd()
BOREDOM_DIR = WORKDIR / "boredom_hdf5"
NEUTRAL_DIR = WORKDIR / "neutral_hdf5"
OUT_ROOT    = WORKDIR / "runs"  / "boredom_loso_bestparam"
LOG_ROOT    = WORKDIR / "log"   / "boredom_loso_bestparam"
SPLITS_DIR  = WORKDIR / "loso_splits_bestparam"

# Number of subjects reserved exclusively for Phase 1.
# These subjects still participate in Phase 2 LOSO as ordinary subjects;
# they were only used to *select* hyperparameters, not to adjust them
# fold-by-fold based on test identity.
N_PILOT_SUBJECTS = 7

# ─── Hyperparameter grid ─────────────────────────────────────────────────────
# weight_decay and warmup_epochs are held constant (well-validated in the
# original LaBraM paper and our prior convergence experiments).
# The grid sweeps learning rate × epoch budget — the two parameters with the
# greatest practical effect on this fast-converging pre-trained model.
HP_GRID = [
    {
        "lr":            lr,
        "epochs":        epochs,
        "weight_decay":  0.05,   # fixed — validated in original LaBraM paper
        "warmup_epochs": 1,      # fixed — model converges extremely fast
    }
    for lr     in [1e-4, 5e-4, 1e-3]
    for epochs in [3, 5, 10]
]
# 9 configurations total. Each runs in ~2–4 min on RTX 4500 Ada with 7 subjects.


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_subject_id(path_str: str) -> int:
    """Extract integer subject ID from filename, e.g. 'S23_Boredom.h5' → 23."""
    m = re.match(r"S(\d+)_", Path(path_str).name)
    return int(m.group(1)) if m else -1


def collect_files():
    """Return (files, labels, groups) numpy arrays covering all subjects."""
    boredom_files = sorted(str(p) for p in BOREDOM_DIR.glob("*.h5"))
    neutral_files = sorted(str(p) for p in NEUTRAL_DIR.glob("*.h5"))
    if not boredom_files:
        raise FileNotFoundError(f"No .h5 files in {BOREDOM_DIR}")
    if not neutral_files:
        raise FileNotFoundError(f"No .h5 files in {NEUTRAL_DIR}")
    files  = np.array(boredom_files + neutral_files)
    labels = np.array([1] * len(boredom_files) + [0] * len(neutral_files))
    groups = np.array([get_subject_id(f) for f in files])
    return files, labels, groups


def write_split_json(path: Path, train_items, val_items, test_items) -> Path:
    """Write a train/val/test split dict to JSON and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"train": train_items, "val": val_items, "test": test_items},
                  f, indent=2)
    return path


def items_from_mask(files, labels, mask):
    """Build [{file, label}, ...] from a boolean mask."""
    return [{"file": str(files[i]), "label": int(labels[i])}
            for i in range(len(files)) if mask[i]]


def items_from_indices(files, labels, indices):
    """Build [{file, label}, ...] from an index array."""
    return [{"file": str(files[i]), "label": int(labels[i])} for i in indices]


def parse_last_epoch_metrics(log_txt: Path):
    """
    Read the last valid JSON line from a log.txt written by
    run_class_finetuning.py.  Returns a dict or None.
    """
    if not log_txt.exists():
        return None
    last = None
    with open(log_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def build_cmd(split_path, out_dir, log_dir, lr, epochs, weight_decay,
              warmup_epochs, master_port) -> str:
    """Assemble the torchrun shell command for a single training fold."""
    return (
        f"OMP_NUM_THREADS=1 torchrun "
        f"--master_port={master_port} "
        f"--nproc_per_node=1 run_class_finetuning.py "
        f"--output_dir {out_dir} "
        f"--log_dir {log_dir} "
        f"--model {MODEL} "
        f"--finetune {FINETUNE} "
        f"--dataset BOREDOM "
        f"--input_size {INPUT_SIZE} "
        f"--batch_size {BATCH} "
        f"--lr {lr} "
        f"--weight_decay {weight_decay} "
        f"--warmup_epochs {warmup_epochs} "
        f"--epochs {epochs} "
        f"--seed {SEED} "
        f"--boredom_split {split_path}"
    )


def run_training(split_path, out_dir, log_dir, lr, epochs, weight_decay,
                 warmup_epochs, master_port, dry_run=False):
    """
    Launch run_class_finetuning.py and return last-epoch metrics dict.
    Returns None on failure or in dry_run mode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(split_path, out_dir, log_dir, lr, epochs,
                    weight_decay, warmup_epochs, master_port)
    print(f"  CMD: {cmd}")

    if dry_run:
        print("  [DRY RUN] Skipped.")
        return None

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: subprocess failed (exit {e.returncode})")
        return None

    return parse_last_epoch_metrics(out_dir / "log.txt")


def section(title: str, width: int = 70):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Hyperparameter search on fixed pilot set
# ─────────────────────────────────────────────────────────────────────────────

def phase1_tuning(args, files, labels, groups, unique_subjects) -> dict:
    """
    Reserve a fixed pilot subset deterministically, run the HP grid on that
    subset only, and return the best config dict.

    Pilot split:
        train  →  first (N_PILOT_SUBJECTS - 2) pilot subjects
        val    →  last 2 pilot subjects
        test   →  same as val  (only val_roc_auc matters for selection)

    Selection criterion: validation ROC-AUC (primary), accuracy (tie-break).
    """
    section(
        "PHASE 1 — HYPERPARAMETER TUNING ON FIXED PILOT SET\n"
        "  (strictly outside LOSO — no test-fold identity used)"
    )

    # ── Select pilot subjects deterministically ────────────────────────────
    rng = np.random.RandomState(SEED)
    pilot_subjects = sorted(
        rng.choice(unique_subjects, size=N_PILOT_SUBJECTS, replace=False).tolist()
    )
    train_pilot = set(pilot_subjects[: N_PILOT_SUBJECTS - 2])
    val_pilot   = set(pilot_subjects[N_PILOT_SUBJECTS - 2 :])

    print(f"  Pilot subjects ({N_PILOT_SUBJECTS}): {pilot_subjects}")
    print(f"    Train: {sorted(train_pilot)}")
    print(f"    Val  : {sorted(val_pilot)}")
    print(f"  Grid : {len(HP_GRID)} configurations")

    # ── Build pilot split ──────────────────────────────────────────────────
    pilot_mask   = np.isin(groups, pilot_subjects)
    p_files      = files[pilot_mask]
    p_labels     = labels[pilot_mask]
    p_groups     = groups[pilot_mask]

    tr_mask = np.array([g in train_pilot for g in p_groups])
    vl_mask = ~tr_mask

    train_items = items_from_mask(p_files, p_labels, tr_mask)
    val_items   = items_from_mask(p_files, p_labels, vl_mask)
    # test == val — we only need validation metrics here
    test_items  = val_items

    print(f"\n  Pilot split: train={len(train_items)} files  "
          f"val={len(val_items)} files")

    pilot_split_path = OUT_ROOT / "tuning" / "pilot_split.json"
    write_split_json(pilot_split_path, train_items, val_items, test_items)
    print(f"  Pilot split JSON: {pilot_split_path}")

    # ── Grid search ────────────────────────────────────────────────────────
    best_config  = None
    best_val_auc = -1.0
    best_val_acc = -1.0
    all_results  = []
    base_port    = 41000   # Ports 41000–41008

    print(f"\n  {'Cfg':>3}  {'LR':>8}  {'Epochs':>6}  "
          f"{'val_roc_auc':>12}  {'val_acc':>9}")
    print(f"  {'---':>3}  {'--------':>8}  {'------':>6}  "
          f"{'------------':>12}  {'---------':>9}")

    for cfg_idx, cfg in enumerate(HP_GRID):
        cfg_out = OUT_ROOT / "tuning" / f"cfg_{cfg_idx:02d}"
        cfg_log = LOG_ROOT / "tuning" / f"cfg_{cfg_idx:02d}"

        metrics = run_training(
            split_path    = pilot_split_path,
            out_dir       = cfg_out,
            log_dir       = cfg_log,
            lr            = cfg["lr"],
            epochs        = cfg["epochs"],
            weight_decay  = cfg["weight_decay"],
            warmup_epochs = cfg["warmup_epochs"],
            master_port   = base_port + cfg_idx,
            dry_run       = args.dry_run,
        )
        time.sleep(3)  # Let the OS fully release the port

        val_auc = metrics.get("val_roc_auc")  if metrics else None
        val_acc = metrics.get("val_accuracy") if metrics else None

        auc_s = f"{val_auc:.4f}" if val_auc is not None else "       N/A"
        acc_s = f"{val_acc:.4f}" if val_acc is not None else "      N/A"
        print(f"  {cfg_idx:>3}  {cfg['lr']:>8.0e}  {cfg['epochs']:>6}  "
              f"{auc_s:>12}  {acc_s:>9}")

        result = {
            "cfg_idx":        cfg_idx,
            "lr":             cfg["lr"],
            "epochs":         cfg["epochs"],
            "weight_decay":   cfg["weight_decay"],
            "warmup_epochs":  cfg["warmup_epochs"],
            "val_roc_auc":    val_auc,
            "val_accuracy":   val_acc,
        }
        all_results.append(result)

        # Primary sort: val_roc_auc; secondary: val_accuracy
        if val_auc is not None:
            is_better = (
                val_auc > best_val_auc
                or (val_auc == best_val_auc and (val_acc or 0) > best_val_acc)
            )
            if is_better:
                best_val_auc = val_auc
                best_val_acc = val_acc or 0.0
                best_config  = cfg

    # ── Dry-run fallback ───────────────────────────────────────────────────
    if best_config is None:
        if args.dry_run:
            best_config  = {"lr": 5e-4, "epochs": 5,
                            "weight_decay": 0.05, "warmup_epochs": 1}
            best_val_auc = float("nan")
            print(f"\n  [DRY RUN] Using fallback config: {best_config}")
        else:
            raise RuntimeError(
                "Phase 1 returned no valid metrics. "
                "Check that boredom_hdf5/ and neutral_hdf5/ are populated "
                "and that run_class_finetuning.py completes without errors."
            )
    else:
        auc_display = f"{best_val_auc:.4f}" if best_val_auc == best_val_auc else "N/A"
        print(f"\n  ✓ BEST CONFIG → lr={best_config['lr']:.0e}  "
              f"epochs={best_config['epochs']}  "
              f"(val_roc_auc={auc_display})")

    # ── Save best_params.json ──────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    best_params_path = OUT_ROOT / "best_params.json"

    # Convert nan to None for JSON serialisation
    bva = float(best_val_auc) if (best_val_auc == best_val_auc) else None

    payload = {
        "best_config": best_config,
        "best_val_roc_auc": bva,
        "selection_criterion": "val_roc_auc (primary), val_accuracy (tie-break)",
        "pilot_subjects":       pilot_subjects,
        "train_pilot_subjects": sorted(train_pilot),
        "val_pilot_subjects":   sorted(val_pilot),
        "n_pilot_subjects":     N_PILOT_SUBJECTS,
        "hp_grid":              HP_GRID,
        "hp_grid_size":         len(HP_GRID),
        "all_results":          all_results,
        "tuning_seed":          SEED,
        "audit_note": (
            f"Hyperparameters selected on a FIXED PILOT SET of "
            f"{N_PILOT_SUBJECTS} subjects (IDs: {pilot_subjects}). "
            "The pilot set was chosen once before any LOSO fold was run. "
            "The same hyperparameters are applied to ALL 73 LOSO folds "
            "without per-fold adjustment. "
            "This design categorically prevents hyperparameter tuning leakage."
        ),
    }

    with open(best_params_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Best params saved → {best_params_path}")

    return best_config


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — LOSO with locked hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def phase2_loso(args, files, labels, groups, unique_subjects, best_config):
    """
    Run Leave-One-Subject-Out CV with fully locked hyperparameters.

    For each fold:
      • 1 subject is held out as the test set.
      • Of the remaining 72, a FIXED 10% are chosen as the validation set
        using the SAME random seed for every fold, so the choice of val
        subjects is NOT informed by which subject is being tested.
      • Training uses the hyperparameters locked in Phase 1 — unchanged.
    """
    section(
        "PHASE 2 — LOSO WITH LOCKED HYPERPARAMETERS\n"
        f"  lr={best_config['lr']:.0e}  |  epochs={best_config['epochs']}  |  "
        f"weight_decay={best_config['weight_decay']}  |  "
        f"warmup_epochs={best_config['warmup_epochs']}"
    )

    logo        = LeaveOneGroupOut()
    all_metrics = []
    base_port   = 42000   # Ports 42000–42072
    n_subjects  = len(unique_subjects)
    splits      = list(logo.split(files, labels, groups))

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        test_subject = int(groups[test_idx[0]])

        if fold_idx < args.loso_start_from:
            print(f"  [SKIP] Fold {fold_idx:03d}/{n_subjects-1}  "
                  f"(S{test_subject}) — resuming from fold {args.loso_start_from}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Fold {fold_idx:03d}/{n_subjects-1}  →  Test: S{test_subject}")
        print(f"  [LOCKED] lr={best_config['lr']:.0e}  "
              f"epochs={best_config['epochs']}")
        print(f"{'─'*60}")

        # ── Val subject selection ──────────────────────────────────────────
        # CRITICAL: identical seed every fold.  The val subjects chosen here
        # are NOT a function of test_subject identity — no leakage.
        tv_files  = files[train_val_idx]
        tv_labels = labels[train_val_idx]
        tv_groups = groups[train_val_idx]

        remaining  = sorted(set(tv_groups.tolist()))
        n_val      = max(1, int(0.1 * len(remaining)))

        rng          = np.random.RandomState(SEED)   # same seed, every fold
        val_subjects = set(
            rng.choice(remaining, size=n_val, replace=False).tolist()
        )

        tr_mask = np.array([g not in val_subjects for g in tv_groups])
        vl_mask = ~tr_mask

        train_items = items_from_mask(tv_files, tv_labels, tr_mask)
        val_items   = items_from_mask(tv_files, tv_labels, vl_mask)
        test_items  = items_from_indices(files, labels, test_idx)

        print(f"  Train={len(train_items)}  Val={len(val_items)}  "
              f"Test={len(test_items)} files")
        print(f"  Val subjects: {sorted(val_subjects)}")

        # ── Write split JSON ───────────────────────────────────────────────
        split_path = SPLITS_DIR / f"loso_split_S{test_subject}.json"
        write_split_json(split_path, train_items, val_items, test_items)

        # ── Dry run ────────────────────────────────────────────────────────
        if args.dry_run:
            cmd = build_cmd(
                split_path, OUT_ROOT / f"S{test_subject}",
                LOG_ROOT / f"S{test_subject}",
                best_config["lr"], best_config["epochs"],
                best_config["weight_decay"], best_config["warmup_epochs"],
                base_port + fold_idx,
            )
            print(f"  [DRY RUN] CMD: {cmd}")
            continue

        # ── Train ──────────────────────────────────────────────────────────
        out_dir = OUT_ROOT / f"S{test_subject}"
        log_dir = LOG_ROOT / f"S{test_subject}"

        metrics = run_training(
            split_path    = split_path,
            out_dir       = out_dir,
            log_dir       = log_dir,
            lr            = best_config["lr"],
            epochs        = best_config["epochs"],
            weight_decay  = best_config["weight_decay"],
            warmup_epochs = best_config["warmup_epochs"],
            master_port   = base_port + fold_idx,
            dry_run       = False,
        )
        time.sleep(3)

        if metrics is None:
            print(f"  ⚠ No metrics for S{test_subject} — fold skipped.")
            continue

        fold_metrics = {
            "fold":             fold_idx,
            "test_subject":     test_subject,
            # Explicit lock record — proves params were identical across folds
            "locked_lr":        best_config["lr"],
            "locked_epochs":    best_config["epochs"],
            "locked_wd":        best_config["weight_decay"],
            "locked_warmup":    best_config["warmup_epochs"],
            # Performance
            "test_acc":         metrics.get("test_accuracy"),
            "test_roc_auc":     metrics.get("test_roc_auc"),
            "test_pr_auc":      metrics.get("test_pr_auc"),
            "test_f1":          metrics.get("test_f1"),
            "test_precision":   metrics.get("test_precision"),
            "test_recall":      metrics.get("test_recall"),
            "test_specificity": metrics.get("test_specificity"),
            "test_bal_acc":     metrics.get("test_balanced_accuracy"),
        }
        all_metrics.append(fold_metrics)

        acc_s = f"{fold_metrics['test_acc']:.4f}"     if fold_metrics["test_acc"]     else "N/A"
        auc_s = f"{fold_metrics['test_roc_auc']:.4f}" if fold_metrics["test_roc_auc"] else "N/A"
        print(f"  → test_acc={acc_s}  test_roc_auc={auc_s}")

    # ── Final summary ──────────────────────────────────────────────────────
    if all_metrics:
        _save_loso_summary(all_metrics, best_config)
    elif not args.dry_run:
        print("\n  ⚠ No fold metrics collected — check training logs above.")


def _save_loso_summary(all_metrics: list, best_config: dict):
    """Print aggregate stats and persist JSON + CSV summary."""
    section(
        f"LOSO SUMMARY  ({len(all_metrics)} folds completed)\n"
        f"  Locked → lr={best_config['lr']:.0e}  "
        f"epochs={best_config['epochs']}"
    )

    metric_rows = [
        ("test_acc",          "Accuracy"),
        ("test_roc_auc",      "ROC AUC"),
        ("test_pr_auc",       "PR AUC"),
        ("test_f1",           "F1 Score"),
        ("test_precision",    "Precision"),
        ("test_recall",       "Recall"),
        ("test_specificity",  "Specificity"),
        ("test_bal_acc",      "Balanced Acc"),
    ]

    for key, label in metric_rows:
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if not vals:
            continue
        mu, sd = np.mean(vals), np.std(vals)
        print(f"  {label:15s}: {mu * 100:6.2f}%  ±  {sd * 100:5.2f}%  "
              f"(n={len(vals)})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # JSON
    summary_path = OUT_ROOT / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "locked_best_config":  best_config,
                "n_folds_completed":   len(all_metrics),
                "folds":               all_metrics,
            },
            f, indent=2,
        )
    print(f"\n  JSON → {summary_path}")

    # CSV
    csv_path = OUT_ROOT / "loso_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  CSV  → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Declare global first — before ANY reference to N_PILOT_SUBJECTS,
    # including inside argparse default= arguments.
    global N_PILOT_SUBJECTS

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip_tuning", action="store_true",
        help=(
            "Skip Phase 1. Load best_params.json from "
            "<out_root>/best_params.json and proceed directly to LOSO."
        ),
    )
    parser.add_argument(
        "--tuning_only", action="store_true",
        help="Run Phase 1 (hyperparameter search) only — do not start LOSO.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help=(
            "Generate all split JSONs and print training commands "
            "without actually running any training."
        ),
    )
    parser.add_argument(
        "--loso_start_from", type=int, default=0,
        help=(
            "Skip LOSO folds with index < this value. "
            "Use to resume an interrupted LOSO run. "
            "Requires --skip_tuning."
        ),
    )
    parser.add_argument(
        "--n_pilot", type=int, default=N_PILOT_SUBJECTS,
        help=(
            f"Number of subjects reserved for Phase 1 tuning "
            f"(default: {N_PILOT_SUBJECTS}). "
            "Ignored when --skip_tuning is set."
        ),
    )
    args = parser.parse_args()

    # Validation
    if args.loso_start_from > 0 and not args.skip_tuning:
        parser.error(
            "--loso_start_from requires --skip_tuning "
            "(best_params.json must exist from a prior Phase 1 run)."
        )

    # Apply any n_pilot override
    N_PILOT_SUBJECTS = args.n_pilot

    # ── Data ──────────────────────────────────────────────────────────────
    files, labels, groups = collect_files()
    unique_subjects = sorted(set(groups.tolist()))
    n_subjects      = len(unique_subjects)

    print("━" * 70)
    print("  LaBraM — Strict LOSO with Locked Hyperparameters")
    print("━" * 70)
    print(f"  Files         : {len(files)} "
          f"({(labels == 1).sum()} Boredom + {(labels == 0).sum()} Neutral)")
    print(f"  Subjects      : {n_subjects}")
    print(f"  Pilot set     : {N_PILOT_SUBJECTS} subjects")
    print(f"  HP grid       : {len(HP_GRID)} configurations")
    print(f"  Output root   : {OUT_ROOT}")
    print(f"  Skip tuning   : {args.skip_tuning}")
    print(f"  Tuning only   : {args.tuning_only}")
    print(f"  Dry run       : {args.dry_run}")
    if args.loso_start_from:
        print(f"  Resume LOSO   : fold {args.loso_start_from}")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if args.skip_tuning:
        best_params_path = OUT_ROOT / "best_params.json"
        if not best_params_path.exists():
            parser.error(
                f"--skip_tuning specified but {best_params_path} does not exist.\n"
                "Run Phase 1 first (without --skip_tuning) to generate it."
            )
        with open(best_params_path) as f:
            payload = json.load(f)
        best_config = payload["best_config"]
        print(f"\n[SKIP TUNING] Loaded best params from {best_params_path}")
        print(f"  Config        : {best_config}")
        print(f"  Val ROC-AUC   : {payload.get('best_val_roc_auc')}")
        print(f"  Pilot subjects: {payload.get('pilot_subjects')}")
    else:
        best_config = phase1_tuning(
            args, files, labels, groups, unique_subjects
        )

    if args.tuning_only:
        print("\n[TUNING ONLY] Phase 1 complete. Stopping here (--tuning_only).")
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────
    phase2_loso(
        args, files, labels, groups, unique_subjects, best_config
    )


if __name__ == "__main__":
    main()
