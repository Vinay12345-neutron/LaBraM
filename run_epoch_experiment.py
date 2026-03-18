#!/usr/bin/env python3
"""
Run epoch experiments: train fold 0 with different epoch counts and compare.
Reuses the existing boredom_split_fold0.json.

Usage:
    python run_epoch_experiment.py [--epochs_list 5 10 25 50]
"""
import subprocess, json, argparse
from pathlib import Path
import numpy as np

MODEL = "labram_base_patch200_200"
FINETUNE = "./checkpoints/labram-base.pth"
INPUT_SIZE = 512
BATCH = 16
SEED = 12345
SPLIT = "boredom_split_fold0.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_list", type=int, nargs="+", default=[5, 10, 25, 50],
                        help="List of epoch counts to try")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    split_file = f"boredom_split_fold{args.fold}.json"
    results = []

    for epochs in args.epochs_list:
        out_dir = Path(f"runs/epoch_exp/ep{epochs}")
        log_dir = Path(f"log/epoch_exp/ep{epochs}")
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
            "--epochs", str(epochs),
            "--seed", str(SEED),
            "--boredom_split", str(split_file)
        ]
        shell_cmd = " ".join(cmd)
        print(f"\n{'='*60}")
        print(f"Running with {epochs} epochs...")
        print(f"CMD: {shell_cmd}")
        print(f"{'='*60}")

        if args.dry_run:
            print("[DRY RUN] Skipping execution")
            continue

        try:
            subprocess.run(shell_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")
            continue

        # Parse results
        log_file = out_dir / "log.txt"
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
            # Find best epoch (highest val accuracy)
            best_val_acc = 0
            best_epoch_data = None
            for line in lines:
                try:
                    d = json.loads(line.strip())
                    if d.get("val_accuracy", 0) > best_val_acc:
                        best_val_acc = d["val_accuracy"]
                        best_epoch_data = d
                except:
                    pass
            if best_epoch_data:
                results.append({
                    "max_epochs": epochs,
                    "best_epoch": best_epoch_data.get("epoch"),
                    "val_acc": best_val_acc,
                    "test_acc": best_epoch_data.get("test_accuracy"),
                    "test_roc_auc": best_epoch_data.get("test_roc_auc"),
                    "test_f1": best_epoch_data.get("test_f1"),
                    "val_loss": best_epoch_data.get("val_loss"),
                })

    # Print summary
    if results:
        print(f"\n{'='*70}")
        print(f"EPOCH EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"{'MaxEp':>6} | {'BestEp':>6} | {'ValAcc':>8} | {'TestAcc':>8} | {'ROCAUC':>8} | {'ValLoss':>8}")
        print("-" * 60)
        for r in results:
            print(f"{r['max_epochs']:>6} | {r['best_epoch']:>6} | "
                  f"{r['val_acc']:.4f}   | {r['test_acc']:.4f}   | "
                  f"{r['test_roc_auc']:.4f}   | {r['val_loss']:.4f}")

        with open("runs/epoch_exp/epoch_experiment_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to runs/epoch_exp/epoch_experiment_summary.json")


if __name__ == "__main__":
    main()
