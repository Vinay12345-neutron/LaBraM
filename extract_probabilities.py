#!/usr/bin/env python3
"""
Extract and display per-epoch probability values from existing test_predictions.json files.
Also re-generates probability tables from the log.txt if test_predictions.json is missing.

Usage:
    python extract_probabilities.py [--fold 0] [--epochs 1 2 3]
"""
import json
import argparse
import os
import numpy as np
from pathlib import Path
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Which fold to analyze")
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 2],
                        help="Which epochs to show probabilities for (0-indexed)")
    parser.add_argument("--output_dir", type=str, default="runs/boredom_cv",
                        help="Root output directory")
    args = parser.parse_args()

    fold_dir = Path(args.output_dir) / f"fold{args.fold}"
    pred_file = fold_dir / "test_predictions.json"

    if not pred_file.exists():
        print(f"No test_predictions.json found at {pred_file}")
        print("This file is created during training by run_class_finetuning.py.")
        print("You need to re-run training to generate it, or check if the path is correct.")
        return

    # Parse JSONL format (one JSON per line, one per epoch)
    epoch_data = {}
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ep = d["epoch"]
                epoch_data[ep] = d
            except (json.JSONDecodeError, KeyError):
                continue

    available_epochs = sorted(epoch_data.keys())
    print(f"Fold {args.fold}: Available epochs with predictions: {available_epochs}")

    for ep in args.epochs:
        if ep not in epoch_data:
            print(f"\nEpoch {ep}: No predictions found (available: {available_epochs})")
            continue

        preds = np.array(epoch_data[ep]["preds"])
        trues = np.array(epoch_data[ep]["trues"])

        # Handle multi-dim preds (sometimes stored as [[0.98], [0.02], ...])
        if preds.ndim > 1:
            preds = preds.squeeze()

        pred_labels = (preds >= 0.5).astype(int)
        correct = (pred_labels == trues)

        print(f"\n{'='*70}")
        print(f"Epoch {ep} — File-level Predictions (Fold {args.fold})")
        print(f"{'='*70}")
        print(f"{'File':>6} | {'True':>5} | {'Pred Prob':>10} | {'Pred Label':>10} | {'Correct':>7}")
        print("-" * 55)
        for i in range(len(preds)):
            label_name = "BOR" if trues[i] == 1 else "NEU"
            status = "✓" if correct[i] else "✗"
            print(f"{i:>6} | {label_name:>5} | {preds[i]:>10.4f} | {pred_labels[i]:>10} | {status:>7}")

        acc = correct.mean()
        print(f"\nAccuracy: {acc*100:.2f}% ({correct.sum()}/{len(correct)})")

        # Save as CSV
        csv_path = fold_dir / f"epoch_{ep}_probabilities.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_idx", "true_label", "pred_prob", "pred_label", "correct"])
            for i in range(len(preds)):
                writer.writerow([i, int(trues[i]), float(preds[i]), int(pred_labels[i]), bool(correct[i])])
        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
