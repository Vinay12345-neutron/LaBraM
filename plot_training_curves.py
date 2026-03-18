#!/usr/bin/env python3
"""
Plot training curves (loss and accuracy) from existing training logs.
Parses log.txt files in runs/boredom_cv/foldX/ directories.

Usage:
    python plot_training_curves.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_ROOT = Path("runs/boredom_cv")
FOLDS = 5


def parse_log(log_path):
    """Parse a log.txt file and return per-epoch metrics."""
    epochs = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = d.get("epoch")
            if ep is None:
                continue
            # Keep the LAST entry per epoch (most refined sub-epoch eval)
            epochs[ep] = d
    # Sort by epoch
    sorted_epochs = sorted(epochs.items())
    return [v for _, v in sorted_epochs]


def plot_fold(fold_id, data, save_dir):
    """Plot loss and accuracy curves for a single fold."""
    epochs = [d["epoch"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    if "train_loss" in data[0]:
        ax.plot(epochs, [d["train_loss"] for d in data], 'o-', label="Train Loss", color="#2196F3")
    if "val_loss" in data[0]:
        ax.plot(epochs, [d.get("val_loss", np.nan) for d in data], 's-', label="Val Loss", color="#FF9800")
    if "test_loss" in data[0]:
        ax.plot(epochs, [d.get("test_loss", np.nan) for d in data], '^-', label="Test Loss", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Fold {fold_id} — Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    train_acc_key = "train_window_acc" if "train_window_acc" in data[0] else "train_class_acc"
    if train_acc_key in data[0]:
        ax.plot(epochs, [d.get(train_acc_key, np.nan) for d in data], 'o-', label="Train Acc (window)", color="#2196F3")
    if "val_accuracy" in data[0]:
        ax.plot(epochs, [d.get("val_accuracy", np.nan) for d in data], 's-', label="Val Acc (file)", color="#FF9800")
    if "test_accuracy" in data[0]:
        ax.plot(epochs, [d.get("test_accuracy", np.nan) for d in data], '^-', label="Test Acc (file)", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Fold {fold_id} — Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    out_path = save_dir / f"training_curves_fold{fold_id}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_all_folds(all_data, save_dir):
    """Plot averaged curves across all folds."""
    # Find common epochs across all folds
    all_epoch_sets = [set(d["epoch"] for d in fold_data) for fold_data in all_data.values()]
    common_epochs = sorted(set.intersection(*all_epoch_sets)) if all_epoch_sets else []

    if not common_epochs:
        print("No common epochs across folds. Skipping averaged plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect metrics per epoch
    metrics_keys = {
        "loss": ["train_loss", "val_loss", "test_loss"],
        "acc": ["train_window_acc", "val_accuracy", "test_accuracy"]
    }
    colors = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}

    for panel_idx, (panel_name, keys) in enumerate(metrics_keys.items()):
        ax = axes[panel_idx]
        for key in keys:
            values_per_epoch = []
            for ep in common_epochs:
                vals = []
                for fold_data in all_data.values():
                    ep_data = [d for d in fold_data if d["epoch"] == ep]
                    if ep_data:
                        alt_key = "train_class_acc" if key == "train_window_acc" and key not in ep_data[0] else key
                        v = ep_data[0].get(alt_key, np.nan)
                        if v is not None:
                            vals.append(v)
                if vals:
                    values_per_epoch.append((np.mean(vals), np.std(vals)))
                else:
                    values_per_epoch.append((np.nan, np.nan))

            means = [v[0] for v in values_per_epoch]
            stds = [v[1] for v in values_per_epoch]

            label_prefix = key.split("_")[0].capitalize()
            if "train" in key:
                c = colors["train"]
            elif "val" in key:
                c = colors["val"]
            else:
                c = colors["test"]

            ax.plot(common_epochs, means, 'o-', label=key, color=c)
            ax.fill_between(common_epochs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=c)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if panel_name == "loss" else "Accuracy")
        ax.set_title(f"All Folds (Mean ± Std) — {panel_name.capitalize()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if panel_name == "acc":
            ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    out_path = save_dir / "training_curves_all_folds.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    all_data = {}
    for fold_id in range(FOLDS):
        log_path = OUT_ROOT / f"fold{fold_id}" / "log.txt"
        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping fold {fold_id}")
            continue
        data = parse_log(log_path)
        if not data:
            print(f"Warning: No epoch data in {log_path}")
            continue
        all_data[fold_id] = data
        plot_fold(fold_id, data, OUT_ROOT)

    if all_data:
        plot_all_folds(all_data, OUT_ROOT)


if __name__ == "__main__":
    main()
