#!/usr/bin/env python3
"""
Generate publication-quality Learning Curves for LaBraM Boredom Fine-tuning.

Specifications:
- Resolution: 350 DPI
- Font Family: Arial (fallback to sans-serif)
- Font Size: 16-18 pt
- Test Loss & Test Accuracy completely REMOVED.
- Plots generated:
  1. Validation AUC Curve against Epochs (curve_val_auc.png)
  2. Train Loss vs Validation Loss against Epochs (curve_loss.png)
  3. Train Accuracy vs Validation Accuracy against Epochs (curve_accuracy.png)
  4. Train AUC vs Validation AUC against Epochs (curve_auc.png)
  5. Individual Fold Revised Learning Curves (training_curves_foldX.png)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication style parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

OUT_ROOT = Path("runs/boredom_cv")
FOLDS = 5
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def parse_log(log_path):
    """Parse a log.txt file and return per-epoch metrics (last entry per epoch)."""
    epochs_dict = {}
    with open(log_path, "r") as f:
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
            # Keep the last recorded entry per epoch
            epochs_dict[ep] = d
            
    sorted_epochs = sorted(epochs_dict.keys())
    return [epochs_dict[ep] for ep in sorted_epochs]

def extract_metric_matrix(all_data, metric_key, alt_key=None):
    """Safely extract 2D numpy array of shape (n_folds, max_epochs) with 1-indexed epoch grid."""
    all_eps = set()
    for fold_data in all_data.values():
        for d in fold_data:
            # Map 0-indexed epoch to 1-indexed for display
            ep = d["epoch"] + 1 if min(x["epoch"] for x in fold_data) == 0 else d["epoch"]
            all_eps.add(ep)
    
    max_ep = max(all_eps) if all_eps else 25
    ep_grid = np.arange(1, max_ep + 1)
    matrix = []
    
    for fold_id, fold_data in all_data.items():
        ep_map = {}
        for d in fold_data:
            ep = d["epoch"] + 1 if min(x["epoch"] for x in fold_data) == 0 else d["epoch"]
            ep_map[ep] = d
            
        row = []
        for ep in ep_grid:
            if ep in ep_map:
                v = ep_map[ep].get(metric_key)
                if v is None and alt_key:
                    v = ep_map[ep].get(alt_key, np.nan)
                if v is None:
                    v = np.nan
            else:
                v = np.nan
            row.append(v)
        matrix.append(row)
    return ep_grid, np.array(matrix)

def plot_val_auc_curve(all_data, save_dir):
    """Plot 1: Validation AUC curve against epochs (5 folds + mean)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=350)
    
    ep_grid, auc_matrix = extract_metric_matrix(all_data, "val_roc_auc")
    
    for fold_id in range(len(auc_matrix)):
        ax.plot(ep_grid, auc_matrix[fold_id], color=COLORS[fold_id % len(COLORS)],
                lw=1.5, alpha=0.45, label=f"Fold {fold_id}")
        
    mean_val_auc = np.nanmean(auc_matrix, axis=0)
    std_val_auc = np.nanstd(auc_matrix, axis=0)
    
    ax.plot(ep_grid, mean_val_auc, color="#004d40", lw=3.5, label="Mean Val AUC")
    ax.fill_between(ep_grid, mean_val_auc - std_val_auc, mean_val_auc + std_val_auc,
                    color="#80cbc4", alpha=0.3, label=r"± 1 Std. Dev.")
    
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Validation ROC-AUC", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    out_path = save_dir / "5fold_validation_auc_curve.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_loss_curve(all_data, save_dir):
    """Plot 2: Train Loss and Validation Loss against epochs (NO Test Loss)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=350)
    
    ep_grid, train_losses = extract_metric_matrix(all_data, "train_loss")
    _, val_losses = extract_metric_matrix(all_data, "val_loss")
    
    mean_train_loss = np.nanmean(train_losses, axis=0)
    std_train_loss = np.nanstd(train_losses, axis=0)
    
    mean_val_loss = np.nanmean(val_losses, axis=0)
    std_val_loss = np.nanstd(val_losses, axis=0)
    
    ax.plot(ep_grid, mean_train_loss, 'o-', color="#1e88e5", lw=2.5, label="Train Loss")
    ax.fill_between(ep_grid, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                    color="#90caf9", alpha=0.3)
    
    ax.plot(ep_grid, mean_val_loss, 's-', color="#fb8c00", lw=2.5, label="Validation Loss")
    ax.fill_between(ep_grid, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                    color="#ffe082", alpha=0.3)
    
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Loss", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="upper right", fontsize=14, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_path = save_dir / "5fold_mean_train_val_loss_curve.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_accuracy_curve(all_data, save_dir):
    """Plot 3: Train Accuracy and Validation Accuracy against epochs (NO Test Accuracy)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=350)
    
    ep_grid, train_accs = extract_metric_matrix(all_data, "train_window_acc", alt_key="train_class_acc")
    _, val_accs = extract_metric_matrix(all_data, "val_accuracy")
    
    mean_train_acc = np.nanmean(train_accs, axis=0)
    std_train_acc = np.nanstd(train_accs, axis=0)
    
    mean_val_acc = np.nanmean(val_accs, axis=0)
    std_val_acc = np.nanstd(val_accs, axis=0)
    
    ax.plot(ep_grid, mean_train_acc, 'o-', color="#1e88e5", lw=2.5, label="Train Accuracy")
    ax.fill_between(ep_grid, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                    color="#90caf9", alpha=0.3)
    
    ax.plot(ep_grid, mean_val_acc, 's-', color="#fb8c00", lw=2.5, label="Validation Accuracy")
    ax.fill_between(ep_grid, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                    color="#ffe082", alpha=0.3)
    
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="lower right", fontsize=14, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    out_path = save_dir / "5fold_mean_train_val_accuracy_curve.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_train_val_auc_curve(all_data, save_dir):
    """Plot 4: Train AUC and Validation AUC against epochs."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=350)
    
    ep_grid, train_aucs = extract_metric_matrix(all_data, "train_roc_auc", alt_key="train_window_acc")
    _, val_aucs = extract_metric_matrix(all_data, "val_roc_auc")
    
    mean_train_auc = np.nanmean(train_aucs, axis=0)
    std_train_auc = np.nanstd(train_aucs, axis=0)
    
    mean_val_auc = np.nanmean(val_aucs, axis=0)
    std_val_auc = np.nanstd(val_aucs, axis=0)
    
    ax.plot(ep_grid, mean_train_auc, 'o-', color="#1e88e5", lw=2.5, label="Train AUC")
    ax.fill_between(ep_grid, mean_train_auc - std_train_auc, mean_train_auc + std_train_auc,
                    color="#90caf9", alpha=0.3)
    
    ax.plot(ep_grid, mean_val_auc, 's-', color="#43a047", lw=2.5, label="Validation AUC")
    ax.fill_between(ep_grid, mean_val_auc - std_val_auc, mean_val_auc + std_val_auc,
                    color="#a5d6a7", alpha=0.3)
    
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("ROC-AUC", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="lower right", fontsize=14, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    out_path = save_dir / "5fold_mean_train_val_auc_curve.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_fold(fold_id, data, save_dir):
    """Plot revised loss and accuracy curves for a single fold without test loss/accuracy."""
    epochs = [d["epoch"] for d in data]
    ep_grid = range(1, len(epochs) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=350)

    # --- Loss Panel ---
    ax = axes[0]
    if "train_loss" in data[0]:
        ax.plot(ep_grid, [d["train_loss"] for d in data], 'o-', label="Train Loss", color="#1e88e5", lw=2)
    if "val_loss" in data[0]:
        ax.plot(ep_grid, [d.get("val_loss", np.nan) for d in data], 's-', label="Val Loss", color="#fb8c00", lw=2)
    ax.set_xlabel("Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- Accuracy & AUC Panel ---
    ax = axes[1]
    train_acc_key = "train_window_acc" if "train_window_acc" in data[0] else "train_class_acc"
    if train_acc_key in data[0]:
        ax.plot(ep_grid, [d.get(train_acc_key, np.nan) for d in data], 'o-', label="Train Acc", color="#1e88e5", lw=2)
    if "val_accuracy" in data[0]:
        ax.plot(ep_grid, [d.get("val_accuracy", np.nan) for d in data], 's-', label="Val Acc", color="#fb8c00", lw=2)
    if "val_roc_auc" in data[0]:
        ax.plot(ep_grid, [d.get("val_roc_auc", np.nan) for d in data], '^-', label="Val AUC", color="#43a047", lw=2)
        
    ax.set_xlabel("Epoch", fontsize=16, fontweight="bold")
    ax.set_ylabel("Accuracy and AUC", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    out_path = save_dir / f"training_curves_fold{fold_id}.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

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
        plot_val_auc_curve(all_data, OUT_ROOT)
        plot_loss_curve(all_data, OUT_ROOT)
        plot_accuracy_curve(all_data, OUT_ROOT)
        plot_train_val_auc_curve(all_data, OUT_ROOT)
        print(f"\nSuccessfully generated revised learning curves matching all publication specifications.")

if __name__ == "__main__":
    main()
