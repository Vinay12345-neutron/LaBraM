#!/usr/bin/env python3
"""
Generate publication-quality Confusion Matrices for each 5-fold CV fold and overall summary.

Specifications:
- Resolution: 350 DPI
- Font Family: Arial (fallback to sans-serif)
- Font Size: 16-18 pt
- Labels: "Non Boredom" (0) and "Boredom" (1) [replaced "Neutral"]
- Explicit Tags in Matrix: TN, FP, FN, TP with raw counts
- Titles: Clean fold title without accuracy text ("Fold 0", "Fold 1", ..., "Overall 5-Fold Confusion Matrix")
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

RUNS_DIR = Path("runs/boredom_cv")
CLASSES = ["Non Boredom", "Boredom"]

def load_predictions_for_fold(fold_dir):
    """Load the final epoch test predictions and true labels from fold directory."""
    pred_file = fold_dir / "test_predictions.json"
    if not pred_file.exists():
        print(f"Warning: {pred_file} not found.")
        return None, None
    
    with open(pred_file, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return None, None
    
    # Read the last epoch line
    last_line = json.loads(lines[-1].strip())
    preds = np.array(last_line["preds"])
    trues = np.array(last_line["trues"])
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    preds_bin = (preds >= 0.5).astype(int)
    trues_bin = trues.astype(int)
    
    return trues_bin, preds_bin

def compute_confusion_matrix(trues, preds):
    """Compute 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    tn = np.sum((trues == 0) & (preds == 0))
    fp = np.sum((trues == 0) & (preds == 1))
    fn = np.sum((trues == 1) & (preds == 0))
    tp = np.sum((trues == 1) & (preds == 1))
    return np.array([[tn, fp], [fn, tp]])

def plot_confusion_matrix(cm, title, output_path):
    """Plot formatted 2x2 confusion matrix with cell percentages according to PNG specifications."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=350)
    
    # Calculate row-normalized percentage (true label percentage)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero if row_sum is 0
    cm_perc = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0) * 100.0
    
    # Heatmap rendering using a pleasant Teal/Green colormap (GnBu) with range 0 to 100%
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.GnBu, vmin=0, vmax=100)
    
    # Colorbar showing percentage ticks (0%, 20%, 40%, 60%, 80%, 100%)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    
    # Matrix annotations with TP, TN, FP, FN tags and percentage values
    tags = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            perc_val = cm_perc[i, j]
            tag = tags[i][j]
            text = f"{tag}\n{perc_val:.1f}%"
            
            # Determine text color based on background intensity
            color = "white" if perc_val > 50.0 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=18, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES, fontsize=16, fontweight="bold")
    ax.set_yticklabels(CLASSES, fontsize=16, fontweight="bold", rotation=90, va="center")
    
    ax.set_xlabel("Predicted Label", fontsize=18, labelpad=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=18, labelpad=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    total_cm = np.zeros((2, 2), dtype=int)
    all_folds_processed = 0

    for fold_idx in range(5):
        fold_dir = RUNS_DIR / f"fold{fold_idx}"
        trues, preds = load_predictions_for_fold(fold_dir)
        
        if trues is None or preds is None:
            continue
            
        cm = compute_confusion_matrix(trues, preds)
        total_cm += cm
        all_folds_processed += 1
        
        # Plot fold-specific matrix
        out_path = RUNS_DIR / f"confusion_matrix_fold{fold_idx}.png"
        plot_confusion_matrix(cm, f"Fold {fold_idx}", out_path)

    if all_folds_processed > 0:
        # Plot overall 5-fold combined matrix
        out_path_total = RUNS_DIR / "confusion_matrix_overall.png"
        plot_confusion_matrix(total_cm, "Overall 5-Fold Confusion Matrix", out_path_total)
        print(f"\nSuccessfully generated {all_folds_processed} fold matrices and 1 overall matrix.")

if __name__ == "__main__":
    main()
