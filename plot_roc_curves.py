#!/usr/bin/env python3
"""
Generate publication-quality ROC Curves for each 5-fold CV fold and overall summary.

Specifications:
- Resolution: 350 DPI
- Font Family: Arial (fallback to sans-serif)
- Font Size: 16-18 pt
- Displays exact AUC for each fold and Mean AUC ± Std Dev for the overall curve
- Color palette: Pleasant Teal/Emerald/Navy palette
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc

# Set publication style parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

RUNS_DIR = Path("runs/boredom_cv")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def load_predictions_for_fold(fold_dir):
    """Load final epoch prediction probabilities and true labels."""
    pred_file = fold_dir / "test_predictions.json"
    if not pred_file.exists():
        print(f"Warning: {pred_file} not found.")
        return None, None
    
    with open(pred_file, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return None, None
    
    last_line = json.loads(lines[-1].strip())
    preds = np.array(last_line["preds"])
    trues = np.array(last_line["trues"])
    return trues, preds

def plot_individual_roc(fpr, tpr, roc_auc, fold_idx, output_path):
    """Plot individual ROC curve for a single fold."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=350)
    
    ax.plot(fpr, tpr, color="#008080", lw=3, label=f"Fold {fold_idx} AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, linestyle="--")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="lower right", fontsize=14, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def plot_overall_roc(tprs, aucs, mean_fpr, output_path):
    """Plot aggregated ROC curves across all 5 folds with mean AUC ± std dev."""
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=350)
    
    # Plot individual fold ROC curves
    for i in range(len(tprs)):
        ax.plot(mean_fpr, tprs[i], color=COLORS[i % len(COLORS)], lw=1.5, alpha=0.45,
                label=f"Fold {i} AUC = {aucs[i]:.4f}")
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plot mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color="#004d40", lw=3.5,
            label=f"Mean ROC AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Plot standard deviation shaded region
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="#80cbc4", alpha=0.3,
                    label=r"± 1 Std. Dev.")
    
    # Thin, light-colored diagonal baseline (excluded from legend)
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, linestyle="--")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_folds_processed = 0

    for fold_idx in range(5):
        fold_dir = RUNS_DIR / f"fold{fold_idx}"
        trues, preds = load_predictions_for_fold(fold_dir)
        
        if trues is None or preds is None:
            continue
            
        fpr, tpr, _ = roc_curve(trues, preds)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPRs to align on mean_fpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        all_folds_processed += 1
        
        # Save individual fold ROC curve
        out_path = RUNS_DIR / f"roc_curve_fold{fold_idx}.png"
        plot_individual_roc(fpr, tpr, roc_auc, fold_idx, out_path)

    if all_folds_processed > 0:
        # Save overall aggregated ROC curves
        out_path_total = RUNS_DIR / "roc_curve_overall.png"
        plot_overall_roc(tprs, aucs, mean_fpr, out_path_total)
        print(f"\nSuccessfully generated {all_folds_processed} individual ROC curves and 1 overall summary ROC plot.")

if __name__ == "__main__":
    main()
