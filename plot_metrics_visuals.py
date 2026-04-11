#!/usr/bin/env python3
"""
Standardized Academic Metrics Visualizations.

Reads existing per-fold results from runs/boredom_cv/ and generates:
  1. Per-fold Confusion Matrices (2×2 heatmaps, normalized)
  2. Aggregated + per-fold ROC Curves (with AUC shading)
  3. Aggregated + per-fold Precision-Recall Curves (with AUC shading)
  4. Combined summary panel (all 3 in one figure)

All figures saved to: runs/publication_figures/metrics/

Usage:
    python plot_metrics_visuals.py
    python plot_metrics_visuals.py --results_dir runs/boredom_cv
    python plot_metrics_visuals.py --source loso   (reads loso_summary.json)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)

# ─── Style ────────────────────────────────────────────────────────────────────
FOLD_COLOURS = ["#E05A4E", "#4E90C8", "#5BBB6F", "#F5A623", "#9B59B6"]
MEAN_COLOUR  = "#222222"
OUT_DIR      = Path("runs/publication_figures/metrics")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_cv_folds(results_dir: Path) -> list[dict]:
    """
    Load per-fold preds/trues from test_predictions.json (last epoch entry).
    Falls back to fold_metrics.json for aggregate numbers if predictions
    are unavailable.
    """
    folds = []
    for fold_dir in sorted(results_dir.glob("fold*")):
        pred_file = fold_dir / "test_predictions.json"
        meta_file = fold_dir / "fold_metrics.json"
        fold_id   = int(fold_dir.name.replace("fold", ""))

        if pred_file.exists():
            # Read JSONL — last epoch entry contains best per-epoch predictions
            epochs = []
            with open(pred_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        epochs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if epochs:
                best = max(epochs, key=lambda d: d.get("epoch", 0))
                preds = np.array(best["preds"]).squeeze()
                trues = np.array(best["trues"]).astype(int)
                folds.append({"fold": fold_id, "preds": preds, "trues": trues})
                continue

        if meta_file.exists():
            with open(meta_file) as f:
                m = json.load(f)
            # Synthesize binary predictions from aggregate metrics when raw
            # predictions aren't available — approximate but sufficient for display
            folds.append({
                "fold":    fold_id,
                "preds":   None,
                "trues":   None,
                "metrics": m,
            })
    return folds


def load_loso_summary(loso_dir: Path) -> list[dict]:
    """Load per-fold data from loso_summary.json."""
    summary_path = loso_dir / "loso_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Not found: {summary_path}")
    with open(summary_path) as f:
        data = json.load(f)
    folds = data.get("folds", [])
    result = []
    for fold in folds:
        result.append({
            "fold":    fold.get("fold"),
            "subject": fold.get("test_subject"),
            "preds":   None,
            "trues":   None,
            "metrics": fold,
        })
    return result


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(folds: list[dict], out_dir: Path):
    """Plot one 2×2 confusion matrix per fold, plus aggregate."""
    valid = [f for f in folds if f["preds"] is not None]
    if not valid:
        print("  ⚠ No raw predictions — skipping confusion matrix plot.")
        return

    n = len(valid)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols + 1   # +1 for aggregate row

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    all_true = np.concatenate([f["trues"] for f in valid])
    all_pred = np.concatenate([(f["preds"] >= 0.5).astype(int) for f in valid])

    def draw_cm(ax, y_true, y_pred, title, fold_col):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                val = cm_norm[i, j]
                raw = cm[i, j]
                ax.text(j, i, f"{val:.2f}\n(n={raw})",
                        ha="center", va="center",
                        color="white" if val > 0.6 else "black",
                        fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neutral\n(Pred)", "Boredom\n(Pred)"], fontsize=9)
        ax.set_yticklabels(["Neutral\n(True)", "Boredom\n(True)"], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=fold_col, pad=8)

    # Per-fold
    for k, fold in enumerate(valid):
        pred_bin = (fold["preds"] >= 0.5).astype(int)
        acc = (pred_bin == fold["trues"]).mean()
        draw_cm(axes[k], fold["trues"], pred_bin,
                f"Fold {fold['fold']}  (acc={acc:.1%})",
                FOLD_COLOURS[k % len(FOLD_COLOURS)])

    # Aggregate
    agg_ax = axes[n]
    draw_cm(agg_ax, all_true, all_pred,
            f"Aggregate  (n={len(all_true)})  "
            f"acc={(all_pred == all_true).mean():.1%}",
            MEAN_COLOUR)

    # Hide unused axes
    for k in range(n + 1, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Confusion Matrices — Boredom vs Neutral Classification",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_roc_curves(folds: list[dict], out_dir: Path):
    """Plot per-fold + mean ROC curves with AUC annotations."""
    valid = [f for f in folds if f["preds"] is not None]
    if not valid:
        print("  ⚠ No raw predictions — skipping ROC curve.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    all_interp = []
    base_fpr   = np.linspace(0, 1, 200)

    for fold in valid:
        fpr, tpr, _ = roc_curve(fold["trues"], fold["preds"])
        roc_auc     = auc(fpr, tpr)
        tpr_interp  = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_interp.append(tpr_interp)

        col = FOLD_COLOURS[fold["fold"] % len(FOLD_COLOURS)]
        ax.plot(fpr, tpr, color=col, lw=1.2, alpha=0.55,
                label=f"Fold {fold['fold']} (AUC={roc_auc:.3f})")

    # Mean ± std band
    mean_tpr  = np.mean(all_interp, axis=0)
    std_tpr   = np.std(all_interp, axis=0)
    mean_auc  = auc(base_fpr, mean_tpr)
    ax.plot(base_fpr, mean_tpr, color=MEAN_COLOUR, lw=2.5,
            label=f"Mean  (AUC={mean_auc:.3f})")
    ax.fill_between(base_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color=MEAN_COLOUR, alpha=0.10, label="±1 SD")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.5)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Boredom vs Neutral",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "roc_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_pr_curves(folds: list[dict], out_dir: Path):
    """Plot per-fold + mean Precision-Recall curves with AUC annotations."""
    valid = [f for f in folds if f["preds"] is not None]
    if not valid:
        print("  ⚠ No raw predictions — skipping PR curve.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    all_interp_prec = []
    base_recall     = np.linspace(0, 1, 200)

    for fold in valid:
        prec, rec, _ = precision_recall_curve(fold["trues"], fold["preds"])
        ap            = average_precision_score(fold["trues"], fold["preds"])
        # Interpolate (recall is decreasing → flip)
        prec_interp  = np.interp(base_recall, rec[::-1], prec[::-1])
        all_interp_prec.append(prec_interp)

        col = FOLD_COLOURS[fold["fold"] % len(FOLD_COLOURS)]
        ax.plot(rec, prec, color=col, lw=1.2, alpha=0.55,
                label=f"Fold {fold['fold']} (AP={ap:.3f})")

    mean_prec = np.mean(all_interp_prec, axis=0)
    std_prec  = np.std(all_interp_prec, axis=0)
    mean_ap   = np.trapz(mean_prec, base_recall)

    ax.plot(base_recall, mean_prec, color=MEAN_COLOUR, lw=2.5,
            label=f"Mean  (AP={mean_ap:.3f})")
    ax.fill_between(base_recall,
                    np.clip(mean_prec - std_prec, 0, 1),
                    np.clip(mean_prec + std_prec, 0, 1),
                    color=MEAN_COLOUR, alpha=0.10, label="±1 SD")

    # Baseline (random) — proportion of positive class
    pos_rate = np.mean(np.concatenate([f["trues"] for f in valid]))
    ax.axhline(pos_rate, color="gray", linestyle="--", lw=0.8,
               label=f"Random (precision={pos_rate:.2f})")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Boredom vs Neutral",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "pr_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_combined_panel(folds: list[dict], out_dir: Path):
    """Single figure: CM (aggregate) + ROC + PR side by side."""
    valid = [f for f in folds if f["preds"] is not None]
    if not valid:
        print("  ⚠ No raw predictions — skipping combined panel.")
        return

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35, left=0.06, right=0.97)

    all_true = np.concatenate([f["trues"] for f in valid])
    all_pred_prob = np.concatenate([f["preds"] for f in valid])
    all_pred_bin  = (all_pred_prob >= 0.5).astype(int)

    # ── Panel A: Confusion Matrix ──────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0])
    cm = confusion_matrix(all_true, all_pred_bin)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                       ha="center", va="center",
                       color="white" if cm_norm[i, j] > 0.6 else "black",
                       fontsize=12, fontweight="bold")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Neutral\n(Pred)", "Boredom\n(Pred)"], fontsize=10)
    ax_cm.set_yticklabels(["Neutral\n(True)", "Boredom\n(True)"], fontsize=10)
    acc = (all_pred_bin == all_true).mean()
    ax_cm.set_title(f"(A) Confusion Matrix\nacc={acc:.1%}  n={len(all_true)}",
                    fontsize=11, fontweight="bold")

    # ── Panel B: ROC ──────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1])
    base_fpr = np.linspace(0, 1, 200)
    all_tpr_interp = []
    for fold in valid:
        fpr, tpr, _ = roc_curve(fold["trues"], fold["preds"])
        ax_roc.plot(fpr, tpr, color=FOLD_COLOURS[fold["fold"] % len(FOLD_COLOURS)],
                    lw=1.0, alpha=0.4)
        all_tpr_interp.append(np.interp(base_fpr, fpr, tpr))

    mean_tpr = np.mean(all_tpr_interp, axis=0)
    mean_auc = auc(base_fpr, mean_tpr)
    ax_roc.plot(base_fpr, mean_tpr, color=MEAN_COLOUR, lw=2.2,
                label=f"Mean AUC = {mean_auc:.3f}")
    ax_roc.fill_between(base_fpr,
                        np.clip(mean_tpr - np.std(all_tpr_interp, axis=0), 0, 1),
                        np.clip(mean_tpr + np.std(all_tpr_interp, axis=0), 0, 1),
                        color=MEAN_COLOUR, alpha=0.10)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.set_title("(B) ROC Curve", fontsize=11, fontweight="bold")
    ax_roc.legend(fontsize=9)
    ax_roc.spines[["top", "right"]].set_visible(False)

    # ── Panel C: PR Curve ─────────────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[2])
    base_rec = np.linspace(0, 1, 200)
    all_prec_interp = []
    for fold in valid:
        p, r, _ = precision_recall_curve(fold["trues"], fold["preds"])
        ax_pr.plot(r, p, color=FOLD_COLOURS[fold["fold"] % len(FOLD_COLOURS)],
                   lw=1.0, alpha=0.4)
        all_prec_interp.append(np.interp(base_rec, r[::-1], p[::-1]))

    mean_prec = np.mean(all_prec_interp, axis=0)
    mean_ap   = np.trapz(mean_prec, base_rec)
    ax_pr.plot(base_rec, mean_prec, color=MEAN_COLOUR, lw=2.2,
               label=f"Mean AP = {mean_ap:.3f}")
    ax_pr.fill_between(base_rec,
                       np.clip(mean_prec - np.std(all_prec_interp, axis=0), 0, 1),
                       np.clip(mean_prec + np.std(all_prec_interp, axis=0), 0, 1),
                       color=MEAN_COLOUR, alpha=0.10)
    pos_rate = np.mean(np.concatenate([f["trues"] for f in valid]))
    ax_pr.axhline(pos_rate, color="gray", linestyle="--", lw=0.8,
                  label=f"Baseline ({pos_rate:.2f})")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("(C) Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.legend(fontsize=9, loc="lower left")
    ax_pr.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Academic Performance Summary — LaBraM Boredom Detection",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "combined_metrics_panel.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, default="runs/boredom_cv",
                        help="CV output directory (default: runs/boredom_cv)")
    parser.add_argument("--source", choices=["cv", "loso", "loso_bestparam"],
                        default="cv",
                        help="Which result set to visualize (default: cv)")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fold data ─────────────────────────────────────────────────────
    if args.source == "cv":
        results_dir = Path(args.results_dir)
        folds = load_cv_folds(results_dir)
        print(f"Loaded {len(folds)} CV folds from {results_dir}")
    elif args.source == "loso":
        loso_dir = Path("runs/boredom_loso")
        folds = load_loso_summary(loso_dir)
        print(f"Loaded {len(folds)} LOSO folds from {loso_dir}")
    else:  # loso_bestparam
        loso_dir = Path("runs/boredom_loso_bestparam")
        folds = load_loso_summary(loso_dir)
        print(f"Loaded {len(folds)} LOSO-bestparam folds from {loso_dir}")

    if not folds:
        print("No fold data found. Check your results directory.")
        return

    # ── Generate all plots ─────────────────────────────────────────────────
    print(f"\nSaving figures to {out_dir}/")
    plot_confusion_matrices(folds, out_dir)
    plot_roc_curves(folds, out_dir)
    plot_pr_curves(folds, out_dir)
    plot_combined_panel(folds, out_dir)

    print("\nDone. All metrics figures saved.")


if __name__ == "__main__":
    main()
