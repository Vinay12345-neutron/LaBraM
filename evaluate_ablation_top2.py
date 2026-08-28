#!/usr/bin/env python3
"""
Comprehensive Evaluation & Publication-Grade Plotting Script for Ablation 3:
Partial Fine-Tuning of Top Two Transformer Blocks (Blocks 11–12) + Head
- Computes fold-wise metrics (ACC, SEN, SPE, PRE, F1, ROC-AUC, PR-AUC)
- Computes Mean ± SD and pooled confusion matrix across all 18,396 test windows
- Compares head-to-head against Full Fine-Tuning and Frozen Linear Probe
- Computes Parameter Reduction percentage
- Generates 4 publication figures (PNG @ 350 DPI and Vector PDF):
  1. training_curves (train & val loss curves across all 5 folds)
  2. comparison_3way (Full FT vs Frozen vs Top-2 across all metrics)
  3. parameter_efficiency (Trainable Parameters vs. Accuracy trade-off curve)
  4. confusion_matrix_pooled (Pooled Confusion Matrix)
"""

import json
import csv
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from modeling_finetune import labram_base_patch200_200

# IEEE Publication Typography
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAIN_ROOT = Path("runs/boredom_cv")
FROZEN_ROOT = Path("runs/ablation_frozen")
TOP2_ROOT = Path("runs/ablation_top2")

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'IZ'
]

def get_input_chans(ch_names):
    input_chans = [0]
    for name in ch_names:
        clean_name = name.upper().strip()
        if clean_name in standard_1020:
            idx = standard_1020.index(clean_name) + 1
            input_chans.append(idx)
        else:
            input_chans.append(1)
    return input_chans

def evaluate_model_fold(fold_id, root_dir):
    ckpt_path = root_dir / f"fold{fold_id}" / "checkpoint-best.pth"
    if not ckpt_path.exists():
        ckpt_path = root_dir / f"fold{fold_id}" / "checkpoint.pth"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    split_file = Path(f"boredom_split_fold{fold_id}.json")
    with open(split_file) as f:
        split = json.load(f)
    test_data = split['test']

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    y_trues, y_probs, y_preds = [], [], []

    for item in test_data:
        fpath = item['file']
        label = int(item['label'])
        try:
            with h5py.File(fpath, 'r') as f:
                keys = list(f.keys())
                if not keys:
                    continue
                obj = f[keys[0]]
                dset = obj['eeg'][:] if 'eeg' in obj else (obj['data'][:] if 'data' in obj else obj[:])
                ch_names = obj.attrs.get('chOrder', standard_1020)
                if isinstance(obj, h5py.Group) and 'eeg' in obj and 'chOrder' in obj['eeg'].attrs:
                    ch_names = obj['eeg'].attrs['chOrder']

                ch_names = [x.decode('utf-8') if isinstance(x, bytes) else x for x in ch_names]
                input_chans = torch.tensor(get_input_chans(ch_names), dtype=torch.long).to(DEVICE)

                if dset.shape[0] > dset.shape[1]:
                    dset = dset.T
                n_samples = dset.shape[1]

                for start in range(0, n_samples - 512 + 1, 512):
                    window = (dset[:, start:start+512] / 100.0)
                    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    p1 = tensor[:, :, 0:200]
                    p2 = tensor[:, :, 156:356]
                    p3 = tensor[:, :, 312:512]
                    x_cat = torch.cat([p1, p2, p3], dim=2)
                    x_input = rearrange(x_cat, 'b n (a t) -> b n a t', t=200)

                    out = model(x_input, input_chans=input_chans)
                    prob = torch.sigmoid(out[0, 0]).item()
                    pred = 1 if prob >= 0.5 else 0

                    y_trues.append(label)
                    y_probs.append(prob)
                    y_preds.append(pred)
        except Exception:
            continue

    y_trues = np.array(y_trues)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds, labels=[0, 1]).ravel()
    acc = accuracy_score(y_trues, y_preds) * 100.0
    sen = recall_score(y_trues, y_preds, zero_division=0) * 100.0
    spe = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    pre = precision_score(y_trues, y_preds, zero_division=0) * 100.0
    f1 = f1_score(y_trues, y_preds, zero_division=0) * 100.0
    roc_auc = roc_auc_score(y_trues, y_probs)
    pr_auc = average_precision_score(y_trues, y_probs)

    return {
        "fold": fold_id + 1,
        "n_samples": len(y_trues),
        "acc": acc, "sen": sen, "spe": spe, "pre": pre, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trues": y_trues, "preds": y_preds, "probs": y_probs
    }

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=350)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for fold_id in range(5):
        log_path = TOP2_ROOT / f"fold{fold_id}" / "log.txt"
        if not log_path.exists():
            continue
        train_losses, val_losses, epochs = [], [], []
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    epochs.append(d.get('epoch', len(epochs)))
                    train_losses.append(d.get('train_loss', np.nan))
                    val_losses.append(d.get('val_loss', np.nan))

        axes[0].plot(epochs, train_losses, label=f"Fold {fold_id+1}", color=colors[fold_id], linewidth=2.0)
        axes[1].plot(epochs, val_losses, label=f"Fold {fold_id+1}", color=colors[fold_id], linewidth=2.0)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(f"Ablation 3 (Top-2 Blocks): {title}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
        ax.set_ylabel("BCEWithLogitsLoss", fontsize=13, fontweight="bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = TOP2_ROOT / f"training_curves.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Training Curves to: {TOP2_ROOT}/training_curves.png & .pdf")

def plot_3way_comparison(ft_means, frozen_means, top2_means):
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=350)
    rects1 = ax.bar(x - width, [ft_means[k] for k in ['acc', 'sen', 'spe', 'pre', 'f1']], width, label='Full Fine-Tuning (~5.8M)', color='#1f77b4', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, [frozen_means[k] for k in ['acc', 'sen', 'spe', 'pre', 'f1']], width, label='Frozen Probe (201 params)', color='#d62728', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, [top2_means[k] for k in ['acc', 'sen', 'spe', 'pre', 'f1']], width, label='Top-2 Blocks (~0.97M)', color='#2ca02c', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Hierarchy Ablation: Full Fine-Tuning vs. Frozen Linear Probe vs. Top-2 Fine-Tuning', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = TOP2_ROOT / f"comparison_3way.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-Way Comparison Plot to: {TOP2_ROOT}/comparison_3way.png & .pdf")

def plot_parameter_efficiency(ft_acc, frozen_acc, top2_acc):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=350)

    params = [201, 966361, 5799137]
    accs = [frozen_acc, top2_acc, ft_acc]
    names = ['Frozen Linear Probe\n(201 params)', 'Top-2 Fine-Tuning\n(0.97M params, -83.3%)', 'Full Fine-Tuning\n(5.8M params)']
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    ax.plot(params, accs, marker='o', markersize=10, linestyle='--', color='#555555', linewidth=2.0, zorder=2)

    for p, a, n, c in zip(params, accs, names, colors):
        ax.scatter(p, a, color=c, s=180, zorder=3, edgecolor='black', linewidth=1.2)
        offset_y = 3.5 if p != 201 else -8.0
        ax.annotate(f"{n}\nAcc: {a:.2f}%", (p, a), textcoords="offset points", xytext=(0, offset_y),
                    ha='center', fontsize=11, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Trainable Parameters (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean 5-Fold Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Parameter-Efficiency vs. Accuracy Trade-Off', fontsize=15, fontweight='bold', pad=12)
    ax.set_ylim(40, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = TOP2_ROOT / f"parameter_efficiency.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Parameter Efficiency Plot to: {TOP2_ROOT}/parameter_efficiency.png & .pdf")

def plot_pooled_confusion_matrix(tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=350)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Non-Boredom (0)", "Boredom (1)"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel="True Ground Truth", xlabel="Predicted Label")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = TOP2_ROOT / f"confusion_matrix_pooled.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Pooled Confusion Matrix to: {TOP2_ROOT}/confusion_matrix_pooled.png & .pdf")

def main():
    print("=" * 100)
    print("EVALUATING ABLATION 3: TOP-2 TRANSFORMER BLOCKS (BLOCKS 11-12) + HEAD FINE-TUNING")
    print("=" * 100)

    top2_results, main_results, frozen_results = [], [], []
    all_t_trues, all_t_preds, all_t_probs = [], [], []

    for fold_id in range(5):
        print(f"Evaluating Fold {fold_id + 1}...")
        rt = evaluate_model_fold(fold_id, TOP2_ROOT)
        rm = evaluate_model_fold(fold_id, MAIN_ROOT)
        rf = evaluate_model_fold(fold_id, FROZEN_ROOT)

        top2_results.append(rt)
        main_results.append(rm)
        frozen_results.append(rf)

        all_t_trues.extend(rt['trues'])
        all_t_preds.extend(rt['preds'])
        all_t_probs.extend(rt['probs'])

    # 1. Fold-wise Results Table
    print("\n" + "=" * 100)
    print("ABLATION 3 (TOP-2 BLOCKS): 5-FOLD SUBJECT-DISJOINT CV RESULTS")
    print("=" * 100)
    print(f"{'Fold':<10} | {'ACC (%)':<10} | {'SEN / REC (%)':<15} | {'SPE (%)':<10} | {'PRE (%)':<10} | {'F1-S (%)':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 100)
    for r in top2_results:
        print(f"Fold {r['fold']:<5} | {r['acc']:<10.2f} | {r['sen']:<15.2f} | {r['spe']:<10.2f} | {r['pre']:<10.2f} | {r['f1']:<10.2f} | {r['roc_auc']:<10.4f} | {r['pr_auc']:<10.4f}")

    t_accs = [r['acc'] for r in top2_results]
    t_sens = [r['sen'] for r in top2_results]
    t_spes = [r['spe'] for r in top2_results]
    t_pres = [r['pre'] for r in top2_results]
    t_f1s  = [r['f1'] for r in top2_results]
    t_rocs = [r['roc_auc'] for r in top2_results]
    t_prs  = [r['pr_auc'] for r in top2_results]

    print("-" * 100)
    print(f"{'Mean ± SD':<10} | {np.mean(t_accs):.2f}±{np.std(t_accs):.2f}{'':<3} | {np.mean(t_sens):.2f}±{np.std(t_sens):.2f}{'':<5} | {np.mean(t_spes):.2f}±{np.std(t_spes):.2f}{'':<3} | {np.mean(t_pres):.2f}±{np.std(t_pres):.2f}{'':<3} | {np.mean(t_f1s):.2f}±{np.std(t_f1s):.2f}{'':<3} | {np.mean(t_rocs):.4f}±{np.std(t_rocs):.4f} | {np.mean(t_prs):.4f}±{np.std(t_prs):.4f}")
    print("=" * 100)

    # 2. Pooled Confusion Matrix
    all_t_trues = np.array(all_t_trues)
    all_t_preds = np.array(all_t_preds)
    all_t_probs = np.array(all_t_probs)

    tn, fp, fn, tp = confusion_matrix(all_t_trues, all_t_preds, labels=[0, 1]).ravel()
    pooled_acc = (tn + tp) / len(all_t_trues) * 100.0
    pooled_sen = tp / (tp + fn) * 100.0
    pooled_spe = tn / (tn + fp) * 100.0
    pooled_pre = tp / (tp + fp) * 100.0
    pooled_f1  = 2 * tp / (2 * tp + fp + fn) * 100.0

    print("\n" + "=" * 100)
    print("POOLED TEST-SET METRICS (N = 18,396 HELD-OUT WINDOWS)")
    print("=" * 100)
    print(f"  - True Negatives  (TN - Non-Boredom) : {tn:,} / 9,198 ({(tn/9198)*100:.2f}%)")
    print(f"  - False Positives (FP - False Alarm) : {fp:,} / 9,198 ({(fp/9198)*100:.2f}%)")
    print(f"  - False Negatives (FN - Missed)      : {fn:,} / 9,198 ({(fn/9198)*100:.2f}%)")
    print(f"  - True Positives  (TP - Boredom)     : {tp:,} / 9,198 ({(tp/9198)*100:.2f}%)")
    print(f"  - Pooled Accuracy                    : {pooled_acc:.2f}% ({tn+tp:,} / 18,396)")
    print(f"  - Pooled Sensitivity / Recall        : {pooled_sen:.2f}%")
    print(f"  - Pooled Specificity                 : {pooled_spe:.2f}%")
    print(f"  - Pooled Precision                   : {pooled_pre:.2f}%")
    print(f"  - Pooled F1-Score                    : {pooled_f1:.2f}%")
    print("=" * 100)

    # 3. Head-to-Head Comparison: Full FT vs Top-2 FT
    m_accs = [r['acc'] for r in main_results]
    m_sens = [r['sen'] for r in main_results]
    m_spes = [r['spe'] for r in main_results]
    m_pres = [r['pre'] for r in main_results]
    m_f1s  = [r['f1'] for r in main_results]
    m_rocs = [r['roc_auc'] for r in main_results]
    m_prs  = [r['pr_auc'] for r in main_results]

    metrics = [
        ("Accuracy (%)", m_accs, t_accs, False),
        ("Sensitivity / Recall (%)", m_sens, t_sens, False),
        ("Specificity (%)", m_spes, t_spes, False),
        ("Precision (%)", m_pres, t_pres, False),
        ("F1-Score (%)", m_f1s, t_f1s, False),
        ("ROC-AUC", m_rocs, t_rocs, True),
        ("PR-AUC", m_prs, t_prs, True)
    ]

    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD COMPARISON: FULL FINE-TUNING vs. TOP-2 FINE-TUNING (ABLATION 3)")
    print("=" * 100)
    print(f"{'Metric':<26} | {'Full Fine-Tuning (~5.8M)':<24} | {'Top-2 Blocks (~0.97M)':<24} | {'Delta (Full FT - Top2)'}")
    print("-" * 100)

    summary_rows = []
    for label, m_vals, t_vals, is_auc in metrics:
        m_mean, m_std = np.mean(m_vals), np.std(m_vals)
        t_mean, t_std = np.mean(t_vals), np.std(t_vals)
        delta = m_mean - t_mean
        t_stat, p_val = ttest_rel(m_vals, t_vals)

        if is_auc:
            print(f"{label:<26} | {m_mean:.4f} ± {m_std:.4f}{'':<10} | {t_mean:.4f} ± {t_std:.4f}{'':<10} | {delta:+.4f} (p = {p_val:.4f})")
        else:
            print(f"{label:<26} | {m_mean:.2f}% ± {m_std:.2f}%{'':<8} | {t_mean:.2f}% ± {t_std:.2f}%{'':<8} | {delta:+.2f} pp (p = {p_val:.4f})")
        summary_rows.append({"metric": label, "ft_mean": m_mean, "ft_std": m_std, "top2_mean": t_mean, "top2_std": t_std, "delta": delta, "p_val": p_val})

    print("=" * 100)

    # 4. Parameter Reduction
    total_full = 5799137
    total_top2 = 966361
    param_reduction = (1 - (total_top2 / total_full)) * 100.0
    print(f"\nParameter Reduction: {param_reduction:.2f}% (from 5,799,137 down to 966,361 trainable parameters)")

    # 5. 3-Way Summary Table
    f_accs = [r['acc'] for r in frozen_results]
    f_sens = [r['sen'] for r in frozen_results]
    f_spes = [r['spe'] for r in frozen_results]
    f_pres = [r['pre'] for r in frozen_results]
    f_f1s  = [r['f1'] for r in frozen_results]

    print("\n" + "=" * 100)
    print("3-WAY COMPARISON: FULL FT vs. FROZEN LINEAR PROBE vs. TOP-2 FINE-TUNING")
    print("=" * 100)
    print(f"{'Configuration':<25} | {'Trainable Params':<18} | {'Accuracy (%)':<15} | {'Sensitivity (%)':<17} | {'Specificity (%)'}")
    print("-" * 100)
    print(f"{'Full Fine-Tuning':<25} | {'5,799,137 (100.0%)':<18} | {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%{'':<3} | {np.mean(m_sens):.2f}% ± {np.std(m_sens):.2f}%{'':<5} | {np.mean(m_spes):.2f}% ± {np.std(m_spes):.2f}%")
    print(f"{'Frozen Linear Probe':<25} | {'201 (0.003%)':<18} | {np.mean(f_accs):.2f}% ± {np.std(f_accs):.2f}%{'':<3} | {np.mean(f_sens):.2f}% ± {np.std(f_sens):.2f}%{'':<5} | {np.mean(f_spes):.2f}% ± {np.std(f_spes):.2f}%")
    print(f"{'Top-2 Fine-Tuning':<25} | {'966,361 (16.66%)':<18} | {np.mean(t_accs):.2f}% ± {np.std(t_accs):.2f}%{'':<3} | {np.mean(t_sens):.2f}% ± {np.std(t_sens):.2f}%{'':<5} | {np.mean(t_spes):.2f}% ± {np.std(t_spes):.2f}%")
    print("=" * 100)

    # Save summary CSV
    csv_path = TOP2_ROOT / "results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "ft_mean", "ft_std", "top2_mean", "top2_std", "delta", "p_val"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved Summary CSV to: {csv_path}")

    # Generate Figures
    plot_training_curves()
    plot_3way_comparison(
        {"acc": np.mean(m_accs), "sen": np.mean(m_sens), "spe": np.mean(m_spes), "pre": np.mean(m_pres), "f1": np.mean(m_f1s)},
        {"acc": np.mean(f_accs), "sen": np.mean(f_sens), "spe": np.mean(f_spes), "pre": np.mean(f_pres), "f1": np.mean(f_f1s)},
        {"acc": np.mean(t_accs), "sen": np.mean(t_sens), "spe": np.mean(t_spes), "pre": np.mean(t_pres), "f1": np.mean(t_f1s)}
    )
    plot_parameter_efficiency(np.mean(m_accs), np.mean(f_accs), np.mean(t_accs))
    plot_pooled_confusion_matrix(tn, fp, fn, tp)

if __name__ == "__main__":
    main()
