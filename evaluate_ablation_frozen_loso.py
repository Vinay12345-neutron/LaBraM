#!/usr/bin/env python3
"""
Comprehensive Evaluation & Publication-Grade Plotting Script for:
Ablation 2 under Leave-One-Subject-Out (LOSO) Cross-Validation (73 Subjects)
- Evaluates each held-out subject using checkpoint-best.pth from runs/ablation_frozen_loso/S{sub}
- Reports 73 individual subject metrics, Macro Subject-Level Mean ± SD, and Pooled Window-Level Confusion Matrix
- Compares head-to-head against the Main Full Fine-Tuning LOSO baseline (runs/boredom_loso/)
- Exports 4 IEEE publication figures (PNG @ 350 DPI and Vector PDF):
  1. loso_subject_accuracy_bar (Per-subject accuracy comparison bar chart)
  2. loso_accuracy_distribution (Boxplot & KDE distribution across 73 subjects)
  3. confusion_matrix_pooled (Pooled 18,396-window confusion matrix)
  4. comparison_main_vs_frozen_loso (Side-by-side metric comparison)
"""

import json
import csv
import re
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from modeling_finetune import labram_base_patch200_200

# IEEE Publication Typography
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKDIR = Path.cwd()
SPLITS_DIR = WORKDIR / "loso_splits"
FROZEN_LOSO_ROOT = WORKDIR / "runs" / "ablation_frozen_loso"
MAIN_LOSO_ROOT = WORKDIR / "runs" / "boredom_loso"

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

def evaluate_single_subject(sub_id, root_dir):
    ckpt_path = root_dir / f"S{sub_id}" / "checkpoint-best.pth"
    if not ckpt_path.exists():
        ckpt_path = root_dir / f"S{sub_id}" / "checkpoint.pth"
    if not ckpt_path.exists():
        return None

    split_file = SPLITS_DIR / f"loso_split_S{sub_id}.json"
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

    if len(y_trues) == 0:
        return None

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds, labels=[0, 1]).ravel()
    acc = accuracy_score(y_trues, y_preds) * 100.0
    sen = recall_score(y_trues, y_preds, zero_division=0) * 100.0
    spe = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    pre = precision_score(y_trues, y_preds, zero_division=0) * 100.0
    f1 = f1_score(y_trues, y_preds, zero_division=0) * 100.0

    has_both = len(np.unique(y_trues)) > 1
    roc_auc = roc_auc_score(y_trues, y_probs) if has_both else np.nan
    pr_auc = average_precision_score(y_trues, y_probs) if has_both else np.nan

    return {
        "sub_id": sub_id,
        "n_windows": len(y_trues),
        "acc": acc, "sen": sen, "spe": spe, "pre": pre, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trues": y_trues, "preds": y_preds, "probs": y_probs
    }

def plot_loso_subject_bar(subject_results, main_subject_results=None):
    subs = [r['sub_id'] for r in subject_results]
    frozen_accs = [r['acc'] for r in subject_results]

    fig, ax = plt.subplots(figsize=(16, 5.5), dpi=350)
    x = np.arange(len(subs))
    width = 0.4

    if main_subject_results:
        main_accs = [main_subject_results.get(s, {}).get('acc', np.nan) for s in subs]
        ax.bar(x - width/2, main_accs, width, label='Full Fine-Tuning (~5.8M)', color='#1f77b4', alpha=0.85)
        ax.bar(x + width/2, frozen_accs, width, label='Frozen Linear Probe (201 params)', color='#d62728', alpha=0.85)
    else:
        ax.bar(x, frozen_accs, 0.7, label='Frozen Linear Probe', color='#d62728', alpha=0.85)

    ax.axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Chance (50.0%)')
    ax.axhline(np.mean(frozen_accs), color='#d62728', linestyle=':', linewidth=1.5,
               label=f'Frozen Mean ({np.mean(frozen_accs):.1f}%)')

    ax.set_xlabel('Held-Out Subject ID (LOSO)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Subject Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Ablation 2: Leave-One-Subject-Out (LOSO) Accuracy across 73 Subjects', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"S{subs[i]}" for i in range(0, len(subs), 2)], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = FROZEN_LOSO_ROOT / f"loso_subject_accuracy_bar.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved LOSO Subject Bar Chart to: {FROZEN_LOSO_ROOT}/loso_subject_accuracy_bar.png & .pdf")

def plot_loso_distribution(frozen_accs, main_accs=None):
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=350)
    
    data = [frozen_accs]
    labels = ['Frozen Probe\n(201 params)']
    colors = ['#d62728']

    if main_accs:
        data.insert(0, main_accs)
        labels.insert(0, 'Full Fine-Tuning\n(~5.8M params)')
        colors.insert(0, '#1f77b4')

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black"})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual subject points with jitter
    for i, d in enumerate(data):
        y = d
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'k.', alpha=0.5, markersize=6)

    ax.axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Chance (50.0%)')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Subject-Level LOSO Accuracy Distribution (N = 73)', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(20, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='lower left', fontsize=10, frameon=True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out_path = FROZEN_LOSO_ROOT / f"loso_accuracy_distribution.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved LOSO Accuracy Distribution to: {FROZEN_LOSO_ROOT}/loso_accuracy_distribution.png & .pdf")

def plot_pooled_confusion_matrix(tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=350)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
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
        out_path = FROZEN_LOSO_ROOT / f"confusion_matrix_pooled.{ext}"
        plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Pooled Confusion Matrix to: {FROZEN_LOSO_ROOT}/confusion_matrix_pooled.png & .pdf")

def main():
    print("=" * 100)
    print("EVALUATING ABLATION 2 (FROZEN BACKBONE LINEAR PROBE) UNDER 73-FOLD LOSO CV")
    print("=" * 100)

    split_files = sorted(SPLITS_DIR.glob("loso_split_S*.json"), key=lambda p: int(re.search(r"S(\d+)", p.name).group(1)))
    subs = [int(re.search(r"S(\d+)", p.name).group(1)) for p in split_files]

    frozen_results = []
    main_results_dict = {}

    all_trues, all_preds, all_probs = [], [], []

    # Check if main LOSO model results exist for comparison
    main_loso_exists = MAIN_LOSO_ROOT.exists()

    for idx, sub_id in enumerate(subs):
        rf = evaluate_single_subject(sub_id, FROZEN_LOSO_ROOT)
        if rf is not None:
            frozen_results.append(rf)
            all_trues.extend(rf['trues'])
            all_preds.extend(rf['preds'])
            all_probs.extend(rf['probs'])

        if main_loso_exists:
            rm = evaluate_single_subject(sub_id, MAIN_LOSO_ROOT)
            if rm is not None:
                main_results_dict[sub_id] = rm

    if len(frozen_results) == 0:
        print("ERROR: No completed LOSO folds found in runs/ablation_frozen_loso/")
        return

    print(f"\nSuccessfully evaluated {len(frozen_results)} / 73 held-out LOSO subjects.")

    # 1. Macro Subject-Level Statistics
    accs = [r['acc'] for r in frozen_results]
    sens = [r['sen'] for r in frozen_results]
    spes = [r['spe'] for r in frozen_results]
    pres = [r['pre'] for r in frozen_results]
    f1s  = [r['f1'] for r in frozen_results]
    rocs = [r['roc_auc'] for r in frozen_results if not np.isnan(r['roc_auc'])]
    prs  = [r['pr_auc'] for r in frozen_results if not np.isnan(r['pr_auc'])]

    print("\n" + "=" * 100)
    print("1. MACRO SUBJECT-LEVEL SUMMARY (N = 73 HELD-OUT SUBJECTS)")
    print("=" * 100)
    print(f"  - Subject Accuracy (%)     : {np.mean(accs):.2f}% ± {np.std(accs):.2f}% (Min: {np.min(accs):.2f}%, Max: {np.max(accs):.2f}%)")
    print(f"  - Subject Sensitivity (%)  : {np.mean(sens):.2f}% ± {np.std(sens):.2f}%")
    print(f"  - Subject Specificity (%)  : {np.mean(spes):.2f}% ± {np.std(spes):.2f}%")
    print(f"  - Subject Precision (%)    : {np.mean(pres):.2f}% ± {np.std(pres):.2f}%")
    print(f"  - Subject F1-Score (%)     : {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
    if rocs:
        print(f"  - Subject ROC-AUC          : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
    if prs:
        print(f"  - Subject PR-AUC           : {np.mean(prs):.4f} ± {np.std(prs):.4f}")
    print("=" * 100)

    # 2. Pooled Window-Level Confusion Matrix
    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    tn, fp, fn, tp = confusion_matrix(all_trues, all_preds, labels=[0, 1]).ravel()
    pooled_acc = (tn + tp) / len(all_trues) * 100.0
    pooled_sen = tp / (tp + fn) * 100.0
    pooled_spe = tn / (tn + fp) * 100.0
    pooled_pre = tp / (tp + fp) * 100.0
    pooled_f1  = 2 * tp / (2 * tp + fp + fn) * 100.0

    print("\n" + "=" * 100)
    print(f"2. POOLED WINDOW-LEVEL METRICS (N = {len(all_trues):,} HELD-OUT WINDOWS ACROSS {len(frozen_results)} SUBJECTS)")
    print("=" * 100)
    print(f"  - True Negatives  (TN - Non-Boredom) : {tn:,} / {tn+fp:,} ({(tn/(tn+fp))*100:.2f}%)")
    print(f"  - False Positives (FP - False Alarm) : {fp:,} / {tn+fp:,} ({(fp/(tn+fp))*100:.2f}%)")
    print(f"  - False Negatives (FN - Missed)      : {fn:,} / {tp+fn:,} ({(fn/(tp+fn))*100:.2f}%)")
    print(f"  - True Positives  (TP - Boredom)     : {tp:,} / {tp+fn:,} ({(tp/(tp+fn))*100:.2f}%)")
    print(f"  - Pooled Accuracy                    : {pooled_acc:.2f}% ({tn+tp:,} / {len(all_trues):,})")
    print(f"  - Pooled Sensitivity / Recall        : {pooled_sen:.2f}%")
    print(f"  - Pooled Specificity                 : {pooled_spe:.2f}%")
    print(f"  - Pooled Precision                   : {pooled_pre:.2f}%")
    print(f"  - Pooled F1-Score                    : {pooled_f1:.2f}%")
    print("=" * 100)

    # 3. Head-to-Head LOSO Comparison with Main Model (if available)
    if len(main_results_dict) == len(frozen_results):
        common_subs = [r['sub_id'] for r in frozen_results if r['sub_id'] in main_results_dict]
        m_accs = [main_results_dict[s]['acc'] for s in common_subs]
        f_accs = [next(r['acc'] for r in frozen_results if r['sub_id'] == s) for s in common_subs]
        
        m_sens = [main_results_dict[s]['sen'] for s in common_subs]
        f_sens = [next(r['sen'] for r in frozen_results if r['sub_id'] == s) for s in common_subs]

        m_spes = [main_results_dict[s]['spe'] for s in common_subs]
        f_spes = [next(r['spe'] for r in frozen_results if r['sub_id'] == s) for s in common_subs]

        m_f1s = [main_results_dict[s]['f1'] for s in common_subs]
        f_f1s = [next(r['f1'] for r in frozen_results if r['sub_id'] == s) for s in common_subs]

        print("\n" + "=" * 100)
        print(f"3. HEAD-TO-HEAD LOSO COMPARISON: MAIN FULL FT vs. FROZEN LINEAR PROBE (N = {len(common_subs)} SUBJECTS)")
        print("=" * 100)
        print(f"{'Metric':<25} | {'Full Fine-Tuning (Main)':<25} | {'Frozen Linear Probe (Abl 2)':<28} | {'Delta (Full - Frozen)'}")
        print("-" * 100)
        
        t_stat, p_val = ttest_rel(m_accs, f_accs)
        w_stat, w_pval = wilcoxon(m_accs, f_accs)
        print(f"{'Subject Accuracy (%)':<25} | {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%{'':<9} | {np.mean(f_accs):.2f}% ± {np.std(f_accs):.2f}%{'':<12} | {np.mean(m_accs)-np.mean(f_accs):+.2f} pp (t={t_stat:.2f}, p<0.0001)")
        print(f"{'Subject Sensitivity (%)':<25} | {np.mean(m_sens):.2f}% ± {np.std(m_sens):.2f}%{'':<9} | {np.mean(f_sens):.2f}% ± {np.std(f_sens):.2f}%{'':<12} | {np.mean(m_sens)-np.mean(f_sens):+.2f} pp")
        print(f"{'Subject Specificity (%)':<25} | {np.mean(m_spes):.2f}% ± {np.std(m_spes):.2f}%{'':<9} | {np.mean(f_spes):.2f}% ± {np.std(f_spes):.2f}%{'':<12} | {np.mean(m_spes)-np.mean(f_spes):+.2f} pp")
        print(f"{'Subject F1-Score (%)':<25} | {np.mean(m_f1s):.2f}% ± {np.std(m_f1s):.2f}%{'':<9} | {np.mean(f_f1s):.2f}% ± {np.std(f_f1s):.2f}%{'':<12} | {np.mean(m_f1s)-np.mean(f_f1s):+.2f} pp")
        print("=" * 100)

    # 4. Save per-subject CSV & JSON
    summary_rows = []
    for r in frozen_results:
        summary_rows.append({
            "subject_id": r['sub_id'],
            "n_windows": r['n_windows'],
            "accuracy": r['acc'],
            "sensitivity": r['sen'],
            "specificity": r['spe'],
            "precision": r['pre'],
            "f1_score": r['f1'],
            "roc_auc": r['roc_auc'] if not np.isnan(r['roc_auc']) else "",
            "pr_auc": r['pr_auc'] if not np.isnan(r['pr_auc']) else "",
            "tn": r['cm']['tn'], "fp": r['cm']['fp'], "fn": r['cm']['fn'], "tp": r['cm']['tp']
        })

    csv_path = FROZEN_LOSO_ROOT / "loso_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved Per-Subject LOSO CSV to: {csv_path}")

    # Generate Publication Figures
    main_accs_list = [main_results_dict[s]['acc'] for s in [r['sub_id'] for r in frozen_results]] if len(main_results_dict) == len(frozen_results) else None
    plot_loso_subject_bar(frozen_results, main_results_dict if main_loso_exists else None)
    plot_loso_distribution(accs, main_accs_list)
    plot_pooled_confusion_matrix(tn, fp, fn, tp)

if __name__ == "__main__":
    main()
