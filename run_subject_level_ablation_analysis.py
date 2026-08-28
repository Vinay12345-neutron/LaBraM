#!/usr/bin/env python3
"""
Subject-Level Analysis for Ablation 1 (Pretrained vs. Scratch):
1. Evaluates all 73 independent subjects on their held-out test windows (252 windows per subject).
2. Computes per-subject Accuracy, Sensitivity, Specificity, and F1 for both models.
3. Calculates paired differences (Pretrained - Scratch), Wins/Losses/Ties, Mean, and Median.
4. Performs non-parametric Paired Wilcoxon Signed-Rank Test across N=73 independent subjects.
5. Generates a publication-quality paired subject-level comparison plot.
"""

import json
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from modeling_finetune import labram_base_patch200_200

# IEEE Publication Typography
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAIN_ROOT = Path("runs/boredom_cv")
SCRATCH_ROOT = Path("runs/ablation_scratch")

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

def load_model(root_dir, fold_id):
    ckpt_path = root_dir / f"fold{fold_id}" / "checkpoint-best.pth"
    if not ckpt_path.exists():
        ckpt_path = root_dir / f"fold{fold_id}" / "checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def extract_subject_predictions():
    print("Extracting test predictions per subject across all 5 folds...")
    subject_data = {} # subj_id -> {"y_true": [], "y_pre": [], "y_scr": []}

    for fold_id in range(5):
        split_file = Path(f"boredom_split_fold{fold_id}.json")
        with open(split_file) as f:
            split = json.load(f)
        test_data = split['test']

        m_pre = load_model(MAIN_ROOT, fold_id)
        m_scr = load_model(SCRATCH_ROOT, fold_id)

        for item in test_data:
            fpath = item['file']
            label = int(item['label'])
            filename = Path(fpath).name
            subj_id = filename.split('_')[0]

            if subj_id not in subject_data:
                subject_data[subj_id] = {"y_true": [], "y_pre": [], "y_scr": []}

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

                        # Pretrained
                        out_p = m_pre(x_input, input_chans=input_chans)
                        pred_p = 1 if torch.sigmoid(out_p[0, 0]).item() >= 0.5 else 0

                        # Scratch
                        out_s = m_scr(x_input, input_chans=input_chans)
                        pred_s = 1 if torch.sigmoid(out_s[0, 0]).item() >= 0.5 else 0

                        subject_data[subj_id]["y_true"].append(label)
                        subject_data[subj_id]["y_pre"].append(pred_p)
                        subject_data[subj_id]["y_scr"].append(pred_s)
            except Exception:
                continue

    return subject_data

def compute_subject_metrics(subject_data):
    subj_ids = sorted(list(subject_data.keys()), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    print(f"Total Unique Subjects Analyzed: {len(subj_ids)}")

    results = []
    for sid in subj_ids:
        y_true = np.array(subject_data[sid]["y_true"])
        y_pre  = np.array(subject_data[sid]["y_pre"])
        y_scr  = np.array(subject_data[sid]["y_scr"])

        # Pretrained metrics
        acc_p = accuracy_score(y_true, y_pre) * 100.0
        sen_p = recall_score(y_true, y_pre, zero_division=0) * 100.0
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true, y_pre, labels=[0, 1]).ravel()
        spe_p = (tn_p / (tn_p + fp_p)) * 100.0 if (tn_p + fp_p) > 0 else 0.0
        f1_p  = f1_score(y_true, y_pre, zero_division=0) * 100.0

        # Scratch metrics
        acc_s = accuracy_score(y_true, y_scr) * 100.0
        sen_s = recall_score(y_true, y_scr, zero_division=0) * 100.0
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, y_scr, labels=[0, 1]).ravel()
        spe_s = (tn_s / (tn_s + fp_s)) * 100.0 if (tn_s + fp_s) > 0 else 0.0
        f1_s  = f1_score(y_true, y_scr, zero_division=0) * 100.0

        results.append({
            "subject": sid,
            "n_windows": len(y_true),
            "acc_pre": acc_p, "acc_scr": acc_s, "acc_diff": acc_p - acc_s,
            "sen_pre": sen_p, "sen_scr": sen_s, "sen_diff": sen_p - sen_s,
            "spe_pre": spe_p, "spe_scr": spe_s, "spe_diff": spe_p - spe_s,
            "f1_pre":  f1_p,  "f1_scr":  f1_s,  "f1_diff":  f1_p - f1_s,
        })

    return results

def report_analysis(results):
    print("\n" + "=" * 90)
    print("SUBJECT-LEVEL PERFORMANCE COMPARISON (N = 73 INDEPENDENT HUMAN SUBJECTS)")
    print("=" * 90)

    acc_p = np.array([r["acc_pre"] for r in results])
    acc_s = np.array([r["acc_scr"] for r in results])
    acc_d = np.array([r["acc_diff"] for r in results])

    sen_d = np.array([r["sen_diff"] for r in results])
    spe_d = np.array([r["spe_diff"] for r in results])
    f1_d  = np.array([r["f1_diff"] for r in results])

    def summarize_metric(diffs, name):
        wins = np.sum(diffs > 1e-4)
        losses = np.sum(diffs < -1e-4)
        ties = np.sum(np.abs(diffs) <= 1e-4)
        mean_d = np.mean(diffs)
        median_d = np.median(diffs)
        std_d = np.std(diffs)
        return {
            "name": name, "mean": mean_d, "median": median_d, "std": std_d,
            "wins": wins, "losses": losses, "ties": ties
        }

    s_acc = summarize_metric(acc_d, "Accuracy (%)")
    s_sen = summarize_metric(sen_d, "Sensitivity (%)")
    s_spe = summarize_metric(spe_d, "Specificity (%)")
    s_f1  = summarize_metric(f1_d,  "F1-Score (%)")

    print(f"{'Metric':<18} | {'Mean Diff':<14} | {'Median Diff':<14} | {'Pretrained > Scratch (Wins)':<30} | {'Scratch > Pretrained':<20} | {'Ties'}")
    print("-" * 115)
    for s in [s_acc, s_sen, s_spe, s_f1]:
        print(f"{s['name']:<18} | {s['mean']:+.2f}% ± {s['std']:.2f}% | {s['median']:+.2f}%{'':<6} | {s['wins']:<2} / 73 ({s['wins']/73*100:.1f}%)                 | {s['losses']:<2} / 73 ({s['losses']/73*100:.1f}%)        | {s['ties']:<2} / 73 ({s['ties']/73*100:.1f}%)")
    print("=" * 115)

    # Statistical significance across N=73 subjects
    # Non-parametric Wilcoxon Signed-Rank Test (paired)
    w_stat, w_pval = wilcoxon(acc_p, acc_s, alternative='two-sided')
    t_stat, t_pval = ttest_rel(acc_p, acc_s)

    print("\n" + "=" * 90)
    print("FORMAL SUBJECT-LEVEL STATISTICAL SIGNIFICANCE TESTING (N = 73 SUBJECTS)")
    print("=" * 90)
    print(f"  - Paired Wilcoxon Signed-Rank Test : W = {w_stat:.1f}, p = {w_pval:.4e}")
    print(f"  - Paired Student's t-test          : t = {t_stat:.3f}, p = {t_pval:.4e}")
    if w_pval < 0.05:
        print(f"  - Conclusion: The pretraining accuracy advantage across the 73 subjects is STATISTICALLY SIGNIFICANT (p = {w_pval:.4e} < 0.05).")
    else:
        print(f"  - Conclusion: Difference is not statistically significant at alpha = 0.05.")
    print("=" * 90)

    return results

def plot_subject_level_comparison(results):
    """Generate paired subject-level accuracy comparison figure."""
    subj_names = [r["subject"] for r in results]
    acc_p = [r["acc_pre"] for r in results]
    acc_s = [r["acc_scr"] for r in results]
    acc_diff = [r["acc_diff"] for r in results]
    N = len(results)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), dpi=350, gridspec_kw={'height_ratios': [1.2, 1.0]})

    x = np.arange(N)
    width = 0.38

    # 1. Paired Bar Plot for each of the 73 subjects
    ax1 = axes[0]
    ax1.bar(x - width/2, acc_p, width, label="Pretrained LaBraM", color="#1e88e5", alpha=0.9, edgecolor="none")
    ax1.bar(x + width/2, acc_s, width, label="Trained from Scratch", color="#e53935", alpha=0.85, edgecolor="none")

    ax1.set_ylabel("Accuracy (%)", fontsize=15, fontweight="bold")
    ax1.set_ylim([70, 103])
    ax1.set_xticks(x)
    ax1.set_xticklabels(subj_names, rotation=90, fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax1.legend(loc="lower left", fontsize=13, frameon=True, framealpha=0.95, edgecolor="#cccccc")

    # 2. Subject-by-Subject Delta (Pretrained - Scratch)
    ax2 = axes[1]
    colors = ["#2e7d32" if d > 0 else ("#d32f2f" if d < 0 else "#757575") for d in acc_diff]
    ax2.bar(x, acc_diff, width=0.6, color=colors, edgecolor="none", alpha=0.9)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.axhline(np.mean(acc_diff), color="#1565c0", linestyle="--", linewidth=1.8,
                label=f"Mean Advantage: +{np.mean(acc_diff):.2f}%")

    ax2.set_xlabel("Participant ID (73 Unique Subjects)", fontsize=15, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Δ Accuracy (%)\n(Pretrained − Scratch)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(subj_names, rotation=90, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax2.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    out_path = SCRATCH_ROOT / "subject_level_accuracy_comparison.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Paired Subject-Level Comparison Plot to: {out_path}")

def main():
    subj_data = extract_subject_predictions()
    results = compute_subject_metrics(subj_data)
    report_analysis(results)
    plot_subject_level_comparison(results)

if __name__ == "__main__":
    main()
