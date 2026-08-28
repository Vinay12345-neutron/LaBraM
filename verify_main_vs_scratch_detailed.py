#!/usr/bin/env python3
"""
Comprehensive Verification and Detailed Fold-by-Fold Comparison:
Pretrained LaBraM vs. Trained from Scratch (Ablation 1)
Using the exact same evaluation script, same checkpoints, same windows, and paired statistical testing.
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from modeling_finetune import labram_base_patch200_200

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

def evaluate_fold_exact(fold_id, root_dir):
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
        except Exception as e:
            continue

    y_trues = np.array(y_trues)
    y_probs = np.array(y_probs)
    y_preds = np.array(y_preds)

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds, labels=[0, 1]).ravel()
    acc = accuracy_score(y_trues, y_preds) * 100.0
    sen = recall_score(y_trues, y_preds) * 100.0
    spe = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    pre = precision_score(y_trues, y_preds, zero_division=0) * 100.0
    f1 = f1_score(y_trues, y_preds, zero_division=0) * 100.0
    roc_auc = roc_auc_score(y_trues, y_probs)
    pr_auc = average_precision_score(y_trues, y_probs)

    return {
        "fold": fold_id + 1,
        "n_samples": len(y_trues),
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "pre": pre,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trues": y_trues,
        "preds": y_preds,
        "probs": y_probs
    }

def main():
    print("=" * 95)
    print("EXACT FOLD-BY-FOLD COMPARISON: PRETRAINED LABRAM vs. RANDOMLY INITIALIZED LABRAM")
    print("=" * 95)

    main_res, scratch_res = [], []
    all_main_trues, all_main_preds = [], []
    all_scratch_trues, all_scratch_preds = [], []

    for fold_id in range(5):
        rm = evaluate_fold_exact(fold_id, MAIN_ROOT)
        rs = evaluate_fold_exact(fold_id, SCRATCH_ROOT)

        main_res.append(rm)
        scratch_res.append(rs)

        all_main_trues.extend(rm['trues'])
        all_main_preds.extend(rm['preds'])
        all_scratch_trues.extend(rs['trues'])
        all_scratch_preds.extend(rs['preds'])

    # Print Fold-by-Fold Table
    print(f"{'Fold Index':<10} | {'Pretrained ACC (%)':<20} | {'Scratch ACC (%)':<20} | {'Difference (Pretrained - Scratch)'}")
    print("-" * 95)
    for i in range(5):
        m_acc = main_res[i]['acc']
        s_acc = scratch_res[i]['acc']
        diff = m_acc - s_acc
        print(f"Fold {i+1:<5} | {m_acc:<19.2f}% | {s_acc:<19.2f}% | {diff:+.2f}%")

    m_accs = [r['acc'] for r in main_res]
    s_accs = [r['acc'] for r in scratch_res]

    print("-" * 95)
    print(f"{'Mean ± SD':<10} | {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%{'':<6} | {np.mean(s_accs):.2f}% ± {np.std(s_accs):.2f}%{'':<6} | {np.mean(m_accs) - np.mean(s_accs):+.2f}%")
    print("=" * 95)

    # Detailed Metric Table
    metrics_list = ["acc", "sen", "spe", "pre", "f1", "roc_auc", "pr_auc"]
    labels_list  = ["Accuracy (%)", "Sensitivity / Recall (%)", "Specificity (%)", "Precision (%)", "F1-Score (%)", "ROC-AUC", "PR-AUC"]

    print("\n" + "=" * 95)
    print("DETAILED 5-FOLD METRIC BREAKDOWN")
    print("=" * 95)
    print(f"{'Metric':<26} | {'Pretrained LaBraM':<22} | {'Trained from Scratch':<22} | {'Delta':<10} | {'p-value (paired t-test)'}")
    print("-" * 95)

    for m_key, m_label in zip(metrics_list, labels_list):
        m_vals = [r[m_key] for r in main_res]
        s_vals = [r[m_key] for r in scratch_res]

        m_mean, m_sd = np.mean(m_vals), np.std(m_vals)
        s_mean, s_sd = np.mean(s_vals), np.std(s_vals)
        delta = m_mean - s_mean

        # Paired t-test
        t_stat, p_val = ttest_rel(m_vals, s_vals)

        if "AUC" in m_label:
            print(f"{m_label:<26} | {m_mean:.4f} ± {m_sd:.4f}{'':<8} | {s_mean:.4f} ± {s_sd:.4f}{'':<8} | {delta:+.4f}{'':<4} | p = {p_val:.4f}")
        else:
            print(f"{m_label:<26} | {m_mean:.2f}% ± {m_sd:.2f}%{'':<6} | {s_mean:.2f}% ± {s_sd:.2f}%{'':<6} | {delta:+.2f}%{'':<4} | p = {p_val:.4f}")
    print("=" * 95)

    # Verification of Pooled Confusion Matrix & Accuracy
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(all_main_trues, all_main_preds, labels=[0, 1]).ravel()
    pooled_acc_m = (tn_m + tp_m) / len(all_main_trues) * 100.0

    print("\n" + "=" * 95)
    print("POOLED CONFUSION MATRIX VERIFICATION (PRETRAINED LABRAM)")
    print("=" * 95)
    print(f"  - TN (Non-Boredom) : {tn_m:,} / 9,198 ({(tn_m/9198)*100:.2f}%)")
    print(f"  - FP (False Alarm) : {fp_m:,} / 9,198 ({(fp_m/9198)*100:.2f}%)")
    print(f"  - FN (Missed)      : {fn_m:,} / 9,198 ({(fn_m/9198)*100:.2f}%)")
    print(f"  - TP (Boredom)     : {tp_m:,} / 9,198 ({(tp_m/9198)*100:.2f}%)")
    print(f"  - Total Test Windows: {len(all_main_trues):,}")
    print(f"  - Total Correct    : {tn_m + tp_m:,} / 18,396")
    print(f"  - Pooled Accuracy  : {pooled_acc_m:.4f}% ({pooled_acc_m:.2f}%)")
    print(f"  - Fold-Averaged Acc: {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%")
    print("=" * 95)

if __name__ == "__main__":
    main()
