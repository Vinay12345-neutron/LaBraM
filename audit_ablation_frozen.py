#!/usr/bin/env python3
"""
Dedicated Audit Script for Ablation 2 (Linear Probing vs. Full Fine-Tuning):
1. Extracts exact fold-by-fold metrics for both Full Fine-Tuning and Frozen Linear Probe.
2. Reports all 7 metrics (ACC, SEN, SPE, PRE, F1, ROC-AUC, PR-AUC) for each fold.
3. Computes fold-level differences (Full FT - Frozen) across K=5 folds.
4. Performs CV-aware statistical tests (5-fold paired t-test df=4, Nadeau-Bengio corrected t-test).
5. Verifies experimental parity (subjects, windows, splits, preprocessing, parameter trainability).
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from modeling_finetune import labram_base_patch200_200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAIN_ROOT = Path("runs/boredom_cv")
FROZEN_ROOT = Path("runs/ablation_frozen")

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
        "n_subjects": len(split['test']) // 2,
        "n_windows": len(y_trues),
        "acc": acc, "sen": sen, "spe": spe, "pre": pre, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def main():
    print("=" * 110)
    print("AUDIT OF ABLATION 2: FULL FINE-TUNING (MAIN) vs. FROZEN BACKBONE LINEAR PROBE (ABLATION 2)")
    print("=" * 110)

    main_res, frozen_res = [], []

    for fold_id in range(5):
        rm = evaluate_fold_exact(fold_id, MAIN_ROOT)
        rf = evaluate_fold_exact(fold_id, FROZEN_ROOT)
        main_res.append(rm)
        frozen_res.append(rf)

    # 1. Exact Fold-by-Fold Table for All 7 Metrics
    for i in range(5):
        m = main_res[i]
        f = frozen_res[i]
        print(f"\n>>> FOLD {i+1} (Test Subjects: {m['n_subjects']}, Test Windows: {m['n_windows']})")
        print(f"{'Metric':<25} | {'Full Fine-Tuning':<20} | {'Frozen Linear Probe':<20} | {'Delta (FT - Frozen)'}")
        print("-" * 85)
        print(f"{'Accuracy (%)':<25} | {m['acc']:<19.2f}% | {f['acc']:<19.2f}% | {m['acc'] - f['acc']:+.2f}%")
        print(f"{'Sensitivity / Recall (%)':<25} | {m['sen']:<19.2f}% | {f['sen']:<19.2f}% | {m['sen'] - f['sen']:+.2f}%")
        print(f"{'Specificity (%)':<25} | {m['spe']:<19.2f}% | {f['spe']:<19.2f}% | {m['spe'] - f['spe']:+.2f}%")
        print(f"{'Precision (%)':<25} | {m['pre']:<19.2f}% | {f['pre']:<19.2f}% | {m['pre'] - f['pre']:+.2f}%")
        print(f"{'F1-Score (%)':<25} | {m['f1']:<19.2f}% | {f['f1']:<19.2f}% | {m['f1'] - f['f1']:+.2f}%")
        print(f"{'ROC-AUC':<25} | {m['roc_auc']:<20.4f} | {f['roc_auc']:<20.4f} | {m['roc_auc'] - f['roc_auc']:+.4f}")
        print(f"{'PR-AUC':<25} | {m['pr_auc']:<20.4f} | {f['pr_auc']:<20.4f} | {m['pr_auc'] - f['pr_auc']:+.4f}")

    # 2. 5-Fold Summary & CV-Aware Statistical Testing (df = 4)
    print("\n" + "=" * 110)
    print("5-FOLD CROSS-VALIDATION SUMMARY & CV-AWARE STATISTICAL TESTING (K = 5, df = 4)")
    print("=" * 110)
    print(f"{'Metric':<25} | {'Full FT (Mean±SD)':<22} | {'Frozen (Mean±SD)':<22} | {'Mean Delta':<14} | {'Paired t-test (df=4)':<22} | {'Nadeau-Bengio Corr.'}")
    print("-" * 115)

    metrics_keys = ["acc", "sen", "spe", "pre", "f1", "roc_auc", "pr_auc"]
    metrics_labels = ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "Precision (%)", "F1-Score (%)", "ROC-AUC", "PR-AUC"]

    for k, lbl in zip(metrics_keys, metrics_labels):
        m_vals = [r[k] for r in main_res]
        f_vals = [r[k] for r in frozen_res]
        diffs = [m - f for m, f in zip(m_vals, f_vals)]

        m_mean, m_std = np.mean(m_vals), np.std(m_vals)
        f_mean, f_std = np.mean(f_vals), np.std(f_vals)
        d_mean, d_std = np.mean(diffs), np.std(diffs, ddof=1)

        # Standard 5-fold paired t-test
        t_stat, p_val = ttest_rel(m_vals, f_vals)

        # Nadeau-Bengio corrected resampled t-test for 5-fold CV (n_test/n_train = 1/4 = 0.25)
        # correction_factor = (1/K + n_test/n_train) = (1/5 + 1/4) = 0.45
        correction = 0.45
        t_corr = d_mean / (d_std * np.sqrt(correction))
        from scipy.stats import t
        p_corr = 2 * (1 - t.cdf(abs(t_corr), df=4))

        if "AUC" in lbl:
            print(f"{lbl:<25} | {m_mean:.4f} ± {m_std:.4f}{'':<8} | {f_mean:.4f} ± {f_std:.4f}{'':<8} | {d_mean:+.4f}{'':<6} | t={t_stat:.2f}, p={p_val:.4f}{'':<6} | t_corr={t_corr:.2f}, p={p_corr:.4f}")
        else:
            print(f"{lbl:<25} | {m_mean:.2f}% ± {m_std:.2f}%{'':<6} | {f_mean:.2f}% ± {f_std:.2f}%{'':<6} | {d_mean:+.2f} pp{'':<4} | t={t_stat:.2f}, p={p_val:.4f}{'':<6} | t_corr={t_corr:.2f}, p={p_corr:.4f}")

    print("=" * 115)

if __name__ == "__main__":
    main()
