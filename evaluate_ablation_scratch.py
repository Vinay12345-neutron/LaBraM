#!/usr/bin/env python3
"""
Evaluate Ablation 1 (Trained from Scratch) across all 5 Folds:
1. Computes Fold 1-5 metrics: ACC, SEN, SPE, PRE, REC, F1-S, ROC-AUC, PR-AUC.
2. Computes Mean ± SD across the 5 folds.
3. Computes the Pooled / Aggregated Confusion Matrix across all 18,396 test windows.
4. Generates a publication-grade comparison table between Pretrained LaBraM and Scratch LaBraM.
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from modeling_finetune import labram_base_patch200_200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRATCH_ROOT = Path("runs/ablation_scratch")
MAIN_ROOT = Path("runs/boredom_cv")

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

def evaluate_fold(fold_id, root_dir):
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
                input_chans_list = get_input_chans(ch_names)
                input_chans = torch.tensor(input_chans_list, dtype=torch.long).to(DEVICE)

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
    sen = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
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
        "rec": sen,
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
    print("EVALUATING ABLATION 1: RANDOMLY INITIALIZED LABRAM (TRAINED FROM SCRATCH)")
    print("=" * 95)

    scratch_results = []
    main_results = []

    all_scratch_trues, all_scratch_preds = [], []
    all_main_trues, all_main_preds = [], []

    for fold_id in range(5):
        print(f"Evaluating Fold {fold_id + 1}...")
        res_s = evaluate_fold(fold_id, SCRATCH_ROOT)
        scratch_results.append(res_s)
        all_scratch_trues.extend(res_s['trues'])
        all_scratch_preds.extend(res_s['preds'])

        if MAIN_ROOT.exists():
            res_m = evaluate_fold(fold_id, MAIN_ROOT)
            main_results.append(res_m)
            all_main_trues.extend(res_m['trues'])
            all_main_preds.extend(res_m['preds'])

    # Fold-wise Table for Scratch Model
    print("\n" + "=" * 95)
    print("ABLATION 1 (TRAINED FROM SCRATCH): 5-FOLD SUBJECT-DISJOINT CV RESULTS")
    print("=" * 95)
    print(f"{'Fold':<10} | {'ACC (%)':<10} | {'SEN / REC (%)':<15} | {'SPE (%)':<10} | {'PRE (%)':<10} | {'F1-S (%)':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print("-" * 95)
    for r in scratch_results:
        print(f"Fold {r['fold']:<5} | {r['acc']:<10.2f} | {r['sen']:<15.2f} | {r['spe']:<10.2f} | {r['pre']:<10.2f} | {r['f1']:<10.2f} | {r['roc_auc']:<10.4f} | {r['pr_auc']:<10.4f}")

    # Mean ± SD
    accs = [r['acc'] for r in scratch_results]
    sens = [r['sen'] for r in scratch_results]
    spes = [r['spe'] for r in scratch_results]
    pres = [r['pre'] for r in scratch_results]
    f1s = [r['f1'] for r in scratch_results]
    rocs = [r['roc_auc'] for r in scratch_results]
    prs = [r['pr_auc'] for r in scratch_results]

    print("-" * 95)
    print(f"{'Mean ± SD':<10} | {np.mean(accs):.2f}±{np.std(accs):.2f}  | {np.mean(sens):.2f}±{np.std(sens):.2f}       | {np.mean(spes):.2f}±{np.std(spes):.2f}  | {np.mean(pres):.2f}±{np.std(pres):.2f}  | {np.mean(f1s):.2f}±{np.std(f1s):.2f}  | {np.mean(rocs):.4f}±{np.std(rocs):.4f} | {np.mean(prs):.4f}±{np.std(prs):.4f}")
    print("=" * 95)

    # Aggregated Confusion Matrix
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(all_scratch_trues, all_scratch_preds, labels=[0, 1]).ravel()
    print("\n" + "=" * 95)
    print(f"POOLED CONFUSION MATRIX ACROSS ALL 18,396 HELD-OUT TEST WINDOWS (SCRATCH MODEL)")
    print("=" * 95)
    print(f"  - True Negatives  (TN - Non-Boredom) : {tn_s:,} / 9,198 ({(tn_s/9198)*100:.2f}%)")
    print(f"  - False Positives (FP - Error)        : {fp_s:,} / 9,198 ({(fp_s/9198)*100:.2f}%)")
    print(f"  - False Negatives (FN - Error)        : {fn_s:,} / 9,198 ({(fn_s/9198)*100:.2f}%)")
    print(f"  - True Positives  (TP - Boredom)     : {tp_s:,} / 9,198 ({(tp_s/9198)*100:.2f}%)")
    print(f"  - Overall Pooled Accuracy            : {((tp_s + tn_s) / 18396) * 100:.2f}%")
    print("=" * 95)

    # Direct Comparison Table
    if main_results:
        m_accs = [r['acc'] for r in main_results]
        m_sens = [r['sen'] for r in main_results]
        m_spes = [r['spe'] for r in main_results]
        m_pres = [r['pre'] for r in main_results]
        m_f1s = [r['f1'] for r in main_results]
        m_rocs = [r['roc_auc'] for r in main_results]
        m_prs = [r['pr_auc'] for r in main_results]

        print("\n" + "=" * 95)
        print("HEAD-TO-HEAD COMPARISON: PRETRAINED LABRAM vs. RANDOMLY INITIALIZED LABRAM")
        print("=" * 95)
        print(f"{'Metric':<25} | {'Pretrained LaBraM (Main)':<25} | {'Trained from Scratch':<25} | {'Delta (Pretrained Gain)'}")
        print("-" * 95)
        print(f"{'Accuracy (%)':<25} | {np.mean(m_accs):.2f}% ± {np.std(m_accs):.2f}%{'':<9} | {np.mean(accs):.2f}% ± {np.std(accs):.2f}%{'':<9} | {np.mean(m_accs) - np.mean(accs):+.2f}%")
        print(f"{'Sensitivity / Recall (%)':<25} | {np.mean(m_sens):.2f}% ± {np.std(m_sens):.2f}%{'':<9} | {np.mean(sens):.2f}% ± {np.std(sens):.2f}%{'':<9} | {np.mean(m_sens) - np.mean(sens):+.2f}%")
        print(f"{'Specificity (%)':<25} | {np.mean(m_spes):.2f}% ± {np.std(m_spes):.2f}%{'':<9} | {np.mean(spes):.2f}% ± {np.std(spes):.2f}%{'':<9} | {np.mean(m_spes) - np.mean(spes):+.2f}%")
        print(f"{'Precision (%)':<25} | {np.mean(m_pres):.2f}% ± {np.std(m_pres):.2f}%{'':<9} | {np.mean(pres):.2f}% ± {np.std(pres):.2f}%{'':<9} | {np.mean(m_pres) - np.mean(pres):+.2f}%")
        print(f"{'F1-Score (%)':<25} | {np.mean(m_f1s):.2f}% ± {np.std(m_f1s):.2f}%{'':<9} | {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%{'':<9} | {np.mean(m_f1s) - np.mean(f1s):+.2f}%")
        print(f"{'ROC-AUC':<25} | {np.mean(m_rocs):.4f} ± {np.std(m_rocs):.4f}{'':<11} | {np.mean(rocs):.4f} ± {np.std(rocs):.4f}{'':<11} | {np.mean(m_rocs) - np.mean(rocs):+.4f}")
        print(f"{'PR-AUC':<25} | {np.mean(m_prs):.4f} ± {np.std(m_prs):.4f}{'':<11} | {np.mean(prs):.4f} ± {np.std(prs):.4f}{'':<11} | {np.mean(m_prs) - np.mean(prs):+.4f}")
        print("=" * 95)

if __name__ == "__main__":
    main()
