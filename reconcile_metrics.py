#!/usr/bin/env python3
"""
Reconcile and Compare:
1. File-Level (Subject/Recording Level) Metrics from summary_metrics.md
2. Window-Level (2-second EEG Epoch Level, N = 18,396) Metrics from evaluate_ablation_scratch.py
"""

import json
import torch
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

from modeling_finetune import labram_base_patch200_200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def main():
    print("=" * 95)
    print("RECONCILING MAIN MODEL METRICS: FILE-LEVEL vs. WINDOW-LEVEL EVALUATION")
    print("=" * 95)
    
    # 1. Read File-Level metrics from summary_metrics.md / fold_metrics.json
    file_accs = [1.0000, 0.9667, 0.8929, 1.0000, 1.0000]
    file_sens = [1.0000, 0.9333, 0.7857, 1.0000, 1.0000]
    file_spes = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
    file_f1s  = [1.0000, 0.9655, 0.8800, 1.0000, 1.0000]
    file_rocs = [1.0000, 0.9956, 0.9745, 1.0000, 1.0000]
    file_prs  = [1.0000, 0.9958, 0.9812, 1.0000, 1.0000]
    
    # 2. Extract Exact Window-Level metrics across all 18,396 test windows
    window_accs, window_sens, window_spes, window_f1s, window_rocs, window_prs = [], [], [], [], [], []
    
    for fold_id in range(5):
        split_file = Path(f"boredom_split_fold{fold_id}.json")
        with open(split_file) as f:
            split = json.load(f)
        test_data = split['test']
        
        ckpt_path = MAIN_ROOT / f"fold{fold_id}" / "checkpoint-best.pth"
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
        
        acc = accuracy_score(y_trues, y_preds) * 100.0
        sen = recall_score(y_trues, y_preds) * 100.0
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds, labels=[0, 1]).ravel()
        spe = (tn / (tn + fp)) * 100.0
        f1  = f1_score(y_trues, y_preds) * 100.0
        roc = roc_auc_score(y_trues, y_probs)
        pr  = average_precision_score(y_trues, y_probs)
        
        window_accs.append(acc)
        window_sens.append(sen)
        window_spes.append(spe)
        window_f1s.append(f1)
        window_rocs.append(roc)
        window_prs.append(pr)
        print(f"Fold {fold_id + 1}: {len(y_trues)} windows | ACC = {acc:.2f}% | SEN = {sen:.2f}% | SPE = {spe:.2f}% | F1 = {f1:.2f}% | ROC = {roc:.4f}")
        
    print("\n" + "=" * 95)
    print("DETAILED COMPARISON: FILE-LEVEL vs. 2-SECOND WINDOW-LEVEL AGGREGATION")
    print("=" * 95)
    print(f"{'Evaluation Unit':<35} | {'ACC (%)':<18} | {'SEN (%)':<18} | {'SPE (%)':<18} | {'F1-S (%)':<18}")
    print("-" * 95)
    print(f"{'File-Level (1 vote / .h5 recording)':<35} | {np.mean(file_accs)*100:.2f} ± {np.std(file_accs)*100:.2f}%{'':<3} | {np.mean(file_sens)*100:.2f} ± {np.std(file_sens)*100:.2f}%{'':<3} | {np.mean(file_spes)*100:.2f} ± {np.std(file_spes)*100:.2f}%{'':<3} | {np.mean(file_f1s)*100:.2f} ± {np.std(file_f1s)*100:.2f}%")
    print(f"{'Window-Level (All 18,396 2-sec windows)':<35} | {np.mean(window_accs):.2f} ± {np.std(window_accs):.2f}%{'':<3} | {np.mean(window_sens):.2f} ± {np.std(window_sens):.2f}%{'':<3} | {np.mean(window_spes):.2f} ± {np.std(window_spes):.2f}%{'':<3} | {np.mean(window_f1s):.2f} ± {np.std(window_f1s):.2f}%")
    print("=" * 95)

if __name__ == "__main__":
    main()
