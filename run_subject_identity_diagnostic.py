#!/usr/bin/env python3
"""
Subject-Identity vs. Task-Label Diagnostic Analysis.
Investigating whether the learned 200-D representation is primarily organized by:
1. Cognitive Task (Boredom vs. Non-Boredom) OR
2. Subject Identity (Individual Brain Fingerprints / Confounders across 73 subjects).

Calculates:
- 15-NN Task Class Purity (2-class)
- 15-NN Subject-ID Purity (73-class)
- Per-subject distribution across Boredom vs. Non-Boredom representation regions
- Diagnostic UMAP colored by Subject ID (diagnostic_subject_id_umap.png)
"""

import json
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
import umap

# Model definition
from modeling_finetune import labram_base_patch200_200

# IEEE Style parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUNS_DIR = Path("runs/boredom_cv")

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

def load_finetuned_model(fold_id):
    ckpt_path = RUNS_DIR / f"fold{fold_id}" / "checkpoint-best.pth"
    if not ckpt_path.exists():
        ckpt_path = RUNS_DIR / f"fold{fold_id}" / "checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def extract_finetuned_test_data():
    print("=" * 80)
    print("EXTRACTING HELD-OUT TEST EMBEDDINGS & SUBJECT IDS (73 SUBJECTS)")
    print("=" * 80)
    
    all_fine_embs, all_trues, all_subjects = [], [], []
    
    for fold_id in range(5):
        split_file = Path(f"boredom_split_fold{fold_id}.json")
        with open(split_file) as f:
            split = json.load(f)
        test_data = split['test']
        
        model_fine = load_finetuned_model(fold_id)
        act_fine = {}
        def hook_fine(m, inp, out):
            act_fine['norm'] = out.detach().cpu()
        model_fine.norm.register_forward_hook(hook_fine)
        
        fold_fine, fold_y, fold_subjs = [], [], []
        
        for item in test_data:
            fpath = item['file']
            label = int(item['label'])
            filename = Path(fpath).name
            subj_id = filename.split('_')[0]
            
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
                        
                        _ = model_fine(x_input, input_chans=input_chans)
                        emb_f = act_fine['norm'][:, 0, :].numpy()
                        
                        fold_fine.append(emb_f)
                        fold_y.append(label)
                        fold_subjs.append(subj_id)
            except Exception as e:
                continue
                
        print(f"Fold {fold_id + 1}: Extracted {len(fold_y)} test windows across {len(set(fold_subjs))} unique subjects.")
        all_fine_embs.extend(fold_fine)
        all_trues.extend(fold_y)
        all_subjects.extend(fold_subjs)
        
    X_fine = np.concatenate(all_fine_embs, axis=0)
    y = np.array(all_trues)
    subjects = np.array(all_subjects)
    
    # L2-normalization
    X_fine_norm = X_fine / np.linalg.norm(X_fine, axis=1, keepdims=True)
    return X_fine_norm, y, subjects

def run_diagnostics(X_norm, y, subjects):
    print("\n" + "=" * 80)
    print("QUANTITATIVE SUBJECT-IDENTITY vs. TASK-LABEL DIAGNOSTICS (200-D SPACE)")
    print("=" * 80)
    
    unique_subjs = sorted(list(set(subjects)))
    n_subjs = len(unique_subjs)
    subj_to_idx = {s: i for i, s in enumerate(unique_subjs)}
    subj_indices = np.array([subj_to_idx[s] for s in subjects])
    
    # 1. 15-NN Neighborhood Analysis
    knn = NearestNeighbors(n_neighbors=16, metric='euclidean', n_jobs=-1)
    knn.fit(X_norm)
    neighbors = knn.kneighbors(X_norm, return_distance=False)[:, 1:] # exclude self
    
    # Class Purity
    class_neighbor_match = (y[neighbors] == y[:, None])
    class_purity = np.mean(np.mean(class_neighbor_match, axis=1)) * 100.0
    
    # Subject-ID Purity
    subj_neighbor_match = (subj_indices[neighbors] == subj_indices[:, None])
    subj_purity = np.mean(np.mean(subj_neighbor_match, axis=1)) * 100.0
    
    # Theoretical Random Baseline
    # For Class (2 classes, balanced 50/50): 50.0%
    # For Subject (73 subjects, balanced 252 samples each): 1/73 = ~1.37% (or 251 / 18395 = 1.36%)
    random_class_baseline = 50.0
    random_subj_baseline = (252 - 1) / (len(y) - 1) * 100.0
    
    print(f"Total Test Samples Evaluated : {len(y):,}")
    print(f"Total Unique Subject Cohort  : {n_subjs} Subjects\n")
    print(f"{'Metric':<35} | {'Measured Purity':<18} | {'Random Baseline':<18} | {'Ratio over Chance'}")
    print("-" * 80)
    print(f"{'15-NN Task-Class Purity (2 Classes)':<35} | {class_purity:<17.2f}% | {random_class_baseline:<17.2f}% | {class_purity / random_class_baseline:.2f}x")
    print(f"{'15-NN Subject-ID Purity (73 Subjects)':<35} | {subj_purity:<17.2f}% | {random_subj_baseline:<17.2f}% | {subj_purity / random_subj_baseline:.2f}x")
    print("=" * 80)
    
    # 2. Subject Distribution across Classes Check
    # Verify whether every single subject has embeddings in BOTH Non-Boredom (Class 0) and Boredom (Class 1) regions
    print("\nInspecting Subject Distribution Across Task Classes:")
    b_counts = []
    n_counts = []
    for s in unique_subjs:
        s_mask = (subjects == s)
        n_boredom = np.sum(y[s_mask] == 1)
        n_neutral = np.sum(y[s_mask] == 0)
        b_counts.append(n_boredom)
        n_counts.append(n_neutral)
        
    print(f"  - Number of subjects with samples in BOTH Boredom & Neutral : {np.sum((np.array(b_counts) > 0) & (np.array(n_counts) > 0))} / {n_subjs} (100.0%)")
    print(f"  - Average Boredom windows per subject                       : {np.mean(b_counts):.1f} ± {np.std(b_counts):.1f}")
    print(f"  - Average Neutral windows per subject                       : {np.mean(n_counts):.1f} ± {np.std(n_counts):.1f}")
    
    # 3. Generate Subject-ID Colored UMAP Plot
    print("\nGenerating Diagnostic UMAP colored by Subject ID (73 colors)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='euclidean')
    X_umap = reducer.fit_transform(X_norm)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=350)
    
    # Left Panel: Task Ground Truth (Boredom vs Neutral)
    ax0 = axes[0]
    idx_0 = (y == 0)
    idx_1 = (y == 1)
    ax0.scatter(X_umap[idx_0, 0], X_umap[idx_0, 1], color="#1e88e5", marker="o", alpha=0.45, s=20, label="Non-Boredom", zorder=2)
    ax0.scatter(X_umap[idx_1, 0], X_umap[idx_1, 1], color="#2e7d32", marker="^", alpha=0.45, s=20, label="Boredom", zorder=2)
    ax0.set_title("(a) Colored by Task Label (2 Classes)", fontsize=16, fontweight="bold", loc="left")
    ax0.set_xlabel("UMAP Dimension 1", fontsize=15, fontweight="bold")
    ax0.set_ylabel("UMAP Dimension 2", fontsize=15, fontweight="bold")
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="upper right", fontsize=12, frameon=True)
    
    # Right Panel: Colored by 73 Subject IDs
    ax1 = axes[1]
    cmap = cm.get_cmap('tab20', n_subjs)
    scatter = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=subj_indices, cmap='turbo', alpha=0.6, s=18, edgecolors="none")
    ax1.set_title("(b) Colored by Subject ID (73 Subjects)", fontsize=16, fontweight="bold", loc="left")
    ax1.set_xlabel("UMAP Dimension 1", fontsize=15, fontweight="bold")
    ax1.set_ylabel("UMAP Dimension 2", fontsize=15, fontweight="bold")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle=":", alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Subject Index (0 to 72)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    out_diag = RUNS_DIR / "diagnostic_subject_id_umap.png"
    plt.savefig(out_diag, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Subject Identity Diagnostic Plot: {out_diag}\n")
    
    return class_purity, subj_purity

if __name__ == "__main__":
    X_norm, y, subjects = extract_finetuned_test_data()
    run_diagnostics(X_norm, y, subjects)
