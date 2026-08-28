#!/usr/bin/env python3
"""
Scientifically Rigorous 3-Panel UMAP Manifold Analysis & Quantitative Representation Validation.
IEEE Transactions / Neurocomputing Publication Format.

Panels:
  (a) Pretrained LaBraM (Ground-Truth Classes: Non-Boredom vs Boredom)
  (b) Fine-tuned LaBraM (Ground-Truth Classes: Non-Boredom vs Boredom)
  (c) Fine-tuned LaBraM (Prediction Outcome: TN, TP, FP, FN)

Features:
  - Exact 1-to-1 paired EEG test windows (N = 18,396 from 73 held-out subjects across 5 folds).
  - Joint UMAP fitting on [X_pretrained; X_finetuned] (36,792 x 200) for shared coordinate geometry.
  - Quantitative 200-dimensional evaluation (Silhouette, k-NN purity, within/between class distances).
  - 350 DPI PNG + Vector PDF export.
"""

import json
import time
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
import umap

# Model definition
from modeling_finetune import labram_base_patch200_200

# IEEE Publication Typography
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUNS_DIR = Path("runs/boredom_cv")
PRETRAINED_CKPT = Path("checkpoints/labram-base.pth")

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

def load_pretrained_model():
    print("Loading Pretrained LaBraM Foundation Model (labram-base.pth)...")
    ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    msg = model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

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

def extract_paired_test_data():
    """Extract strictly paired (Pretrained, Finetuned) embeddings on held-out test sets across 5 folds."""
    model_pre = load_pretrained_model()
    
    act_pre = {}
    def hook_pre(m, inp, out):
        act_pre['norm'] = out.detach().cpu()
    model_pre.norm.register_forward_hook(hook_pre)
    
    all_pre_embs = []
    all_fine_embs = []
    all_trues = []
    all_preds = []
    all_subjects = []
    
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
        
        fold_pre, fold_fine, fold_y, fold_pred, fold_subjs = [], [], [], [], []
        
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
                        
                        # 1. Forward through Pretrained LaBraM
                        _ = model_pre(x_input, input_chans=input_chans)
                        emb_p = act_pre['norm'][:, 0, :].numpy() # (1, 200)
                        
                        # 2. Forward through Fine-Tuned LaBraM
                        out_f = model_fine(x_input, input_chans=input_chans)
                        emb_f = act_fine['norm'][:, 0, :].numpy() # (1, 200)
                        
                        prob = torch.sigmoid(out_f[0, 0]).item()
                        pred = 1 if prob >= 0.5 else 0
                        
                        fold_pre.append(emb_p)
                        fold_fine.append(emb_f)
                        fold_y.append(label)
                        fold_pred.append(pred)
                        fold_subjs.append(subj_id)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue
                
        print(f"Fold {fold_id + 1}: Extracted {len(fold_y)} paired windows from {len(set(fold_subjs))} test subjects.")
        all_pre_embs.extend(fold_pre)
        all_fine_embs.extend(fold_fine)
        all_trues.extend(fold_y)
        all_preds.extend(fold_pred)
        all_subjects.extend(fold_subjs)
        
    X_pre = np.concatenate(all_pre_embs, axis=0)
    X_fine = np.concatenate(all_fine_embs, axis=0)
    y = np.array(all_trues)
    y_pred = np.array(all_preds)
    
    assert len(X_pre) == len(X_fine) == len(y) == len(y_pred), "Mismatch in paired embedding dimensions!"
    print(f"\nTotal strictly paired test windows extracted: {len(y)} across {len(set(all_subjects))} subjects.")
    return X_pre, X_fine, y, y_pred, all_subjects

def compute_quantitative_metrics(X, y, state_name="Pretrained"):
    """Compute exact quantitative separation metrics on original 200-D feature space."""
    print(f"\nComputing 200-D representation metrics for {state_name}...")
    
    # 1. Silhouette Score (subsample 10k for fast/stable computation across 18k samples)
    sil_score = silhouette_score(X, y, metric='euclidean', sample_size=10000, random_state=42)
    
    # 2. k-NN Class Purity (k = 15)
    knn = NearestNeighbors(n_neighbors=16, metric='euclidean', n_jobs=-1)
    knn.fit(X)
    neighbors = knn.kneighbors(X, return_distance=False)[:, 1:] # Exclude self
    neighbor_labels = y[neighbors]
    purities = np.mean(neighbor_labels == y[:, None], axis=1)
    mean_purity = np.mean(purities) * 100.0
    
    # 3. Class Centroids & Distances
    idx_0 = (y == 0)
    idx_1 = (y == 1)
    centroid_0 = np.mean(X[idx_0], axis=0)
    centroid_1 = np.mean(X[idx_1], axis=0)
    
    # Within-class distance (mean distance of points to their respective class centroid)
    dist_within_0 = np.linalg.norm(X[idx_0] - centroid_0, axis=1)
    dist_within_1 = np.linalg.norm(X[idx_1] - centroid_1, axis=1)
    mean_within_dist = np.mean(np.concatenate([dist_within_0, dist_within_1]))
    
    # Between-class distance (distance between class centroids)
    between_dist = np.linalg.norm(centroid_0 - centroid_1)
    
    # Separation Ratio
    sep_ratio = between_dist / mean_within_dist if mean_within_dist > 0 else 0.0
    
    return {
        "state": state_name,
        "silhouette": sil_score,
        "knn_purity": mean_purity,
        "within_dist": mean_within_dist,
        "between_dist": between_dist,
        "sep_ratio": sep_ratio
    }

def main():
    X_pre, X_fine, y, y_pred, subjects = extract_paired_test_data()
    
    # -------------------------------------------------------------------------
    # Quantitative Validation (200-D Space)
    # -------------------------------------------------------------------------
    m_pre = compute_quantitative_metrics(X_pre, y, state_name="Pretrained LaBraM")
    m_fine = compute_quantitative_metrics(X_fine, y, state_name="Fine-Tuned LaBraM")
    
    print("\n" + "=" * 80)
    print("QUANTITATIVE SEPARATION METRICS (CALCULATED DIRECTLY ON ORIGINAL 200-D EMBEDDINGS)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Pretrained LaBraM':<20} | {'Fine-Tuned LaBraM':<20} | {'Change / Gain'}")
    print("-" * 80)
    print(f"{'Silhouette Score (-1 to +1)':<35} | {m_pre['silhouette']:<20.4f} | {m_fine['silhouette']:<20.4f} | {m_fine['silhouette'] - m_pre['silhouette']:+.4f}")
    print(f"{'15-NN Class Purity (%)':<35} | {m_pre['knn_purity']:<20.2f}% | {m_fine['knn_purity']:<20.2f}% | {m_fine['knn_purity'] - m_pre['knn_purity']:+.2f}%")
    print(f"{'Mean Within-Class Dist (D_w)':<35} | {m_pre['within_dist']:<20.4f} | {m_fine['within_dist']:<20.4f} | {m_fine['within_dist'] - m_pre['within_dist']:+.4f}")
    print(f"{'Between-Centroid Dist (D_b)':<35} | {m_pre['between_dist']:<20.4f} | {m_fine['between_dist']:<20.4f} | {m_fine['between_dist'] - m_pre['between_dist']:+.4f}")
    print(f"{'Separation Ratio (D_b / D_w)':<35} | {m_pre['sep_ratio']:<20.4f} | {m_fine['sep_ratio']:<20.4f} | {m_fine['sep_ratio'] - m_pre['sep_ratio']:+.4f}")
    print("=" * 80 + "\n")
    
    # -------------------------------------------------------------------------
    # Joint UMAP Manifold Fitting
    # -------------------------------------------------------------------------
    print("Fitting Single Joint UMAP on [Pretrained; Fine-Tuned] (36,792 x 200)...")
    X_joint = np.vstack([X_pre, X_fine])
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='euclidean')
    X_umap_joint = reducer.fit_transform(X_joint)
    
    N = len(y)
    X_umap_pre = X_umap_joint[:N]
    X_umap_fine = X_umap_joint[N:]
    
    # Common Axis Limits
    x_min, x_max = X_umap_joint[:, 0].min(), X_umap_joint[:, 0].max()
    y_min, y_max = X_umap_joint[:, 1].min(), X_umap_joint[:, 1].max()
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)
    xlims = [x_min - x_pad, x_max + x_pad]
    ylims = [y_min - y_pad, y_max + y_pad + 0.12 * (y_max - y_min)] # Top room for panel title/legend
    
    # -------------------------------------------------------------------------
    # Render 3-Panel Publication Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=350)
    
    idx_neutral = (y == 0)
    idx_boredom = (y == 1)
    
    COLOR_NEUTRAL = "#1e88e5" # Sapphire Blue
    COLOR_BOREDOM = "#2e7d32" # Forest Green
    COLOR_FP = "#d32f2f"      # Crimson Red
    COLOR_FN = "#f57c00"      # Amber Orange
    
    # ---------------------------------------------------------
    # Panel (a): Pretrained LaBraM (Ground-Truth)
    # ---------------------------------------------------------
    ax = axes[0]
    ax.scatter(X_umap_pre[idx_neutral, 0], X_umap_pre[idx_neutral, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="Non-Boredom", zorder=2)
    ax.scatter(X_umap_pre[idx_boredom, 0], X_umap_pre[idx_boredom, 1],
               color=COLOR_BOREDOM, marker="^", alpha=0.45, s=20, edgecolors="none",
               label="Boredom", zorder=2)
    
    ax.set_title("(a) Pretrained LaBraM", fontsize=17, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight="bold")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    
    # ---------------------------------------------------------
    # Panel (b): Fine-Tuned LaBraM (Ground-Truth)
    # ---------------------------------------------------------
    ax = axes[1]
    ax.scatter(X_umap_fine[idx_neutral, 0], X_umap_fine[idx_neutral, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="Non-Boredom", zorder=2)
    ax.scatter(X_umap_fine[idx_boredom, 0], X_umap_fine[idx_boredom, 1],
               color=COLOR_BOREDOM, marker="^", alpha=0.45, s=20, edgecolors="none",
               label="Boredom", zorder=2)
    
    ax.set_title("(b) Fine-Tuned LaBraM", fontsize=17, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight="bold")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    
    # ---------------------------------------------------------
    # Panel (c): Fine-Tuned LaBraM (Prediction Outcome: TN, TP, FP, FN)
    # ---------------------------------------------------------
    ax = axes[2]
    idx_tn = (y == 0) & (y_pred == 0)
    idx_tp = (y == 1) & (y_pred == 1)
    idx_fp = (y == 0) & (y_pred == 1)
    idx_fn = (y == 1) & (y_pred == 0)
    
    # Background correct points
    ax.scatter(X_umap_fine[idx_tn, 0], X_umap_fine[idx_tn, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="TN (Non-Boredom)", zorder=2)
    ax.scatter(X_umap_fine[idx_tp, 0], X_umap_fine[idx_tp, 1],
               color=COLOR_BOREDOM, marker="^", alpha=0.45, s=20, edgecolors="none",
               label="TP (Boredom)", zorder=2)
               
    # Foreground error points (Plotted on top with distinct high-contrast markers)
    if np.any(idx_fn):
        ax.scatter(X_umap_fine[idx_fn, 0], X_umap_fine[idx_fn, 1],
                   color=COLOR_FN, marker="s", alpha=0.95, s=50, edgecolors="black", linewidths=0.7,
                   label="FN (False Negative)", zorder=10)
    if np.any(idx_fp):
        ax.scatter(X_umap_fine[idx_fp, 0], X_umap_fine[idx_fp, 1],
                   color=COLOR_FP, marker="x", alpha=1.0, s=75, linewidths=2.0,
                   label="FP (False Positive)", zorder=10)
                   
    ax.set_title("(c) Prediction Outcome", fontsize=17, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight="bold")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    
    plt.tight_layout()
    
    out_png = RUNS_DIR / "umap_3panel_analysis.png"
    out_pdf = RUNS_DIR / "umap_3panel_analysis.pdf"
    plt.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=350, bbox_inches="tight")
    plt.close()
    
    print(f"Saved Publication 3-Panel PNG: {out_png}")
    print(f"Saved Publication Vector PDF:  {out_pdf}")

if __name__ == "__main__":
    main()
