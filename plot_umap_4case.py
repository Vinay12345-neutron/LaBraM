#!/usr/bin/env python3
"""
Generate high-impact publication-quality 4-Case UMAP Manifold Visualization (TP, TN, FP, FN).
Following Dr. Yuvaraj's exact review guidelines:
1. Distinct marker shapes:
   - TN: Circles ('o')
   - TP: Triangles ('^')
   - FP: Bold Crosses ('x')
   - FN: Squares ('s')
2. Direct on-plot class/cluster annotations ("Non-Boredom", "Boredom") near cluster centroids.
3. Overlap / Decision Boundary region prominently displayed with prominent FP/FN overlay (zorder=10).
4. Clean legend without 'n=...' counts.
5. Publication specifications: 350 DPI, Arial font (16-18pt), titleless format.
"""
import os
import json
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
import umap

from modeling_finetune import labram_base_patch200_200

# Publication styling
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

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

def load_model_for_fold(fold_id):
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

def extract_fold_data(fold_id):
    split_file = Path(f"boredom_split_fold{fold_id}.json")
    if not split_file.exists():
        return [], [], []
        
    with open(split_file) as f:
        split = json.load(f)
    test_data = split['test']
    
    model = load_model_for_fold(fold_id)
    
    activations = {}
    def hook_norm(m, inp, out):
        activations['norm'] = out.detach().cpu()
    model.norm.register_forward_hook(hook_norm)
    
    embeddings = []
    trues = []
    preds = []
    
    for item in test_data:
        fpath = item['file']
        label = int(item['label'])
        
        try:
            with h5py.File(fpath, 'r') as f:
                keys = list(f.keys())
                if not keys:
                    continue
                obj = f[keys[0]]
                dset = None
                ch_names = []
                
                if isinstance(obj, h5py.Group):
                    if 'eeg' in obj:
                        dset = obj['eeg'][:]
                        if 'chOrder' in obj['eeg'].attrs:
                            ch_names = obj['eeg'].attrs['chOrder']
                    elif 'data' in obj:
                        dset = obj['data'][:]
                        if 'chOrder' in obj['data'].attrs:
                            ch_names = obj['data'].attrs['chOrder']
                    else:
                        subkeys = list(obj.keys())
                        if subkeys:
                            dset = obj[subkeys[0]][:]
                            if 'chOrder' in obj[subkeys[0]].attrs:
                                ch_names = obj[subkeys[0]].attrs['chOrder']
                else:
                    dset = obj[:]
                    if 'chOrder' in obj.attrs:
                        ch_names = obj.attrs['chOrder']
                        
                if dset is None:
                    continue
                    
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
                    x_input = torch.cat([p1, p2, p3], dim=2)
                    x_input = rearrange(x_input, 'b n (a t) -> b n a t', t=200)
                    
                    output = model(x_input, input_chans=input_chans)
                    prob = torch.sigmoid(output[0, 0]).item()
                    pred = 1 if prob >= 0.5 else 0
                    
                    emb = activations['norm'][:, 0, :].numpy()
                    embeddings.append(emb)
                    trues.append(label)
                    preds.append(pred)
        except Exception as e:
            continue
            
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(trues), np.array(preds)

def main():
    print("Extracting test embeddings across all 5 folds...")
    all_embs, all_trues, all_preds = [], [], []
    
    for fold_id in range(5):
        embs, trues, preds = extract_fold_data(fold_id)
        if len(embs) > 0:
            all_embs.append(embs)
            all_trues.append(trues)
            all_preds.append(preds)
            
    X = np.concatenate(all_embs, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    print(f"Total test windows for UMAP: {len(trues)}")
    
    print("Fitting UMAP manifold...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # 4 Classification Cases
    idx_tn = (trues == 0) & (preds == 0)
    idx_tp = (trues == 1) & (preds == 1)
    idx_fp = (trues == 0) & (preds == 1)
    idx_fn = (trues == 1) & (preds == 0)
    
    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=350)
    
    # 1. TN: Circles ('o') in Sapphire Blue
    if np.any(idx_tn):
        ax.scatter(X_umap[idx_tn, 0], X_umap[idx_tn, 1],
                   color="#1e88e5", marker="o", alpha=0.45, s=26, edgecolors="none",
                   label="TN (Non-Boredom)", zorder=2)
                   
    # 2. TP: Triangles ('^') in Forest Green
    if np.any(idx_tp):
        ax.scatter(X_umap[idx_tp, 0], X_umap[idx_tp, 1],
                   color="#2e7d32", marker="^", alpha=0.45, s=26, edgecolors="none",
                   label="TP (Boredom)", zorder=2)
                   
    # 3. FN: Squares ('s') in Amber/Orange with black borders (Plotted on top in boundary zone)
    if np.any(idx_fn):
        ax.scatter(X_umap[idx_fn, 0], X_umap[idx_fn, 1],
                   color="#f57c00", marker="s", alpha=0.95, s=55, edgecolors="black", linewidths=0.8,
                   label="FN (False Negative)", zorder=10)
                   
    # 4. FP: Bold Crosses ('x') in Crimson Red (Plotted on top in boundary zone)
    if np.any(idx_fp):
        ax.scatter(X_umap[idx_fp, 0], X_umap[idx_fp, 1],
                   color="#d32f2f", marker="x", alpha=1.0, s=85, linewidths=2.2,
                   label="FP (False Positive)", zorder=10)
                   
    # Direct On-Plot Cluster Centroid Labels
    if np.any(idx_tn):
        center_tn_x, center_tn_y = np.median(X_umap[idx_tn, 0]), np.median(X_umap[idx_tn, 1])
        ax.text(center_tn_x, center_tn_y + 1.2, "Non-Boredom",
                fontsize=16, fontweight="bold", color="#0d47a1", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e3f2fd", edgecolor="#1e88e5", lw=1.5, alpha=0.95),
                zorder=12)

    if np.any(idx_tp):
        center_tp_x, center_tp_y = np.median(X_umap[idx_tp, 0]), np.median(X_umap[idx_tp, 1])
        ax.text(center_tp_x, center_tp_y + 1.2, "Boredom",
                fontsize=16, fontweight="bold", color="#1b5e20", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1.5, alpha=0.95),
                zorder=12)
    x_min, x_max = X_umap[:, 0].min(), X_umap[:, 0].max()
    y_min, y_max = X_umap[:, 1].min(), X_umap[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Add top breathing room so legend never overlaps with any data points or labels
    ax.set_xlim([x_min - 0.06 * x_range, x_max + 0.06 * x_range])
    ax.set_ylim([y_min - 0.06 * y_range, y_max + 0.20 * y_range])

    ax.set_xlabel("UMAP Dimension 1", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP Dimension 2", fontsize=18, fontweight="bold", labelpad=10)
    
    # Modern publication aesthetics: remove top & right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Clean 2-column legend in dedicated top margin space
    ax.legend(loc="upper right", ncol=2, fontsize=12, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    
    plt.tight_layout()
    
    out_path = RUNS_DIR / "umap_4case_plot.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved updated UMAP plot to: {out_path}")

if __name__ == "__main__":
    main()
