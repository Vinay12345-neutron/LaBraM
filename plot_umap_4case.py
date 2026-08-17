#!/usr/bin/env python3
"""
Generate publication-quality 4-Case UMAP Manifold Visualization (TP, TN, FP, FN).
- Resolution: 350 DPI
- Font: Arial (16-18 pt)
- Titleless format for IEEE/Elsevier submission
- Distinct 4-case markers and color palette with explicit window counts in legend
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

# Import LaBraM model architecture definition
from modeling_finetune import labram_base_patch200_200

# Set publication style parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RUNS_DIR = Path("runs/boredom_cv")

# Standard 10-20 channel list for channel order mapping
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
    input_chans = [0] # CLS token index
    for name in ch_names:
        clean_name = name.upper().strip()
        if clean_name in standard_1020:
            idx = standard_1020.index(clean_name) + 1
            input_chans.append(idx)
        else:
            input_chans.append(1) # fallback
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
                input_chans = torch.tensor(input_chans_list).long().to(DEVICE)
                
                if dset.shape[0] > dset.shape[1]:
                    dset = dset.T
                n_samples = dset.shape[1]
                
                for start in range(0, n_samples - 512 + 1, 512):
                    window = (dset[:, start:start+512] / 100.0)
                    tensor = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
                    
                    p1 = tensor[:, :, 0:200]
                    p2 = tensor[:, :, 156:356]
                    p3 = tensor[:, :, 312:512]
                    x_input = torch.cat([p1, p2, p3], dim=2)
                    x_input = rearrange(x_input, 'b n (a t) -> b n a t', t=200)
                    
                    output = model(x_input, input_chans=input_chans)
                    prob = torch.sigmoid(output[0, 0]).item()
                    pred = 1 if prob >= 0.5 else 0
                    
                    emb = activations['norm'][:, 0, :].numpy() # [CLS] token embedding (1, 200)
                    embeddings.append(emb)
                    trues.append(label)
                    preds.append(pred)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
            
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(trues), np.array(preds)

def main():
    print("Extracting test embeddings and predictions across folds...")
    all_embs = []
    all_trues = []
    all_preds = []
    
    for fold_id in range(5):
        embs, trues, preds = extract_fold_data(fold_id)
        if len(embs) > 0:
            print(f"Fold {fold_id + 1}: extracted {len(trues)} test windows.")
            all_embs.append(embs)
            all_trues.append(trues)
            all_preds.append(preds)
            
    if not all_embs:
        print("No embeddings collected. Exiting.")
        return
        
    X = np.concatenate(all_embs, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    print(f"Total test windows for UMAP: {len(trues)}")
    
    print("Fitting UMAP manifold...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Identify 4 cases
    idx_tn = (trues == 0) & (preds == 0)
    idx_tp = (trues == 1) & (preds == 1)
    idx_fp = (trues == 0) & (preds == 1)
    idx_fn = (trues == 1) & (preds == 0)
    
    n_tn = int(np.sum(idx_tn))
    n_tp = int(np.sum(idx_tp))
    n_fp = int(np.sum(idx_fp))
    n_fn = int(np.sum(idx_fn))
    
    print(f"4-Case Distribution: TP={n_tp}, TN={n_tn}, FP={n_fp}, FN={n_fn}")
    
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=350)
    
    # Plot TN (True Negative - Neutral)
    if np.any(idx_tn):
        ax.scatter(X_umap[idx_tn, 0], X_umap[idx_tn, 1],
                   color="#1e88e5", alpha=0.55, s=28, edgecolors="none",
                   label=f"TN (Neutral) [n={n_tn}]")
                   
    # Plot TP (True Positive - Boredom)
    if np.any(idx_tp):
        ax.scatter(X_umap[idx_tp, 0], X_umap[idx_tp, 1],
                   color="#2e7d32", alpha=0.55, s=28, edgecolors="none",
                   label=f"TP (Boredom) [n={n_tp}]")
                   
    # Plot FP (False Positive - Neutral misclassified as Boredom)
    if np.any(idx_fp):
        ax.scatter(X_umap[idx_fp, 0], X_umap[idx_fp, 1],
                   color="#d32f2f", alpha=0.95, s=80, marker="D", edgecolors="black", linewidths=0.8,
                   label=f"FP [n={n_fp}]")
                   
    # Plot FN (False Negative - Boredom misclassified as Neutral)
    if np.any(idx_fn):
        ax.scatter(X_umap[idx_fn, 0], X_umap[idx_fn, 1],
                   color="#f57c00", alpha=0.95, s=80, marker="^", edgecolors="black", linewidths=0.8,
                   label=f"FN [n={n_fn}]")
                   
    ax.set_xlabel("UMAP Dimension 1", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("UMAP Dimension 2", fontsize=18, fontweight="bold", labelpad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    out_path1 = RUNS_DIR / "umap_4case_plot.png"
    out_path2 = RUNS_DIR / "umap_plot.png"
    plt.savefig(out_path1, dpi=350, bbox_inches="tight")
    plt.savefig(out_path2, dpi=350, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path1}")
    print(f"Saved: {out_path2}")

if __name__ == "__main__":
    main()
