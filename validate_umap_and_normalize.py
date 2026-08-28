#!/usr/bin/env python3
"""
Comprehensive Representation Validation & L2-Normalized UMAP Analysis
Addressing all 10 diagnostic points:
1. Verification of layer extraction, tensor shapes, [CLS] token index.
2. Embedding scale statistics (Mean, Std, Min, Max, L2 norms, dimension variance).
3. L2-normalized 200-D representation metrics (Silhouette, 15-NN purity, within/between distance, separation ratio).
4. Joint UMAP on L2-normalized space (umap_3panel_normalized.png / .pdf).
5. Pretrained representation diagnostics (PCA variance, pairwise distance, 2D PCA projection).
6. Preserving raw 3-panel figure as umap_3panel_raw.png / .pdf.
"""

import json
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from einops import rearrange
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap

# Model definition
from modeling_finetune import labram_base_patch200_200

# IEEE Style parameters
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
    ckpt = torch.load(PRETRAINED_CKPT, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model.load_state_dict(state_dict, strict=False)
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
    print("=" * 80)
    print("EXTRACTING PAIRED TEST WINDOW EMBEDDINGS")
    print("=" * 80)
    
    model_pre = load_pretrained_model()
    act_pre = {}
    def hook_pre(m, inp, out):
        act_pre['norm'] = out.detach().cpu()
    model_pre.norm.register_forward_hook(hook_pre)
    
    all_pre_embs, all_fine_embs, all_trues, all_preds, all_subjects = [], [], [], [], []
    
    shape_logged = False
    
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
                        
                        # Forward Pretrained
                        _ = model_pre(x_input, input_chans=input_chans)
                        if not shape_logged:
                            print(f"[Verification] model.norm tensor output shape: {act_pre['norm'].shape}")
                            print(f"[Verification] Extracted token 0 (CLS) shape  : {act_pre['norm'][:, 0, :].shape}")
                            shape_logged = True
                            
                        emb_p = act_pre['norm'][:, 0, :].numpy()
                        
                        # Forward Fine-Tuned
                        out_f = model_fine(x_input, input_chans=input_chans)
                        emb_f = act_fine['norm'][:, 0, :].numpy()
                        
                        prob = torch.sigmoid(out_f[0, 0]).item()
                        pred = 1 if prob >= 0.5 else 0
                        
                        fold_pre.append(emb_p)
                        fold_fine.append(emb_f)
                        fold_y.append(label)
                        fold_pred.append(pred)
                        fold_subjs.append(subj_id)
            except Exception as e:
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
    
    return X_pre, X_fine, y, y_pred, all_subjects

def report_embedding_scale_stats(X_pre, X_fine):
    print("\n" + "=" * 80)
    print("EMBEDDING SCALE & DISTRIBUTION STATISTICS (RAW 200-D SPACE)")
    print("=" * 80)
    
    norm_pre = np.linalg.norm(X_pre, axis=1)
    norm_fine = np.linalg.norm(X_fine, axis=1)
    
    dim_var_pre = np.var(X_pre, axis=0)
    dim_var_fine = np.var(X_fine, axis=0)
    
    print(f"{'Statistic':<35} | {'Pretrained LaBraM':<20} | {'Fine-Tuned LaBraM':<20}")
    print("-" * 80)
    print(f"{'Global Mean':<35} | {np.mean(X_pre):<20.6f} | {np.mean(X_fine):<20.6f}")
    print(f"{'Global Std Dev':<35} | {np.std(X_pre):<20.6f} | {np.std(X_fine):<20.6f}")
    print(f"{'Global Min':<35} | {np.min(X_pre):<20.6f} | {np.min(X_fine):<20.6f}")
    print(f"{'Global Max':<35} | {np.max(X_pre):<20.6f} | {np.max(X_fine):<20.6f}")
    print(f"{'L2 Norm / Sample (Mean ± SD)':<35} | {np.mean(norm_pre):.4f} ± {np.std(norm_pre):.4f}      | {np.mean(norm_fine):.4f} ± {np.std(norm_fine):.4f}")
    print(f"{'L2 Norm Range (Min - Max)':<35} | [{np.min(norm_pre):.4f}, {np.max(norm_pre):.4f}]   | [{np.min(norm_fine):.4f}, {np.max(norm_fine):.4f}]")
    print(f"{'Mean Variance Across Dims':<35} | {np.mean(dim_var_pre):<20.6f} | {np.mean(dim_var_fine):<20.6f}")
    print("=" * 80)

def compute_metrics_table(X_pre, X_fine, y):
    """Compute representation metrics on both Raw and L2-Normalized 200-D embeddings."""
    
    # Per-sample L2-normalization
    X_pre_norm = X_pre / np.linalg.norm(X_pre, axis=1, keepdims=True)
    X_fine_norm = X_fine / np.linalg.norm(X_fine, axis=1, keepdims=True)
    
    def eval_space(X, y):
        # 1. Silhouette score (subsampled 10k for computation)
        sil = silhouette_score(X, y, metric='euclidean', sample_size=10000, random_state=42)
        
        # 2. 15-NN Purity
        knn = NearestNeighbors(n_neighbors=16, metric='euclidean', n_jobs=-1)
        knn.fit(X)
        neighbors = knn.kneighbors(X, return_distance=False)[:, 1:]
        purity = np.mean(np.mean(y[neighbors] == y[:, None], axis=1)) * 100.0
        
        # 3. Distances
        idx_0 = (y == 0)
        idx_1 = (y == 1)
        c0 = np.mean(X[idx_0], axis=0)
        c1 = np.mean(X[idx_1], axis=0)
        
        d_w0 = np.linalg.norm(X[idx_0] - c0, axis=1)
        d_w1 = np.linalg.norm(X[idx_1] - c1, axis=1)
        d_within = np.mean(np.concatenate([d_w0, d_w1]))
        d_between = np.linalg.norm(c0 - c1)
        sep_ratio = d_between / d_within if d_within > 0 else 0.0
        
        return sil, purity, d_within, d_between, sep_ratio
        
    s_pre_raw, p_pre_raw, dw_pre_raw, db_pre_raw, r_pre_raw = eval_space(X_pre, y)
    s_fine_raw, p_fine_raw, dw_fine_raw, db_fine_raw, r_fine_raw = eval_space(X_fine, y)
    
    s_pre_n, p_pre_n, dw_pre_n, db_pre_n, r_pre_n = eval_space(X_pre_norm, y)
    s_fine_n, p_fine_n, dw_fine_n, db_fine_n, r_fine_n = eval_space(X_fine_norm, y)
    
    print("\n" + "=" * 90)
    print("QUANTITATIVE COMPARISON: RAW vs. L2-NORMALIZED ORIGINAL 200-D REPRESENTATIONS")
    print("=" * 90)
    print(f"{'Evaluation Mode':<28} | {'Silhouette':<12} | {'15-NN Purity':<14} | {'D_within':<12} | {'D_between':<12} | {'Sep. Ratio'}")
    print("-" * 90)
    print(f"{'Pretrained (Raw 200-D)':<28} | {s_pre_raw:<12.4f} | {p_pre_raw:<13.2f}% | {dw_pre_raw:<12.4f} | {db_pre_raw:<12.4f} | {r_pre_raw:<10.4f}")
    print(f"{'Fine-Tuned (Raw 200-D)':<28} | {s_fine_raw:<12.4f} | {p_fine_raw:<13.2f}% | {dw_fine_raw:<12.4f} | {db_fine_raw:<12.4f} | {r_fine_raw:<10.4f}")
    print("-" * 90)
    print(f"{'Pretrained (L2-Norm 200-D)':<28} | {s_pre_n:<12.4f} | {p_pre_n:<13.2f}% | {dw_pre_n:<12.4f} | {db_pre_n:<12.4f} | {r_pre_n:<10.4f}")
    print(f"{'Fine-Tuned (L2-Norm 200-D)':<28} | {s_fine_n:<12.4f} | {p_fine_n:<13.2f}% | {dw_fine_n:<12.4f} | {db_fine_n:<12.4f} | {r_fine_n:<10.4f}")
    print("=" * 90)
    
    return X_pre_norm, X_fine_norm

def run_pretrained_pca_diagnostics(X_pre, y):
    """Run PCA diagnostics on Pretrained 200-D space to test for dimensional collapse."""
    print("\nRunning PCA diagnostics on Pretrained embeddings...")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_pre)
    
    var_exp = pca.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(var_exp)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=350)
    
    # 1. Scree Plot
    ax = axes[0]
    ax.plot(range(1, 21), cum_var[:20], 'o-', color="#1e88e5", lw=2)
    ax.bar(range(1, 21), var_exp[:20], alpha=0.5, color="#90caf9", label="Individual Variance (%)")
    ax.set_xlabel("Principal Component", fontsize=15, fontweight="bold")
    ax.set_ylabel("Variance Explained (%)", fontsize=15, fontweight="bold")
    ax.set_title("Pretrained LaBraM: PCA Scree Plot", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 2. 2D PCA Projection
    ax = axes[1]
    idx_0 = (y == 0)
    idx_1 = (y == 1)
    ax.scatter(X_pca[idx_0, 0], X_pca[idx_0, 1], color="#1e88e5", marker="o", alpha=0.4, s=18, label="Non-Boredom")
    ax.scatter(X_pca[idx_1, 0], X_pca[idx_1, 1], color="#2e7d32", marker="^", alpha=0.4, s=18, label="Boredom")
    ax.set_xlabel(f"PC 1 ({var_exp[0]:.1f}%)", fontsize=15, fontweight="bold")
    ax.set_ylabel(f"PC 2 ({var_exp[1]:.1f}%)", fontsize=15, fontweight="bold")
    ax.set_title("Pretrained LaBraM: 2D PCA Projection", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    out_pca = RUNS_DIR / "pretrained_pca_diagnostics.png"
    plt.savefig(out_pca, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved Pretrained PCA Diagnostics: {out_pca}")

def render_3panel_figure(X_pre_proj, X_fine_proj, y, y_pred, out_prefix):
    """Render standardized 3-panel publication figure."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=350)
    
    idx_neutral = (y == 0)
    idx_boredom = (y == 1)
    
    COLOR_NEUTRAL = "#1e88e5"
    COLOR_BOREDOM = "#2e7d32"
    COLOR_FP = "#d32f2f"
    COLOR_FN = "#f57c00"
    
    # Common Axis Limits
    all_x = np.concatenate([X_pre_proj[:, 0], X_fine_proj[:, 0]])
    all_y = np.concatenate([X_pre_proj[:, 1], X_fine_proj[:, 1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)
    xlims = [x_min - x_pad, x_max + x_pad]
    ylims = [y_min - y_pad, y_max + y_pad + 0.12 * (y_max - y_min)]
    
    # Panel (a): Pretrained
    ax = axes[0]
    ax.scatter(X_pre_proj[idx_neutral, 0], X_pre_proj[idx_neutral, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="Non-Boredom", zorder=2)
    ax.scatter(X_pre_proj[idx_boredom, 0], X_pre_proj[idx_boredom, 1],
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
    
    # Panel (b): Fine-Tuned
    ax = axes[1]
    ax.scatter(X_fine_proj[idx_neutral, 0], X_fine_proj[idx_neutral, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="Non-Boredom", zorder=2)
    ax.scatter(X_fine_proj[idx_boredom, 0], X_fine_proj[idx_boredom, 1],
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
    
    # Panel (c): Prediction Outcome
    ax = axes[2]
    idx_tn = (y == 0) & (y_pred == 0)
    idx_tp = (y == 1) & (y_pred == 1)
    idx_fp = (y == 0) & (y_pred == 1)
    idx_fn = (y == 1) & (y_pred == 0)
    
    ax.scatter(X_fine_proj[idx_tn, 0], X_fine_proj[idx_tn, 1],
               color=COLOR_NEUTRAL, marker="o", alpha=0.45, s=20, edgecolors="none",
               label="TN (Non-Boredom)", zorder=2)
    ax.scatter(X_fine_proj[idx_tp, 0], X_fine_proj[idx_tp, 1],
               color=COLOR_BOREDOM, marker="^", alpha=0.45, s=20, edgecolors="none",
               label="TP (Boredom)", zorder=2)
               
    if np.any(idx_fn):
        ax.scatter(X_fine_proj[idx_fn, 0], X_fine_proj[idx_fn, 1],
                   color=COLOR_FN, marker="s", alpha=0.95, s=50, edgecolors="black", linewidths=0.7,
                   label="FN (False Negative)", zorder=10)
    if np.any(idx_fp):
        ax.scatter(X_fine_proj[idx_fp, 0], X_fine_proj[idx_fp, 1],
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
    
    out_png = RUNS_DIR / f"{out_prefix}.png"
    out_pdf = RUNS_DIR / f"{out_prefix}.pdf"
    plt.savefig(out_png, dpi=350, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=350, bbox_inches="tight")
    plt.close()
    
    print(f"Saved {out_prefix} (PNG): {out_png}")
    print(f"Saved {out_prefix} (PDF): {out_pdf}")

def main():
    X_pre, X_fine, y, y_pred, subjects = extract_paired_test_data()
    
    # 1. Scale & Distribution statistics
    report_embedding_scale_stats(X_pre, X_fine)
    
    # 2. Metric comparison (Raw vs Normalized)
    X_pre_norm, X_fine_norm = compute_metrics_table(X_pre, X_fine, y)
    
    # 3. PCA diagnostics on Pretrained
    run_pretrained_pca_diagnostics(X_pre, y)
    
    # 4. Joint UMAP on RAW embeddings
    print("\nFitting Joint UMAP on RAW embeddings (36,792 x 200)...")
    X_joint_raw = np.vstack([X_pre, X_fine])
    reducer_raw = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='euclidean')
    umap_raw = reducer_raw.fit_transform(X_joint_raw)
    N = len(y)
    render_3panel_figure(umap_raw[:N], umap_raw[N:], y, y_pred, "umap_3panel_raw")
    
    # 5. Joint UMAP on L2-NORMALIZED embeddings
    print("\nFitting Joint UMAP on L2-NORMALIZED embeddings (36,792 x 200)...")
    X_joint_norm = np.vstack([X_pre_norm, X_fine_norm])
    reducer_norm = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='euclidean')
    umap_norm = reducer_norm.fit_transform(X_joint_norm)
    render_3panel_figure(umap_norm[:N], umap_norm[N:], y, y_pred, "umap_3panel_normalized")

if __name__ == "__main__":
    main()
