#!/usr/bin/env python3
"""
Interpretability Topographies — 4-Case Analysis (TP / FP / TN / FN).

Loads a trained checkpoint, runs inference on a test split, categorises
every EEG window into one of four outcome groups, then generates:

  For each outcome (TP, FP, TN, FN):
    • Attention Map   — CLS→patch attention from the last Transformer layer
    • Saliency Map    — Input gradient magnitude (|∂output/∂input|)

  Plus:
    • 2×2 grid comparing all four outcomes for Attention
    • 2×2 grid comparing all four outcomes for Saliency
    • Individual maps at full resolution

Outputs saved to: runs/publication_figures/topography_4case/

Usage:
    python plot_topography_4case.py
    python plot_topography_4case.py --fold 0
    python plot_topography_4case.py --fold 0 --threshold 0.5
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from einops import rearrange

from modeling_finetune import labram_base_patch200_200
from utils import standard_1020, get_input_chans

# ─── Configuration ────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SZ = 512
PATCH_SZ  = 200
OUT_DIR   = Path("runs/publication_figures/topography_4case")

# 4-case labels and colour scheme
CASES = {
    "TP": {"label": "True Positive\n(Boredom → Boredom)", "color": "#2ECC71"},
    "FP": {"label": "False Positive\n(Neutral → Boredom)", "color": "#E74C3C"},
    "TN": {"label": "True Negative\n(Neutral → Neutral)",  "color": "#3498DB"},
    "FN": {"label": "False Negative\n(Boredom → Neutral)", "color": "#F39C12"},
}


# ─── MNE helpers ──────────────────────────────────────────────────────────────

def fix_ch(name: str) -> str:
    """Convert uppercase 10-20 names to MNE-compatible mixed-case."""
    replacements = {
        "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
        "FZ": "Fz",   "CZ": "Cz",   "PZ": "Pz",
        "OZ": "Oz",   "AFZ": "AFz", "FCZ": "FCz",
        "CPZ": "CPz", "POZ": "POz",
    }
    up = name.upper()
    if up in replacements:
        return replacements[up]
    if up.endswith("Z") and len(up) > 1:
        return up[:-1] + "z"
    return up


def build_montage(ch_names_raw: list) -> tuple[list, mne.Info]:
    """Return (mne_ch_names, mne_info) for the given channel list."""
    mne_names = [fix_ch(n) for n in ch_names_raw]
    montage   = mne.channels.make_standard_montage("standard_1020")
    valid     = set(montage.ch_names)
    keep_idx  = [i for i, n in enumerate(mne_names) if n in valid]
    mne_names = [mne_names[i] for i in keep_idx]
    info      = mne.create_info(mne_names, sfreq=200, ch_types="eeg")
    info.set_montage(montage)
    return mne_names, info, keep_idx


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str):
    model = labram_base_patch200_200(
        pretrained=False, num_classes=1, init_values=0.1
    )
    setattr(model, "patch_size", PATCH_SZ)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    msg   = model.load_state_dict(state, strict=False)
    print(f"  Checkpoint: {ckpt_path}  |  {msg}")
    model.to(DEVICE).eval()
    return model


# ─── Feature extraction ───────────────────────────────────────────────────────

class AttentionHook:
    """Grabs scaled attention weights from the last Transformer block."""
    def __init__(self):
        self.attn = None

    def __call__(self, module, inp, out):
        # inp[0] is the pre-softmax attention map (B, H, N, N)
        self.attn = inp[0].detach().cpu()


def extract_window_features(
    model, tensor_raw: torch.Tensor, input_chans: torch.Tensor,
    hook: AttentionHook, label: int, threshold: float
):
    """
    Run one window through the model.
    Returns dict with 'case', 'attn', 'saliency', or None if un-classifiable.
    """
    # tensor_raw: (1, 61, 512)
    p1 = tensor_raw[:, :, 0:200]
    p2 = tensor_raw[:, :, 156:356]
    p3 = tensor_raw[:, :, 312:512]
    x  = torch.cat([p1, p2, p3], dim=2)                          # 1,61,600
    x  = rearrange(x, "b n (a t) -> b n a t", t=PATCH_SZ)        # 1,61,3,200
    x  = x.to(DEVICE)

    # ── Saliency (gradient) ────────────────────────────────────────────────
    x_sal = x.detach().clone().requires_grad_(True)
    output = model(x_sal, input_chans=input_chans)
    score  = output[0, 0]
    # For boredom (label=1) maximise score; for neutral (label=0) minimise it
    target_score = score if label == 1 else -score
    model.zero_grad()
    target_score.backward()
    # Gradient magnitude averaged over patches and time: (61,)
    saliency = x_sal.grad.abs().squeeze(0).mean(dim=(1, 2)).detach().cpu().numpy()

    # ── Attention ─────────────────────────────────────────────────────────
    with torch.no_grad():
        _ = model(x, input_chans=input_chans)
    raw_attn = hook.attn  # (1, H, N, N)  N = 1+61*3 = 184

    # CLS token (row 0) → all patch tokens (cols 1:)
    # Average over heads, take CLS→patch attention
    attn_cls = raw_attn[0].mean(dim=0)[0, 1:].numpy()  # (183,)
    # Reshape (183,) to (61, 3) — per channel, per window
    n_patches = attn_cls.shape[0]
    n_windows = 3
    n_chans   = n_patches // n_windows
    attn_ch   = attn_cls.reshape(n_chans, n_windows).mean(axis=1)  # (61,)

    # Prediction
    prob = torch.sigmoid(output[0, 0]).item()
    pred = 1 if prob >= threshold else 0

    if label == 1 and pred == 1:
        case = "TP"
    elif label == 0 and pred == 1:
        case = "FP"
    elif label == 0 and pred == 0:
        case = "TN"
    else:
        case = "FN"

    return {"case": case, "attn": attn_ch, "saliency": saliency}


# ─── Topography plotting ──────────────────────────────────────────────────────

def plot_single_topo(data: np.ndarray, mne_info: mne.Info,
                     mne_names: list, keep_idx: list,
                     title: str, out_path: Path, cmap: str = "Reds"):
    """Plot a single topography map and save."""
    data_kept = data[keep_idx]
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        im, _ = mne.viz.plot_topomap(
            data_kept, mne_info, axes=ax, show=False,
            cmap=cmap, contours=5,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Importance (a.u.)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
        plt.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out_path}")
    except Exception as e:
        print(f"    Error plotting {title}: {e}")


def plot_4case_grid(case_data: dict, mne_info: mne.Info,
                    mne_names: list, keep_idx: list,
                    kind: str, cmap: str, out_path: Path):
    """
    2×2 grid of topography maps for all four cases.
    kind: 'Attention' or 'Saliency'
    """
    case_keys = ["TP", "TN", "FP", "FN"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()

    max_val = max(
        (case_data[k][kind].max() for k in case_keys if case_data.get(k) is not None),
        default=1.0,
    )

    for ax, key in zip(axes, case_keys):
        meta = CASES[key]
        if case_data.get(key) is None:
            ax.text(0.5, 0.5, f"No {key} samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor=meta["color"], lw=3,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.set_title(meta["label"], fontsize=10, fontweight="bold",
                         color=meta["color"])
            continue
        data_kept = case_data[key][kind][keep_idx]
        try:
            im, _ = mne.viz.plot_topomap(
                data_kept, mne_info, axes=ax, show=False,
                cmap=cmap, contours=5, vlim=(0, max_val),
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        n = case_data[key]["n"]
        ax.set_title(f"{meta['label']}\n(n={n} windows)",
                     fontsize=10, fontweight="bold", color=meta["color"])
        # Coloured border
        for spine in ax.spines.values():
            spine.set_edgecolor(meta["color"])
            spine.set_linewidth(2.5)

    fig.suptitle(
        f"4-Case {kind} Analysis — LaBraM Boredom Detection\n"
        "(TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold",      type=int,   default=0,
                        help="CV fold to use (default 0)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default 0.5)")
    parser.add_argument("--max_files", type=int,   default=None,
                        help="Limit number of test files processed (for speed)")
    parser.add_argument("--out_dir",   type=str,   default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold       = args.fold
    ckpt_path  = f"runs/boredom_cv/fold{fold}/checkpoint-best.pth"
    split_file = f"boredom_split_fold{fold}.json"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not Path(split_file).exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model (fold {fold})...")
    model = load_model(ckpt_path)
    for p in model.parameters():
        p.requires_grad_(False)

    # Register attention hook on the last block's attention drop
    attn_hook = AttentionHook()
    model.blocks[-1].attn.attn_drop.register_forward_hook(attn_hook)

    # ── Load test split ────────────────────────────────────────────────────
    with open(split_file) as f:
        split = json.load(f)
    test_items = split["test"]
    if args.max_files:
        test_items = test_items[: args.max_files]
    print(f"Test files: {len(test_items)}")

    # ── Process test files ─────────────────────────────────────────────────
    accumulators = {
        case: {"attn_sum": None, "sal_sum": None, "n": 0}
        for case in CASES
    }
    input_chans_tensor = None
    used_ch_names      = None

    print(f"\nExtracting features (threshold={args.threshold})...")
    for item in test_items:
        fpath = item["file"]
        label = item["label"]

        try:
            with h5py.File(fpath, "r") as f:
                key = list(f.keys())[0]
                obj = f[key]
                if isinstance(obj, h5py.Group):
                    dset = obj["eeg"] if "eeg" in obj else obj[list(obj.keys())[0]]
                else:
                    dset = obj
                arr      = dset[:].astype(np.float32)
                ch_raw   = [c.decode() if isinstance(c, bytes) else c
                            for c in dset.attrs.get("chOrder", [])]
        except Exception as e:
            print(f"  Skip {fpath}: {e}")
            continue

        if arr.shape[0] > arr.shape[1]:
            arr = arr.T

        # Build input chans once
        if input_chans_tensor is None and ch_raw:
            ch_indices       = get_input_chans(ch_raw)
            input_chans_tensor = torch.tensor(ch_indices).long().to(DEVICE)
            used_ch_names    = [standard_1020[i - 1] for i in ch_indices[1:]]

        n_samples = arr.shape[1]
        for start in range(0, n_samples - WINDOW_SZ + 1, WINDOW_SZ):
            window = arr[:, start: start + WINDOW_SZ] / 100.0
            t      = torch.from_numpy(window).float().unsqueeze(0)  # 1,61,512

            result = extract_window_features(
                model, t, input_chans_tensor,
                attn_hook, label, args.threshold,
            )
            if result is None:
                continue

            case = result["case"]
            acc  = accumulators[case]
            if acc["attn_sum"] is None:
                acc["attn_sum"] = np.zeros_like(result["attn"])
                acc["sal_sum"]  = np.zeros_like(result["saliency"])
            acc["attn_sum"] += result["attn"]
            acc["sal_sum"]  += result["saliency"]
            acc["n"]         += 1

    # Summary
    print("\n  Windows per case:")
    for case, acc in accumulators.items():
        print(f"    {case}: {acc['n']}")

    if input_chans_tensor is None or used_ch_names is None:
        raise RuntimeError("No channels extracted — check test file contents.")

    # ── Build montage ──────────────────────────────────────────────────────
    mne_names, mne_info, keep_idx = build_montage(used_ch_names)
    print(f"\n  Montage: {len(mne_names)} channels matched to MNE standard_1020")

    # ── Compute averages per case ──────────────────────────────────────────
    case_avgs = {}
    for case, acc in accumulators.items():
        if acc["n"] > 0:
            case_avgs[case] = {
                "Attention": acc["attn_sum"] / acc["n"],
                "Saliency":  acc["sal_sum"]  / acc["n"],
                "n":         acc["n"],
            }
        else:
            case_avgs[case] = None

    # ── 2×2 Grid plots ────────────────────────────────────────────────────
    print("\n  Generating 2×2 grid plots...")
    plot_4case_grid(case_avgs, mne_info, mne_names, keep_idx,
                    kind="Attention", cmap="Reds",
                    out_path=out_dir / "4case_attention_grid.png")
    plot_4case_grid(case_avgs, mne_info, mne_names, keep_idx,
                    kind="Saliency", cmap="inferno",
                    out_path=out_dir / "4case_saliency_grid.png")

    # ── Individual full-res maps ───────────────────────────────────────────
    print("\n  Generating individual topography maps...")
    for case, avgs in case_avgs.items():
        if avgs is None:
            continue
        plot_single_topo(
            avgs["Attention"], mne_info, mne_names, keep_idx,
            title=f"Attention — {CASES[case]['label']} (n={avgs['n']})",
            out_path=out_dir / f"attn_{case}.png",
            cmap="Reds",
        )
        plot_single_topo(
            avgs["Saliency"], mne_info, mne_names, keep_idx,
            title=f"Saliency — {CASES[case]['label']} (n={avgs['n']})",
            out_path=out_dir / f"saliency_{case}.png",
            cmap="inferno",
        )

    # ── Combined 4-panel difference figure ────────────────────────────────
    # Boredom reference = mean(TP, FN) — all boredom windows regardless of outcome
    # Neutral reference = mean(TN, FP)
    bor_cases = [case_avgs[c] for c in ["TP", "FN"] if case_avgs.get(c)]
    neu_cases = [case_avgs[c] for c in ["TN", "FP"] if case_avgs.get(c)]
    if bor_cases and neu_cases:
        bor_attn = np.mean([c["Attention"] for c in bor_cases], axis=0)
        neu_attn = np.mean([c["Attention"] for c in neu_cases], axis=0)
        diff_attn = bor_attn - neu_attn
        plot_single_topo(
            diff_attn, mne_info, mne_names, keep_idx,
            title="Attention Difference — Boredom minus Neutral\n(4-case aggregated)",
            out_path=out_dir / "attn_diff_4case.png",
            cmap="RdBu_r",
        )
        bor_sal  = np.mean([c["Saliency"] for c in bor_cases], axis=0)
        neu_sal  = np.mean([c["Saliency"] for c in neu_cases], axis=0)
        diff_sal = bor_sal - neu_sal
        plot_single_topo(
            diff_sal, mne_info, mne_names, keep_idx,
            title="Saliency Difference — Boredom minus Neutral\n(4-case aggregated)",
            out_path=out_dir / "saliency_diff_4case.png",
            cmap="RdBu_r",
        )

    print(f"\nDone. All 4-case topography maps saved to {out_dir}/")


if __name__ == "__main__":
    main()
