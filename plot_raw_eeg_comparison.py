#!/usr/bin/env python3
"""
Raw EEG Waveform Comparison: Boredom vs Neutral.

Generates publication-quality figures for a 2-second (512 sample @ 256 Hz) EEG window:
  1. Multi-channel waveform plot — frontal, central, parietal, occipital groups
  2. Power Spectral Density (PSD) comparison — both conditions on one plot
     with band annotations (Delta, Theta, Alpha, Beta, Gamma)
  3. Frontal-only close-up — Fp1, Fpz, Fp2, F3, Fz, F4 side-by-side

All outputs saved to:  runs/publication_figures/raw_eeg_comparison/

Usage:
    python plot_raw_eeg_comparison.py
    python plot_raw_eeg_comparison.py --boredom_file boredom_hdf5/S5_Boredom.h5
                                      --neutral_file neutral_hdf5/S5_Neutral.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

# ─── Channel layout (ch61.txt order, 0-indexed) ───────────────────────────────
# Frontal: FP1, FPZ, FP2, F7, F3, FZ, F4, F8, AF7, AF3, AF4, AF8, F5, F1, F2, F6
# Central: T7, C3, CZ, C4, T8, C5, C1, C2, C6
# Parietal: CP5,CP1,CP2,CP6, P7,P3,PZ,P4,P8, CP3,CP4, P5,P1,P2,P6
# Occipital: POZ,O1,O2, PO5,PO3,PO4,PO6, PO7,PO8,OZ

CH_NAMES = [
    "FP1","FPZ","FP2","F7","F3","FZ","F4","F8",          # 0–7
    "FC5","FC1","FC2","FC6",                               # 8–11
    "T7","C3","CZ","C4","T8",                             # 12–16
    "CP5","CP1","CP2","CP6",                               # 17–20
    "P7","P3","PZ","P4","P8",                             # 21–25
    "POZ","O1","O2",                                       # 26–28
    "AF7","AF3","AF4","AF8",                               # 29–32
    "F5","F1","F2","F6",                                   # 33–36
    "FC3","FCZ","FC4",                                     # 37–39
    "C5","C1","C2","C6",                                   # 40–43
    "CP3","CP4",                                           # 44–45
    "P5","P1","P2","P6",                                   # 46–49
    "PO5","PO3","PO4","PO6",                               # 50–53
    "FT7","FT8",                                           # 54–55
    "TP7","TP8",                                           # 56–57
    "PO7","PO8","OZ",                                      # 58–60
]

# Representative channels per lobe region
FRONTAL    = [0, 1, 2, 4, 5, 6, 31, 32]   # FP1 FPZ FP2 F3 FZ F4 AF3 AF8
CENTRAL    = [13, 14, 15, 16]              # C3 CZ C4 T8
PARIETAL   = [23, 22, 24]                  # PZ P3 P4
OCCIPITAL  = [27, 28, 60]                  # O1 O2 OZ

# EEG band definitions (Hz)
BANDS = {
    "δ Delta":  (0.5, 4),
    "θ Theta":  (4, 8),
    "α Alpha":  (8, 13),
    "β Beta":   (13, 30),
    "γ Gamma":  (30, 45),
}

SFREQ     = 256        # Hz
WINDOW_SZ = 512        # 2-second window
OUT_DIR   = Path("runs/publication_figures/raw_eeg_comparison")

# Academic colour palette
COL_BOR = "#E05A4E"    # warm red for Boredom
COL_NEU = "#4E90C8"    # steel blue for Neutral


def load_first_window(h5_path: str) -> tuple[np.ndarray, list]:
    """Return (eeg_array [61×512], ch_names) from a HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        obj = f[key]
        dset = obj["eeg"] if "eeg" in obj else obj[list(obj.keys())[0]]
        arr = dset[:]
        ch_names = [c.decode() if isinstance(c, bytes) else c
                    for c in dset.attrs.get("chOrder", [])]
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T                        # ensure (channels, time)
    return arr[:, :WINDOW_SZ].astype(np.float32), ch_names


def normalize_display(arr: np.ndarray) -> np.ndarray:
    """Per-channel normalise for display (zero-mean, scale by iqr)."""
    out = arr.copy()
    for i in range(arr.shape[0]):
        mu = np.mean(arr[i])
        sd = np.std(arr[i]) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out


def compute_psd(arr: np.ndarray, sfreq: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd_db) averaged across all 61 channels."""
    psds = []
    for ch in range(arr.shape[0]):
        f, p = welch(arr[ch], fs=sfreq, nperseg=min(256, arr.shape[1]))
        psds.append(p)
    psd_mean = np.mean(psds, axis=0)
    psd_db   = 10 * np.log10(psd_mean + 1e-12)
    return f, psd_db


def plot_waveforms(bor_arr, neu_arr, ch_indices, ch_labels, title, ax,
                   spacing=6.0, sfreq=SFREQ, t_start=0):
    """Stack multiple channels vertically with offset for clarity."""
    t = np.arange(bor_arr.shape[1]) / sfreq + t_start
    bor_norm = normalize_display(bor_arr)
    neu_norm = normalize_display(neu_arr)

    for k, (i, lbl) in enumerate(zip(ch_indices, ch_labels)):
        offset = -k * spacing
        ax.plot(t, neu_norm[i] + offset, color=COL_NEU, lw=0.9, alpha=0.85)
        ax.plot(t, bor_norm[i] + offset, color=COL_BOR, lw=0.9, alpha=0.85)
        ax.text(t[-1] + 0.02, offset, lbl, fontsize=7, va="center",
                color="#333333")

    ax.set_xlim(t[0], t[-1] + 0.06)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_psd_comparison(bor_arr, neu_arr, ax, sfreq=SFREQ,
                         frontal_only=False, ch_indices=None):
    """Side-by-side PSD curves with frequency band annotations."""
    if frontal_only and ch_indices is not None:
        b_psd_arr = [bor_arr[i] for i in ch_indices]
        n_psd_arr = [neu_arr[i] for i in ch_indices]
        f_b, _ = welch(b_psd_arr[0], fs=sfreq, nperseg=128)
        psd_b  = 10 * np.log10(np.mean([
            welch(ch, fs=sfreq, nperseg=128)[1] for ch in b_psd_arr
        ], axis=0) + 1e-12)
        psd_n  = 10 * np.log10(np.mean([
            welch(ch, fs=sfreq, nperseg=128)[1] for ch in n_psd_arr
        ], axis=0) + 1e-12)
        freqs  = f_b
    else:
        freqs, psd_b = compute_psd(bor_arr, sfreq)
        _,     psd_n = compute_psd(neu_arr, sfreq)

    mask = freqs <= 50
    ax.plot(freqs[mask], psd_b[mask], color=COL_BOR, lw=1.8,
            label="Boredom", zorder=3)
    ax.plot(freqs[mask], psd_n[mask], color=COL_NEU, lw=1.8,
            label="Neutral", zorder=3)

    # Band annotations
    colours = ["#dde8f5", "#d5ead8", "#fef3cd", "#fde6e6", "#ede0f5"]
    for (band, (lo, hi)), bc in zip(BANDS.items(), colours):
        ax.axvspan(lo, hi, alpha=0.25, color=bc, zorder=1)
        mid = (lo + hi) / 2
        ymin, ymax = ax.get_ylim()
        ax.text(mid, ymax * 0.97, band.split()[0], ha="center", va="top",
                fontsize=7, color="#555555")

    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Power (dB)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.set_title("Power Spectral Density", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boredom_file", type=str, default=None,
                        help="Specific boredom .h5 file (defaults to first found)")
    parser.add_argument("--neutral_file", type=str, default=None,
                        help="Specific neutral .h5 file (defaults to matching subject)")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── File selection ─────────────────────────────────────────────────────
    bor_files = sorted(Path("boredom_hdf5").glob("*.h5"))
    neu_files = sorted(Path("neutral_hdf5").glob("*.h5"))
    if not bor_files or not neu_files:
        raise FileNotFoundError("Missing boredom_hdf5/ or neutral_hdf5/ data")

    bor_path = args.boredom_file or str(bor_files[0])
    # Try to match subject ID for a fair comparison
    if args.neutral_file:
        neu_path = args.neutral_file
    else:
        stem = Path(bor_path).stem.replace("_Boredom", "")
        match = [f for f in neu_files if stem in f.stem]
        neu_path = str(match[0] if match else neu_files[0])

    print(f"Boredom file  : {bor_path}")
    print(f"Neutral file  : {neu_path}")

    bor_arr, ch_names = load_first_window(bor_path)
    neu_arr, _        = load_first_window(neu_path)

    # Use ch61.txt names if HDF5 names are generic / empty
    if not ch_names or all(c.startswith("Ch") for c in ch_names):
        ch_names = CH_NAMES

    print(f"Data shape    : {bor_arr.shape} (channels × samples)")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 — Multi-region waveform comparison
    # ─────────────────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.35,
                             left=0.06, right=0.94, top=0.92, bottom=0.06)

    region_cfg = [
        (FRONTAL,   "Frontal (Fp1/Fpz/Fp2, F3/Fz/F4, AF3/AF4)",   (0, 0)),
        (CENTRAL,   "Central (C3 / Cz / C4 / T8)",                 (1, 0)),
        (PARIETAL,  "Parietal (P3 / Pz / P4)",                     (2, 0)),
        (OCCIPITAL, "Occipital (O1 / O2 / Oz)",                    (3, 0)),
    ]
    for idxs, title, (row, col) in region_cfg:
        safe_idxs = [i for i in idxs if i < len(ch_names)]
        lbls       = [ch_names[i] for i in safe_idxs]
        ax = fig.add_subplot(gs[row, col])
        plot_waveforms(bor_arr, neu_arr, safe_idxs, lbls, title, ax)

    # PSD — all channels (right column, top 2 rows spanning)
    ax_psd = fig.add_subplot(gs[0:2, 1])
    plot_psd_comparison(bor_arr, neu_arr, ax_psd)

    # PSD — frontal only (right column, bottom 2 rows spanning)
    ax_front_psd = fig.add_subplot(gs[2:4, 1])
    plot_psd_comparison(bor_arr, neu_arr, ax_front_psd,
                        frontal_only=True, ch_indices=FRONTAL)
    ax_front_psd.set_title("Frontal PSD (Fp1–Fz channels)",
                            fontsize=10, fontweight="bold")

    # Legend patch
    from matplotlib.patches import Patch
    leg_handles = [Patch(color=COL_BOR, label="Boredom"),
                   Patch(color=COL_NEU, label="Neutral")]
    fig.legend(handles=leg_handles, loc="upper center", ncol=2,
               fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.97))

    fig.suptitle("Raw EEG Waveform Comparison — Boredom vs Neutral\n"
                 "(2-second window · 61 channels · 256 Hz)",
                 fontsize=13, fontweight="bold", y=1.00)

    out1 = out_dir / "raw_eeg_waveform_comparison.png"
    fig.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 — Frontal close-up (publication inset)
    # ─────────────────────────────────────────────────────────────────────────
    front6_idx = [0, 1, 2, 4, 5, 6]      # FP1, FPZ, FP2, F3, FZ, F4
    front6_lbl = [ch_names[i] for i in front6_idx if i < len(ch_names)]

    fig2, axes2 = plt.subplots(len(front6_idx), 2, figsize=(14, 10),
                                sharey=False)
    t = np.arange(WINDOW_SZ) / SFREQ

    for k, (ch_i, lbl) in enumerate(zip(front6_idx, front6_lbl)):
        if ch_i >= bor_arr.shape[0]:
            continue
        b = bor_arr[ch_i]
        n = neu_arr[ch_i]
        # Waveform
        axes2[k, 0].plot(t, n, color=COL_NEU, lw=0.9, alpha=0.9,
                          label="Neutral")
        axes2[k, 0].plot(t, b, color=COL_BOR, lw=0.9, alpha=0.9,
                          label="Boredom")
        axes2[k, 0].set_ylabel(f"{lbl}\n(μV)", fontsize=8)
        axes2[k, 0].spines[["top", "right"]].set_visible(False)
        if k == 0:
            axes2[k, 0].legend(fontsize=8, loc="upper right")
        # PSD
        fs_b, ps_b = welch(b, fs=SFREQ, nperseg=128)
        fs_n, ps_n = welch(n, fs=SFREQ, nperseg=128)
        mask = fs_b <= 50
        axes2[k, 1].semilogy(fs_b[mask], ps_b[mask], color=COL_BOR, lw=1.2)
        axes2[k, 1].semilogy(fs_n[mask], ps_n[mask], color=COL_NEU, lw=1.2)
        axes2[k, 1].set_ylabel("PSD (μV²/Hz)", fontsize=7)
        axes2[k, 1].spines[["top", "right"]].set_visible(False)

    axes2[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes2[-1, 1].set_xlabel("Frequency (Hz)", fontsize=9)
    fig2.suptitle("Frontal Channels — Boredom vs Neutral (Waveform & PSD)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()

    out2 = out_dir / "raw_eeg_frontal_closeup.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")

    print("\nDone. All raw EEG comparison figures saved.")


if __name__ == "__main__":
    main()
