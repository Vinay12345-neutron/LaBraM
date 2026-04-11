#!/usr/bin/env python3
"""
Computational Time Profiling for LaBraM Inference Pipeline.

Measures wall-clock inference time broken down into 4 consecutive stages:
  1. Data Loading      — HDF5 read + numpy array construction
  2. Segmentation      — Sliding-window slice (512 samples = 2s @ 256 Hz)
  3. Patchification    — Convert 512-sample window → 3 overlapping 200-sample patches
  4. Model Forward     — Full model forward pass (no grad)

Reports mean ± std in microseconds (μs) across N_REPEATS iterations.
Also prints a summary table and saves results to a CSV.

Usage:
    python profile_inference_time.py
    python profile_inference_time.py --n_repeats 200 --fold 0
    python profile_inference_time.py --n_repeats 100 --out_dir runs/profiling
"""

import argparse
import csv
import time
import json
from pathlib import Path

import numpy as np
import torch
import h5py
from einops import rearrange
from timm.models import create_model

import modeling_finetune  # noqa — registers labram_* models
from utils import get_input_chans

# ─── Config ───────────────────────────────────────────────────────────────────
N_WARMUP   = 20      # warmup iterations (GPU JIT / caching)
N_REPEATS  = 100     # timed iterations
WINDOW_SZ  = 512     # samples per window (2 s @ 256 Hz)
PATCH_SZ   = 200     # samples per patch
N_PATCHES  = 3       # overlapping patches per window
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRONTAL_IDX = [0, 1, 2, 3, 4, 5, 6, 7]   # FP1, FPZ, FP2, F7, F3, FZ, F4, F8


def sync():
    """Flush GPU pipeline so timing is accurate."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, n_warmup: int, n_repeats: int) -> tuple[float, float]:
    """Warmup then time fn(); return (mean_us, std_us)."""
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.mean(times)), float(np.std(times))


def load_checkpoint(ckpt_path: str):
    model = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=1,
        init_values=0.1,
    )
    setattr(model, "patch_size", 200)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold",       type=int, default=0,
                        help="Which CV fold checkpoint to use (default 0)")
    parser.add_argument("--n_repeats",  type=int, default=N_REPEATS,
                        help=f"Timed iterations (default {N_REPEATS})")
    parser.add_argument("--n_warmup",   type=int, default=N_WARMUP,
                        help=f"Warmup iterations (default {N_WARMUP})")
    parser.add_argument("--out_dir",    type=str, default="runs/profiling",
                        help="Directory to save profile results (default runs/profiling)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find a sample HDF5 file ────────────────────────────────────────────
    boredom_files = sorted(Path("boredom_hdf5").glob("*.h5"))
    if not boredom_files:
        raise FileNotFoundError("No .h5 files in boredom_hdf5/")
    h5_path = str(boredom_files[0])
    print(f"Profiling with: {h5_path}")
    print(f"Device        : {DEVICE}")
    print(f"Warmup        : {args.n_warmup} iters")
    print(f"Timed         : {args.n_repeats} iters\n")

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt_path = f"runs/boredom_cv/fold{args.fold}/checkpoint-best.pth"
    if not Path(ckpt_path).exists():
        # Fallback to pretrained backbone
        ckpt_path = "checkpoints/labram-base.pth"
    print(f"Checkpoint    : {ckpt_path}")
    model = load_checkpoint(ckpt_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Data Loading — open HDF5, read one window's worth of samples
    # ─────────────────────────────────────────────────────────────────────────
    def stage1_load():
        with h5py.File(h5_path, "r") as f:
            key = list(f.keys())[0]
            arr = f[key]["eeg"][:, :WINDOW_SZ]          # 61 × 512
        return arr

    # Read channel info once to build input_chans tensor for model forward pass
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        ch_raw = [c.decode() if isinstance(c, bytes) else c
                  for c in f[key]["eeg"].attrs.get("chOrder", [])]
    ch_indices = get_input_chans(ch_raw)
    input_chans = torch.tensor(ch_indices).long().to(DEVICE)

    # Prefetch once so we have data for downstream stages
    raw_window = stage1_load()
    s1_mu, s1_sd = time_fn(stage1_load, args.n_warmup, args.n_repeats)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Segmentation — sliding-window slice from already-loaded array
    # (This is realistic: in practice a big buffer is loaded & windows are sliced)
    # ─────────────────────────────────────────────────────────────────────────
    # Simulate full file load once (as data_loader would do)
    with h5py.File(h5_path, "r") as f:
        key = list(f.keys())[0]
        full_arr = f[key]["eeg"][:]        # full recording in RAM

    def stage2_segment():
        start = 0
        return full_arr[:, start: start + WINDOW_SZ].copy()

    s2_mu, s2_sd = time_fn(stage2_segment, args.n_warmup, args.n_repeats)
    window_np = stage2_segment()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Patchification — build 3 overlapping 200-sample patches
    # Matches the exact logic in engine_for_finetuning.py
    # ─────────────────────────────────────────────────────────────────────────
    # Pre-convert to tensor (float32, normalised) once — this is done by DataLoader
    window_tensor = torch.from_numpy(window_np).float().unsqueeze(0) / 100.0  # 1×61×512

    def stage3_patchify():
        p1 = window_tensor[:, :, 0:200]
        p2 = window_tensor[:, :, 156:356]
        p3 = window_tensor[:, :, 312:512]
        x  = torch.cat([p1, p2, p3], dim=2)             # 1×61×600
        return rearrange(x, "b n (a t) -> b n a t", t=200)  # 1×61×3×200

    s3_mu, s3_sd = time_fn(stage3_patchify, args.n_warmup, args.n_repeats)
    model_input = stage3_patchify().to(DEVICE)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: Model Forward Pass
    # ─────────────────────────────────────────────────────────────────────────
    def stage4_forward():
        with torch.no_grad():
            return model(model_input, input_chans=input_chans)

    # Extra GPU warmup
    for _ in range(args.n_warmup):
        stage4_forward()
    sync()

    s4_mu, s4_sd = time_fn(stage4_forward, args.n_warmup, args.n_repeats)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    stages = [
        ("Data Loading",    s1_mu, s1_sd,
         "HDF5 open + read 61×512 window"),
        ("Segmentation",    s2_mu, s2_sd,
         "Slice 512-sample window from RAM buffer"),
        ("Patchification",  s3_mu, s3_sd,
         "Build 3×200 overlapping patches + rearrange"),
        ("Model Forward",   s4_mu, s4_sd,
         f"LaBraM-Base forward pass (device={DEVICE})"),
    ]

    total_mu = sum(m for _, m, _, _ in stages)
    total_sd = np.sqrt(sum(s**2 for _, _, s, _ in stages))

    print("=" * 80)
    print("  LaBraM Inference Time Profile")
    print(f"  Device: {DEVICE}  |  Repeats: {args.n_repeats}")
    print("=" * 80)
    print(f"  {'Stage':<20}  {'Mean (μs)':>12}  {'Std (μs)':>10}  "
          f"{'% Total':>8}  Notes")
    print(f"  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*30}")
    for name, mu, sd, note in stages:
        pct = 100 * mu / total_mu if total_mu > 0 else 0
        print(f"  {name:<20}  {mu:>12.1f}  {sd:>10.1f}  {pct:>7.1f}%  {note}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*10}")
    print(f"  {'TOTAL':<20}  {total_mu:>12.1f}  {total_sd:>10.1f}")
    print("=" * 80)

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = out_dir / "inference_time_profile.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "mean_us", "std_us", "pct_total", "notes",
                    "n_repeats", "device"])
        for name, mu, sd, note in stages:
            pct = 100 * mu / total_mu if total_mu > 0 else 0
            w.writerow([name, f"{mu:.2f}", f"{sd:.2f}", f"{pct:.1f}", note,
                        args.n_repeats, str(DEVICE)])
        w.writerow(["TOTAL", f"{total_mu:.2f}", f"{total_sd:.2f}", "100.0",
                    "", args.n_repeats, str(DEVICE)])
    print(f"\n  CSV saved → {csv_path}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = out_dir / "inference_time_profile.json"
    result = {
        "device":    str(DEVICE),
        "n_repeats": args.n_repeats,
        "n_warmup":  args.n_warmup,
        "checkpoint": ckpt_path,
        "sample_file": h5_path,
        "stages": [
            {
                "name": name,
                "mean_us": round(mu, 2),
                "std_us":  round(sd, 2),
                "pct_total": round(100 * mu / total_mu, 1) if total_mu > 0 else 0,
                "notes": note,
            }
            for name, mu, sd, note in stages
        ],
        "total_mean_us": round(total_mu, 2),
        "total_std_us":  round(total_sd, 2),
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON saved → {json_path}")


if __name__ == "__main__":
    main()
