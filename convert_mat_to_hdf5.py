#!/usr/bin/env python3
"""
Convert a folder of .mat EEG files into HDF5 files compatible with LaBraM's
SingleShockDataset / ShockDataset.

Each output .h5 will contain one group ('subject_1') with dataset 'eeg'
(shape: [channels, duration_samples]) and dataset attribute 'chOrder' (byte strings).
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
import sys
import warnings
from typing import Optional

DEFAULT_CH_NAMES = [f"Ch{i+1}" for i in range(61)]

def find_eeg_array(mat_dict, varname: Optional[str]=None):
    if varname:
        if varname in mat_dict:
            arr = mat_dict[varname]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr
            else:
                raise ValueError(f"Variable {varname} found but not a 2D ndarray.")
        else:
            raise ValueError(f"Variable {varname} not found in .mat file.")
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            candidates.append((k, v))
    if not candidates:
        raise ValueError("No 2D ndarray variables found in .mat file.")
    def score(item):
        _, arr = item
        score = 0
        if 61 in arr.shape:
            score += 1000
        score += max(arr.shape)
        return score
    candidates.sort(key=score, reverse=True)
    _, arr = candidates[0]
    return arr

def to_channels_by_time(arr: np.ndarray):
    if arr.ndim != 2:
        raise ValueError("EEG array must be 2D")
    if arr.shape[0] == 61:
        return arr.astype(np.float32)
    if arr.shape[1] == 61:
        return arr.T.astype(np.float32)
    warnings.warn(f"Array shape {arr.shape} does not have dimension 61. Will attempt to orient by taking the smaller axis as channels.")
    if arr.shape[0] < arr.shape[1]:
        return arr.astype(np.float32)
    else:
        return arr.T.astype(np.float32)

def load_channel_names_from_file(path: Path):
    with open(path, 'r') as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != 61:
        raise ValueError(f"Channel names file must contain 61 names, found {len(names)}")
    return names

def convert_file(mat_path: Path, out_path: Path, mode: str, sr: int, duration_seconds: float,
                 varname: Optional[str], ch_names: Optional[list], to_microvolt: bool,
                 pad: bool, pad_mode: str):
    mat = loadmat(str(mat_path))
    arr = find_eeg_array(mat, varname=varname)
    arr = to_channels_by_time(arr)  # shape (channels, time)
    channels = arr.shape[0]
    needed = int(round(duration_seconds * sr))
    total_len = arr.shape[1]

    if total_len < needed:
        if not pad:
            raise ValueError(f"{mat_path.name}: time length {total_len} < required {needed} samples")
        deficit = needed - total_len
        if mode == "last":
            pad_width = ((0, 0), (deficit, 0))
            seg = np.pad(arr, pad_width, mode=pad_mode)[:, -needed:]
            print(f"{mat_path.name}: padded {deficit} samples at start (mode={pad_mode})")
        else:
            pad_width = ((0, 0), (0, deficit))
            seg = np.pad(arr, pad_width, mode=pad_mode)[:, :needed]
            print(f"{mat_path.name}: padded {deficit} samples at end (mode={pad_mode})")
    else:
        if mode == "last":
            seg = arr[:, -needed:]
        else:
            seg = arr[:, :needed]

    seg = seg.astype(np.float32)
    if to_microvolt:
        seg = seg * 1e6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        grp = f.create_group("subject_1")
        dset = grp.create_dataset("eeg", data=seg, dtype='float32', compression="gzip")
        if ch_names:
            ch_arr = np.array(ch_names, dtype='S')
        else:
            ch_arr = np.array(DEFAULT_CH_NAMES[:channels], dtype='S')
        dset.attrs['chOrder'] = ch_arr
    return out_path

def main():
    p = argparse.ArgumentParser(description="Convert folder of .mat EEG to per-file .h5 for LaBraM")
    p.add_argument("--input_dir", "-i", required=True, help="Directory with .mat files")
    p.add_argument("--output_dir", "-o", required=True, help="Directory to write .h5 files")
    p.add_argument("--mode", choices=["first", "last"], default="last",
                   help="'first' -> take first duration; 'last' -> take last duration (boredom: last 4.2 min)")
    p.add_argument("--sampling_rate", type=int, default=256)
    p.add_argument("--duration", type=float, default=4.2, help="duration in seconds to extract (default 4.2)")
    p.add_argument("--var", default=None, help="variable name in .mat to use (optional)")
    p.add_argument("--ch_names_file", default=None, help="optional txt file with 61 channel names (one per line)")
    p.add_argument("--to_microvolt", action="store_true", help="multiply values by 1e6 (Volts -> µV)")
    p.add_argument("--ext", default=".mat", help="mat file extension to search (default .mat)")
    p.add_argument("--pad", action="store_true", help="pad files shorter than required instead of failing")
    p.add_argument("--pad_mode", choices=["reflect","edge","wrap"], default="reflect", help="np.pad mode for padding")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.ch_names_file:
        ch_names = load_channel_names_from_file(Path(args.ch_names_file))
    else:
        ch_names = None

    files = sorted([p for p in input_dir.glob(f"*{args.ext}") if p.is_file()])
    if not files:
        print("No files found in", input_dir, file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files. Converting to {output_dir} (mode={args.mode})")
    failed = []
    for mat_file in files:
        try:
            out_file = output_dir / (mat_file.stem + ".h5")
            convert_file(mat_file, out_file, args.mode, args.sampling_rate, args.duration, args.var, ch_names, args.to_microvolt, args.pad, args.pad_mode)
            print("Wrote", out_file)
        except Exception as e:
            print(f"Failed {mat_file.name}: {e}", file=sys.stderr)
            failed.append((mat_file.name, str(e)))
    if failed:
        print(f"\n{len(failed)} files failed. See messages above.")
        sys.exit(2)
    print("All done.")

if __name__ == "__main__":
    main()