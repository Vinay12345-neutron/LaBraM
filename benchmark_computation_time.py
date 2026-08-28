#!/usr/bin/env python3
"""
Benchmark Script for Paper:
1. Segments per participant statistics (Mean, Min, Max, Total across 73 subjects).
2. Table X: Average computational time for evaluating a 61-channel EEG recording of 2 seconds at 256 Hz, evaluated over 100 segments:
   - EEG loading
   - LaBraM model loading from hard drive
   - Preprocessing (patch splitting + channel indexing + normalization)
   - LaBraM evaluation (CPU+GPU)
   - LaBraM evaluation (CPU only)
   - Total time costs
"""

import time
import json
import torch
import numpy as np
import h5py
from pathlib import Path
from einops import rearrange
from modeling_finetune import labram_base_patch200_200

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
            input_chans.append(1)
    return input_chans

def run_participant_segments_stats():
    print("=" * 70)
    print("1. PARTICIPANT SEGMENTS STATISTICS (73 SUBJECTS)")
    print("=" * 70)
    
    with open("boredom_split_fold0.json") as f:
        split_data = json.load(f)
        
    all_files = split_data['train'] + split_data['val'] + split_data['test']
    
    subject_segments = {} # subject_id -> {'boredom': count, 'neutral': count, 'total': count}
    
    for item in all_files:
        fpath = item['file']
        label = item['label'] # 1=Boredom, 0=Neutral
        filename = Path(fpath).name
        subj_id = filename.split('_')[0]
        
        if subj_id not in subject_segments:
            subject_segments[subj_id] = {'boredom': 0, 'neutral': 0, 'total': 0}
            
        try:
            with h5py.File(fpath, 'r') as f:
                key = list(f.keys())[0]
                dset = f[key]['eeg'] if 'eeg' in f[key] else f[key]['data']
                n_samples = dset.shape[1] if dset.shape[0] < dset.shape[1] else dset.shape[0]
                num_windows = (n_samples - 512) // 512 + 1
                
                if label == 1:
                    subject_segments[subj_id]['boredom'] += num_windows
                else:
                    subject_segments[subj_id]['neutral'] += num_windows
                subject_segments[subj_id]['total'] += num_windows
        except Exception as e:
            pass
            
    totals = [v['total'] for v in subject_segments.values()]
    boredoms = [v['boredom'] for v in subject_segments.values()]
    neutrals = [v['neutral'] for v in subject_segments.values()]
    
    print(f"Total Unique Participants Analyzed: {len(subject_segments)}")
    print(f"Total EEG 2-second Segments       : {sum(totals):,}")
    print(f"  - Total Boredom Segments         : {sum(boredoms):,}")
    print(f"  - Total Neutral Segments         : {sum(neutrals):,}")
    print("-" * 70)
    print(f"Average Segments per Participant   : {np.mean(totals):.2f} ± {np.std(totals):.2f} (Min: {np.min(totals)}, Max: {np.max(totals)})")
    print(f"  - Boredom Segments / Participant : {np.mean(boredoms):.2f} ± {np.std(boredoms):.2f} (Min: {np.min(boredoms)}, Max: {np.max(boredoms)})")
    print(f"  - Neutral Segments / Participant : {np.mean(neutrals):.2f} ± {np.std(neutrals):.2f} (Min: {np.min(neutrals)}, Max: {np.max(neutrals)})")
    print("=" * 70)
    print()

def run_table_x_benchmark():
    print("=" * 70)
    print("2. COMPUTATIONAL TIME BENCHMARK (100 SEGMENTS OF 2 SECONDS AT 256 HZ)")
    print("=" * 70)
    
    # 1. Model Loading Time (tested over 10 repeats)
    ckpt_path = Path("runs/boredom_cv/fold0/checkpoint-best.pth")
    if not ckpt_path.exists():
        ckpt_path = Path("runs/boredom_cv/fold0/checkpoint.pth")
        
    model_load_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_temp = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
        model_temp.load_state_dict(ckpt.get('model', ckpt), strict=False)
        t1 = time.perf_counter()
        model_load_times.append(t1 - t0)
        
    mean_model_load = np.mean(model_load_times)
    std_model_load = np.std(model_load_times)
    
    # Setup models on GPU and CPU
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
    model_gpu = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model_gpu.load_state_dict(torch.load(ckpt_path, map_location='cpu').get('model', {}), strict=False)
    model_gpu.to(device_gpu)
    model_gpu.eval()
    
    model_cpu = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    model_cpu.load_state_dict(torch.load(ckpt_path, map_location='cpu').get('model', {}), strict=False)
    model_cpu.to(device_cpu)
    model_cpu.eval()
    
    # Find sample EEG file and channel list
    with open("boredom_split_fold0.json") as f:
        split_data = json.load(f)
    sample_file = split_data['test'][0]['file']
    
    with h5py.File(sample_file, 'r') as f:
        key = list(f.keys())[0]
        dset = f[key]['eeg'] if 'eeg' in f[key] else f[key]['data']
        ch_names = dset.attrs.get('chOrder', standard_1020)
        ch_names = [x.decode('utf-8') if isinstance(x, bytes) else x for x in ch_names]
        input_chans_list = get_input_chans(ch_names)

    chans_gpu = torch.tensor(input_chans_list, dtype=torch.long).to(device_gpu)
    chans_cpu = torch.tensor(input_chans_list, dtype=torch.long).to(device_cpu)

    # Warmup GPU
    dummy_input = torch.randn(1, 61, 3, 200).to(device_gpu)
    for _ in range(10):
        with torch.no_grad():
            _ = model_gpu(dummy_input, input_chans=chans_gpu)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark over 100 segments
    n_segments = 100
    loading_times = []
    preprocessing_times = []
    eval_gpu_times = []
    eval_cpu_times = []
    
    with h5py.File(sample_file, 'r') as f:
        key = list(f.keys())[0]
        dset = f[key]['eeg'] if 'eeg' in f[key] else f[key]['data']
        
        # Benchmark 1: EEG Loading (reading 2-sec window = 512 samples from disk)
        for i in range(n_segments):
            start_idx = (i * 10) % (dset.shape[1] - 512)
            t0 = time.perf_counter()
            raw_segment = dset[:, start_idx:start_idx+512]
            t1 = time.perf_counter()
            loading_times.append(t1 - t0)
            
            # Benchmark 2: Preprocessing (normalization + patch slicing + rearrangement)
            t0 = time.perf_counter()
            segment_norm = raw_segment / 100.0
            tensor_cpu = torch.tensor(segment_norm, dtype=torch.float32).unsqueeze(0)
            p1 = tensor_cpu[:, :, 0:200]
            p2 = tensor_cpu[:, :, 156:356]
            p3 = tensor_cpu[:, :, 312:512]
            x_cat = torch.cat([p1, p2, p3], dim=2)
            x_input_cpu = rearrange(x_cat, 'b n (a t) -> b n a t', t=200)
            chans_cpu = torch.tensor(input_chans_list, dtype=torch.long)
            t1 = time.perf_counter()
            preprocessing_times.append(t1 - t0)
            
            # Benchmark 3: GPU Evaluation
            if torch.cuda.is_available():
                x_input_gpu = x_input_cpu.to(device_gpu)
                chans_gpu = chans_cpu.to(device_gpu)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model_gpu(x_input_gpu, input_chans=chans_gpu)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                eval_gpu_times.append(t1 - t0)
            else:
                eval_gpu_times.append(0.0)
                
            # Benchmark 4: CPU Evaluation
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model_cpu(x_input_cpu, input_chans=chans_cpu)
            t1 = time.perf_counter()
            eval_cpu_times.append(t1 - t0)
            
    mean_load = np.mean(loading_times)
    std_load = np.std(loading_times)
    
    mean_prep = np.mean(preprocessing_times)
    std_prep = np.std(preprocessing_times)
    
    mean_gpu = np.mean(eval_gpu_times)
    std_gpu = np.std(eval_gpu_times)
    
    mean_cpu = np.mean(eval_cpu_times)
    std_cpu = np.std(eval_cpu_times)
    
    total_gpu_pipeline = mean_load + mean_prep + mean_gpu
    std_total_gpu = np.sqrt(std_load**2 + std_prep**2 + std_gpu**2)
    
    total_cpu_pipeline = mean_load + mean_prep + mean_cpu
    std_total_cpu = np.sqrt(std_load**2 + std_prep**2 + std_cpu**2)
    
    print("Table X. Average computational time for evaluating a 61-channel EEG recording")
    print("of 2 seconds at 256 Hz, evaluated over 100 segments.\n")
    print(f"{'Task':<45} | {'Time cost (in sec)'}")
    print("-" * 70)
    print(f"{'EEG loading':<45} | {mean_load:.6f} ± {std_load:.6f}")
    print(f"{'LaBraM model loading from the hard drive':<45} | {mean_model_load:.4f} ± {std_model_load:.4f}")
    print(f"{'Preprocessing':<45} | {mean_prep:.6f} ± {std_prep:.6f}")
    print(f"{'LaBraM evaluation (CPU+GPU)':<45} | {mean_gpu:.6f} ± {std_gpu:.6f}")
    print(f"{'LaBraM evaluation (CPU only)':<45} | {mean_cpu:.6f} ± {std_cpu:.6f}")
    print("-" * 70)
    print(f"{'Total time costs (CPU+GPU Pipeline)':<45} | {total_gpu_pipeline:.6f} ± {std_total_gpu:.6f}")
    print(f"{'Total time costs (CPU-only Pipeline)':<45} | {total_cpu_pipeline:.6f} ± {std_total_cpu:.6f}")
    print("=" * 70)

if __name__ == "__main__":
    run_participant_segments_stats()
    run_table_x_benchmark()
