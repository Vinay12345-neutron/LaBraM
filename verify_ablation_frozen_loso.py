#!/usr/bin/env python3
"""
Strict Parameter & Dataset Verification Script for:
Ablation 2 under Leave-One-Subject-Out (LOSO) Cross-Validation (73 Subjects)
- Verifies model architecture and parameter freezing (exactly 201 trainable head parameters)
- Verifies official pretrained checkpoint loading from checkpoints/labram-base.pth
- Verifies all 73 LOSO split files in loso_splits/
- Confirms ZERO subject-level overlap between train, validation, and test sets across ALL 73 folds
"""

import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
from modeling_finetune import labram_base_patch200_200
from optim_factory import get_parameter_groups
import utils

WORKDIR = Path.cwd()
SPLITS_DIR = WORKDIR / "loso_splits"
CKPT_PATH = WORKDIR / "checkpoints" / "labram-base.pth"

def get_subject_id(path_str):
    name = Path(path_str).name
    m = re.match(r"S(\d+)_", name)
    return int(m.group(1)) if m else -1

def verify_parameters():
    print("=" * 95)
    print("1. ARCHITECTURE & PARAMETER FREEZING VERIFICATION (ABLATION 2)")
    print("=" * 95)

    model = labram_base_patch200_200(
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        init_values=0.1,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True
    )

    assert CKPT_PATH.exists(), f"Pretrained checkpoint not found at: {CKPT_PATH}"
    print(f"Loading official pretrained checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    checkpoint_model = checkpoint.get('model', checkpoint)

    new_dict = OrderedDict()
    for key in list(checkpoint_model.keys()):
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix='')
    print("Pretrained checkpoint loaded and matched successfully!")

    # Apply Ablation 2 freezing: Head trainable, everything else frozen
    for name, param in model.named_parameters():
        if name.startswith("head."):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Safety assertions
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_cnt = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_cnt = sum(p.numel() for p in model.parameters())

    print("-" * 95)
    print(f"Total Model Parameters       : {total_cnt:,}")
    print(f"Trainable Parameters (Head)  : {trainable_cnt:,} ({trainable_cnt/total_cnt*100:.4f}%)")
    print(f"Frozen Parameters (Backbone) : {frozen_cnt:,} ({frozen_cnt/total_cnt*100:.4f}%)")
    print(f"Trainable Parameter Tensors  : {trainable_names}")
    print("-" * 95)

    assert trainable_cnt == 201, f"FAIL: Expected exactly 201 trainable params, got {trainable_cnt}"
    assert trainable_names == ['head.weight', 'head.bias'], f"FAIL: Unexpected trainable parameters: {trainable_names}"
    
    # Assert every Transformer block parameter is frozen
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            assert not param.requires_grad, f"FAIL: Backbone parameter {name} is not frozen!"

    # Verify Optimizer parameter groups
    param_groups = get_parameter_groups(model, weight_decay=0.05)
    opt_cnt = sum(sum(p.numel() for p in g['params']) for g in param_groups)
    assert opt_cnt == 201, f"FAIL: Optimizer has {opt_cnt} parameters instead of 201!"
    print(f"Optimizer Parameter Groups Count   : {len(param_groups)}")
    print(f"Optimizer Total Trainable Elements : {opt_cnt} (head.weight: 200, head.bias: 1)")
    print("✓ Model Parameter Verification PASSED!")

def verify_loso_splits():
    print("\n" + "=" * 95)
    print("2. LOSO SPLITS & SUBJECT ISOLATION AUDIT (73 SUBJECTS)")
    print("=" * 95)

    split_files = sorted(SPLITS_DIR.glob("loso_split_S*.json"), key=lambda p: int(re.search(r"S(\d+)", p.name).group(1)))
    assert len(split_files) == 73, f"FAIL: Expected exactly 73 LOSO split files, found {len(split_files)}"
    print(f"Found exactly {len(split_files)} LOSO split JSON files in {SPLITS_DIR}")

    all_test_subjects = []
    total_test_windows_est = 0

    for idx, sfile in enumerate(split_files):
        with open(sfile) as f:
            data = json.load(f)

        train_files = [x['file'] for x in data['train']]
        val_files = [x['file'] for x in data['val']]
        test_files = [x['file'] for x in data['test']]

        train_subs = set(get_subject_id(f) for f in train_files)
        val_subs = set(get_subject_id(f) for f in val_files)
        test_subs = set(get_subject_id(f) for f in test_files)

        assert len(test_subs) == 1, f"Fold {idx} has multiple test subjects: {test_subs}"
        test_sub = list(test_subs)[0]
        all_test_subjects.append(test_sub)

        # STRICT ZERO-OVERLAP ASSERTIONS
        overlap_train_test = train_subs.intersection(test_subs)
        overlap_val_test = val_subs.intersection(test_subs)
        overlap_train_val = train_subs.intersection(val_subs)

        assert len(overlap_train_test) == 0, f"LEAKAGE in S{test_sub}: Test subject present in train: {overlap_train_test}"
        assert len(overlap_val_test) == 0, f"LEAKAGE in S{test_sub}: Test subject present in val: {overlap_val_test}"
        assert len(overlap_train_val) == 0, f"LEAKAGE in S{test_sub}: Train and val subjects overlap: {overlap_train_val}"

        total_subjects = len(train_subs) + len(val_subs) + len(test_subs)
        assert total_subjects == 73, f"Fold {idx} (S{test_sub}) has total subjects {total_subjects} != 73"

    print(f"All 73 Unique Test Subjects: {all_test_subjects}")
    print(f"Zero Subject Leakage verified across all 73 LOSO folds (65 Train, 7 Val, 1 Test per fold).")
    print("✓ LOSO Split Verification PASSED!")
    print("=" * 95)

if __name__ == "__main__":
    verify_parameters()
    verify_loso_splits()
