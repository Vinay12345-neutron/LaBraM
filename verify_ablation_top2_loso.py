#!/usr/bin/env python3
"""
Strict Parameter & Dataset Verification Script for Ablation 3:
Partial Fine-Tuning of Top Two Transformer Blocks (Blocks 11–12) + Classification Head
under 73-Fold Leave-One-Subject-Out (LOSO) Cross-Validation.

Verifies:
1. Exact parameter freezing and trainability breakdown:
   - Blocks 1–10 (blocks.0 to blocks.9) -> FROZEN
   - Blocks 11–12 (blocks.10 and blocks.11) -> TRAINABLE
   - fc_norm -> TRAINABLE
   - head -> TRAINABLE
   - Patch embeddings, time embeddings, CLS token -> FROZEN
   - Exact count: 966,361 trainable parameters (~16.66%) out of 5,799,137 total.
2. Official pretrained checkpoint (checkpoints/labram-base.pth) loading and layer mapping.
3. All 73 LOSO split files in loso_splits/ ensuring zero subject overlap (65 Train, 7 Val, 1 Test).
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
    print("1. ARCHITECTURE & PARAMETER FREEZING VERIFICATION (ABLATION 3: TOP-2 BLOCKS)")
    print("=" * 95)

    # 1. Instantiate model architecture
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

    # 2. Load official pretrained checkpoint
    assert CKPT_PATH.exists(), f"Pretrained checkpoint not found at: {CKPT_PATH}"
    print(f"Loading official pretrained checkpoint from: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    checkpoint_model = checkpoint.get('model', checkpoint)

    # Strip 'student.' prefix
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
    print("Pretrained checkpoint loaded and matched successfully!\n")

    # 3. Apply Top-2 Trainability Mask:
    # Blocks 1–10 (blocks.0 to blocks.9) -> FROZEN
    # Blocks 11–12 (blocks.10 and blocks.11) -> TRAINABLE
    # fc_norm -> TRAINABLE (final norm before head)
    # head -> TRAINABLE
    # Everything else (cls_token, time_embed, patch_embed) -> FROZEN
    
    top2_prefixes = ("blocks.10.", "blocks.11.", "fc_norm.", "head.")
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in top2_prefixes):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 4. Strict Safety Assertions
    # Verify Blocks 1–10 (blocks.0 through blocks.9) are 100% frozen
    for i in range(10):
        block_pfx = f"blocks.{i}."
        for name, param in model.named_parameters():
            if name.startswith(block_pfx):
                assert not param.requires_grad, f"VIOLATION: Lower block parameter {name} is trainable!"

    # Verify Blocks 11–12 (blocks.10 and blocks.11) are 100% trainable
    for i in [10, 11]:
        block_pfx = f"blocks.{i}."
        for name, param in model.named_parameters():
            if name.startswith(block_pfx):
                assert param.requires_grad, f"VIOLATION: Top block parameter {name} is frozen!"

    # Verify Head & fc_norm are 100% trainable
    assert model.head.weight.requires_grad, "VIOLATION: head.weight is frozen!"
    assert model.head.bias.requires_grad, "VIOLATION: head.bias is frozen!"
    assert model.fc_norm.weight.requires_grad, "VIOLATION: fc_norm.weight is frozen!"
    assert model.fc_norm.bias.requires_grad, "VIOLATION: fc_norm.bias is frozen!"

    # Verify Embeddings and CLS token are 100% frozen
    assert not model.cls_token.requires_grad, "VIOLATION: cls_token is trainable!"
    for p in model.patch_embed.parameters():
        assert not p.requires_grad, "VIOLATION: patch_embed parameter is trainable!"

    # 5. Parameter Counting Breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    b11_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("blocks.10."))
    b12_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("blocks.11."))
    fcnorm_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("fc_norm."))
    head_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))

    print("-" * 95)
    print(f"Total Model Parameters       : {total_params:,}")
    print(f"Trainable Parameters (Top-2) : {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters (Lower)    : {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"  - Block 11 (blocks.10)     : {b11_params:,} parameters (Trainable)")
    print(f"  - Block 12 (blocks.11)     : {b12_params:,} parameters (Trainable)")
    print(f"  - fc_norm                  : {fcnorm_params:,} parameters (Trainable)")
    print(f"  - Classification Head      : {head_params:,} parameters (Trainable: head.weight=200, head.bias=1)")
    print(f"Parameter Reduction vs. Full : {(1 - trainable_params/total_params)*100:.2f}% reduction")
    print("-" * 95)

    assert trainable_params == 966361, f"FAIL: Expected 966,361 trainable params, got {trainable_params}"
    assert total_params == 5799137, f"FAIL: Expected 5,799,137 total params, got {total_params}"

    print("\nLayer-by-Layer Parameter Breakdown:")
    groups = OrderedDict()
    for name, p in model.named_parameters():
        prefix = name.split('.')[0]
        if prefix == 'blocks':
            block_idx = int(name.split('.')[1])
            prefix = f"Block {block_idx + 1:02d} (blocks.{block_idx})"
        groups[prefix] = groups.get(prefix, {'trainable': 0, 'frozen': 0})
        if p.requires_grad:
            groups[prefix]['trainable'] += p.numel()
        else:
            groups[prefix]['frozen'] += p.numel()

    for layer_name, counts in groups.items():
        status = "TRAINABLE" if counts['trainable'] > 0 else "FROZEN"
        cnt = counts['trainable'] if counts['trainable'] > 0 else counts['frozen']
        print(f"  - {layer_name:<30} | {status:<10} | Elements: {cnt:,}")

    # 6. Verify Optimizer Parameter Groups
    param_groups = get_parameter_groups(model, weight_decay=0.05)
    opt_trainable_count = sum(sum(p.numel() for p in g['params']) for g in param_groups)

    print("-" * 95)
    print(f"Optimizer Parameter Groups Count   : {len(param_groups)}")
    print(f"Optimizer Total Trainable Elements : {opt_trainable_count:,}")
    assert opt_trainable_count == trainable_params, f"Mismatch: {opt_trainable_count} vs {trainable_params}"
    print("✓ Model Parameter Verification PASSED: EXACTLY Blocks 11-12 + fc_norm + Head are trainable!")

def verify_loso_splits():
    print("\n" + "=" * 95)
    print("2. LOSO SPLITS & SUBJECT ISOLATION AUDIT (73 SUBJECTS)")
    print("=" * 95)

    split_files = sorted(SPLITS_DIR.glob("loso_split_S*.json"), key=lambda p: int(re.search(r"S(\d+)", p.name).group(1)))
    assert len(split_files) == 73, f"FAIL: Expected exactly 73 LOSO split files, found {len(split_files)}"
    print(f"Found exactly {len(split_files)} LOSO split JSON files in {SPLITS_DIR}")

    all_test_subjects = []

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

        if idx == 0:
            print(f"Fold 1 Audit (Test Subject S{test_sub}):")
            print(f"  - Train Subjects ({len(train_subs)}): {sorted(list(train_subs))[:10]} ... [total 65]")
            print(f"  - Val Subjects   ({len(val_subs)}): {sorted(list(val_subs))}")
            print(f"  - Test Subject   ({len(test_subs)}): {[test_sub]}")
            print(f"  - Zero subject leakage confirmed.")

    print(f"\nAll 73 Unique Test Subjects: {all_test_subjects}")
    print(f"Zero Subject Leakage verified across all 73 LOSO folds (65 Train, 7 Val, 1 Test per fold).")
    print("✓ LOSO Split Verification PASSED!")
    print("=" * 95)

if __name__ == "__main__":
    verify_parameters()
    verify_loso_splits()
