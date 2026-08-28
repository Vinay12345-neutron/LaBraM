#!/usr/bin/env python3
"""
Parameter Verification Script for Ablation 3:
Partial Fine-Tuning of Top Two Transformer Blocks (Blocks 11–12) + Classification Head
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from modeling_finetune import labram_base_patch200_200
from optim_factory import get_parameter_groups
import utils

def main():
    print("=" * 95)
    print("ABLATION 3: TOP-2 TRANSFORMER BLOCKS (BLOCKS 11-12) + HEAD VERIFICATION")
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
    ckpt_path = "checkpoints/labram-base.pth"
    print(f"Loading official pretrained checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
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

    # Verify Head is 100% trainable
    assert model.head.weight.requires_grad, "VIOLATION: head.weight is frozen!"
    assert model.head.bias.requires_grad, "VIOLATION: head.bias is frozen!"

    # 5. Parameter Counting Breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("-" * 95)
    print(f"Total Model Parameters       : {total_params:,}")
    print(f"Trainable Parameters (Top-2) : {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Frozen Parameters (Lower)    : {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"Parameter Reduction vs. Full : {(1 - trainable_params/total_params)*100:.2f}% reduction")
    print("-" * 95)

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
    class DummyArgs:
        opt = 'adamw'
        lr = 5e-4
        weight_decay = 0.05
        filter_name = []
        layer_decay = 0.9

    param_groups = get_parameter_groups(model, weight_decay=DummyArgs.weight_decay)
    opt_trainable_count = sum(sum(p.numel() for p in g['params']) for g in param_groups)

    print("-" * 95)
    print(f"Optimizer Parameter Groups Count   : {len(param_groups)}")
    print(f"Optimizer Total Trainable Elements : {opt_trainable_count:,}")
    assert opt_trainable_count == trainable_params, f"Mismatch: {opt_trainable_count} vs {trainable_params}"
    print("VERIFICATION SUCCESSFUL: EXACTLY the intended Top-2 Blocks + Head are trainable!")
    print("=" * 95)

if __name__ == "__main__":
    main()
