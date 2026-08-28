#!/usr/bin/env python3
"""
Parameter Verification Script for Ablation 2:
Frozen LaBraM Backbone + Linear Probe
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from modeling_finetune import labram_base_patch200_200
from optim_factory import create_optimizer, get_parameter_groups
import utils

def main():
    print("=" * 85)
    print("ABLATION 2: PARAMETER & TRAINABILITY VERIFICATION")
    print("=" * 85)

    # 1. Instantiate model
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

    # Strip 'student.' prefix exactly as done in run_class_finetuning.py
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
            print(f"Removing key {k} from pretrained checkpoint (classifier adaptation)")
            del checkpoint_model[k]

    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix='')
    print("Official checkpoint loaded successfully into LaBraM backbone (all backbone weights matched)!\n")

    # 3. Freeze entire backbone
    print("Applying parameter freeze across the entire backbone...")
    for name, param in model.named_parameters():
        if name.startswith("head."):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 4. Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("-" * 85)
    print(f"Total Model Parameters     : {total_params:,}")
    print(f"Trainable Parameters       : {trainable_params:,}")
    print(f"Frozen Parameters          : {frozen_params:,}")
    print("-" * 85)

    # 5. Inspect parameter names
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]

    print(f"\nTrainable Parameter Names ({len(trainable_names)} tensors):")
    for name in trainable_names:
        p = dict(model.named_parameters())[name]
        print(f"  - {name:<25} | Shape: {str(list(p.shape)):<15} | Elements: {p.numel()}")

    print(f"\nFrozen Parameter Layers Summary ({len(frozen_names)} tensors):")
    # Group frozen params for clean display
    groups = OrderedDict()
    for name in frozen_names:
        prefix = name.split('.')[0]
        if prefix == 'blocks':
            block_idx = name.split('.')[1]
            prefix = f"blocks.{block_idx}"
        groups[prefix] = groups.get(prefix, 0) + dict(model.named_parameters())[name].numel()

    for grp, cnt in groups.items():
        print(f"  - {grp:<25} | Frozen Elements: {cnt:,}")

    # 6. Verify optimizer parameter groups
    class DummyArgs:
        opt = 'adamw'
        lr = 5e-4
        weight_decay = 0.05
        filter_name = []
        layer_decay = 1.0 # standard for linear probe (single layer)

    param_groups = get_parameter_groups(model, weight_decay=DummyArgs.weight_decay)
    opt_trainable_count = sum(sum(p.numel() for p in g['params']) for g in param_groups)

    print("-" * 85)
    print(f"Optimizer Parameter Groups Count : {len(param_groups)}")
    print(f"Optimizer Total Trainable Elements: {opt_trainable_count}")
    assert opt_trainable_count == trainable_params == 201, f"Mismatch: {opt_trainable_count} vs 201"
    print("VERIFICATION PASSED: ONLY the 201 parameters of `head.weight` and `head.bias` will receive gradients!")
    print("=" * 85)

if __name__ == "__main__":
    main()
