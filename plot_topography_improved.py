#!/usr/bin/env python3
"""
Improved Brain Topography Visualization with:
- Channel name labels on the plot
- Better colormaps (RdBu_r, inferno)  
- Colorbars
- Difference maps (Boredom - Neutral)
- Both Attention and Saliency

Usage:
    python plot_topography_improved.py
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from pathlib import Path

from modeling_finetune import labram_base_patch200_200
from utils import standard_1020, get_input_chans
from einops import rearrange
import h5py

# ─── Configuration ───────────────────────────────────────────────────────────
FOLD = 0
CHECKPOINT = f"runs/boredom_cv/fold{FOLD}/checkpoint-best.pth"
SPLIT_FILE = f"boredom_split_fold{FOLD}.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path("runs/boredom_cv")


def fix_ch(name):
    """Convert standard_1020 uppercase names to MNE-compatible format."""
    replacements = {
        'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
        'AFZ': 'AFz', 'FCZ': 'FCz', 'CPZ': 'CPz', 'POZ': 'POz',
    }
    upper = name.upper()
    if upper in replacements:
        return replacements[upper]
    # General z-suffix fix
    if upper.endswith('Z') and len(upper) > 1:
        return upper[:-1] + 'z'
    return upper


def load_model(checkpoint_path):
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded: {msg}")
    return model


def plot_topo(data, ch_names, title, filename, cmap='RdBu_r', show_names=True):
    """Plot topography with channel names and colorbar."""
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names, sfreq=200, ch_types='eeg')
        info.set_montage(montage)

        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Try with names parameter (MNE version compatible)
        topo_kwargs = dict(axes=ax, show=False, cmap=cmap, contours=6)
        if show_names:
            topo_kwargs['names'] = ch_names
        
        try:
            im, _ = mne.viz.plot_topomap(data, info, **topo_kwargs)
        except TypeError:
            # Fallback: remove names if not supported
            topo_kwargs.pop('names', None)
            im, _ = mne.viz.plot_topomap(data, info, **topo_kwargs)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Manually add channel labels if names kwarg failed
        if show_names:
            try:
                pos = mne.channels.make_standard_montage('standard_1020')
                ch_pos = pos.get_positions()['ch_pos']
                for ch in ch_names:
                    if ch in ch_pos:
                        x, y = ch_pos[ch][:2]
                        ax.text(x, y, ch, fontsize=5, ha='center', va='center', alpha=0.7)
            except Exception:
                pass  # Channel labels are nice-to-have, not critical
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Error plotting {title}: {e}")


def main():
    # Load split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    test_data = split.get('test', [])
    print(f"Test data: {len(test_data)} files")

    # Load model
    model = load_model(CHECKPOINT)
    model.to(DEVICE)
    model.eval()

    # Hooks
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    def get_attention(name):
        def hook(model, input, output):
            activations[name] = input[0].detach().cpu()
        return hook

    model.norm.register_forward_hook(get_activation('norm'))
    model.blocks[-1].attn.attn_drop.register_forward_hook(get_attention('attn'))

    # Disable param gradients, enable input gradients
    for param in model.parameters():
        param.requires_grad = False

    # Accumulators
    all_attentions = []
    all_labels = []
    saliency_sums = {0: None, 1: None}
    saliency_counts = {0: 0, 1: 0}
    input_chans_list = None

    print("Extracting features and computing gradients...")
    for item in test_data:
        fpath = item['file']
        label = item['label']
        try:
            with h5py.File(fpath, 'r') as f:
                keys = list(f.keys())
                if not keys:
                    continue
                obj = f[keys[0]]
                dset = None
                ch_names_raw = []

                if isinstance(obj, h5py.Group):
                    for key in ['eeg', 'data']:
                        if key in obj:
                            dset = obj[key][:]
                            if 'chOrder' in obj[key].attrs:
                                ch_names_raw = obj[key].attrs['chOrder']
                            break
                else:
                    dset = obj[:]
                    if 'chOrder' in obj.attrs:
                        ch_names_raw = obj.attrs['chOrder']

                if dset is None:
                    continue

                ch_names_raw = [x.decode('utf-8') if isinstance(x, bytes) else x for x in ch_names_raw]
                input_chans_list = get_input_chans(ch_names_raw)
                input_chans = torch.tensor(input_chans_list).long().to(DEVICE)

                if dset.shape[0] > dset.shape[1]:
                    dset = dset.T

                n_samples = dset.shape[1]
                for start in range(0, n_samples - 512 + 1, 512):
                    window = dset[:, start:start + 512] / 100.0
                    tensor = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
                    tensor.requires_grad_(True)
                    tensor.retain_grad()

                    p1 = tensor[:, :, 0:200]
                    p2 = tensor[:, :, 156:356]
                    p3 = tensor[:, :, 312:512]
                    x_input = torch.cat([p1, p2, p3], dim=2)
                    x_input = rearrange(x_input, 'b n (a t) -> b n a t', t=200)

                    output = model(x_input, input_chans=input_chans)

                    # Saliency
                    score = output[0, 0]
                    if label == 0:
                        score = -score
                    model.zero_grad()
                    score.backward()

                    grads = tensor.grad.abs().detach().cpu().numpy()
                    saliency = np.mean(grads, axis=(0, 2))
                    if saliency_sums[label] is None:
                        saliency_sums[label] = np.zeros_like(saliency)
                    saliency_sums[label] += saliency
                    saliency_counts[label] += 1

                    # Attention
                    attn = activations['attn'].numpy()
                    all_attentions.append(attn)
                    all_labels.append(label)

        except Exception as e:
            print(f"Error: {fpath}: {e}")
            continue

    if not all_attentions:
        print("No data extracted.")
        return

    all_attentions = np.concatenate(all_attentions, axis=0)
    all_labels = np.array(all_labels)
    print(f"Collected {len(all_labels)} windows.")

    # Build channel montage
    if input_chans_list is None:
        print("No channel info.")
        return

    chan_indices = input_chans_list[1:]  # Drop CLS
    raw_names = [standard_1020[i - 1] for i in chan_indices]
    used_montage = [fix_ch(n) for n in raw_names]
    print(f"Channels ({len(used_montage)}): {used_montage}")

    # ─── Attention Topography ────────────────────────────────────────────────
    att_map = all_attentions[:, :, 0, 1:]  # CLS → patches
    att_data = {}

    for class_id, class_name in [(0, 'Neutral'), (1, 'Boredom')]:
        mask = (all_labels == class_id)
        if np.sum(mask) == 0:
            continue
        class_att = att_map[mask]
        avg_att = np.mean(class_att, axis=(0, 1))
        n_patches = avg_att.shape[0]
        n_windows = 3
        computed_chans = n_patches // n_windows
        att_reshaped = avg_att.reshape(computed_chans, n_windows)
        chan_importance = np.mean(att_reshaped, axis=1)

        # Truncate if needed
        min_len = min(len(used_montage), len(chan_importance))
        att_data[class_name] = chan_importance[:min_len]

        plot_topo(chan_importance[:min_len], used_montage[:min_len],
                  f"Attention — {class_name}",
                  OUT_DIR / f"topo_attention_{class_name}.png",
                  cmap='Reds')

    # Difference map
    if 'Boredom' in att_data and 'Neutral' in att_data:
        diff = att_data['Boredom'] - att_data['Neutral']
        min_len = min(len(used_montage), len(diff))
        plot_topo(diff[:min_len], used_montage[:min_len],
                  "Attention Difference (Boredom − Neutral)",
                  OUT_DIR / "topo_attention_diff.png",
                  cmap='RdBu_r')

    # ─── Saliency Topography ─────────────────────────────────────────────────
    sal_data = {}
    for class_id, class_name in [(0, 'Neutral'), (1, 'Boredom')]:
        if saliency_counts[class_id] > 0:
            avg_sal = saliency_sums[class_id] / saliency_counts[class_id]
            min_len = min(len(used_montage), len(avg_sal))
            sal_data[class_name] = avg_sal[:min_len]

            plot_topo(avg_sal[:min_len], used_montage[:min_len],
                      f"Input Saliency — {class_name}",
                      OUT_DIR / f"topo_saliency_{class_name}.png",
                      cmap='inferno')

    # Saliency difference
    if 'Boredom' in sal_data and 'Neutral' in sal_data:
        diff = sal_data['Boredom'] - sal_data['Neutral']
        min_len = min(len(used_montage), len(diff))
        plot_topo(diff[:min_len], used_montage[:min_len],
                  "Saliency Difference (Boredom − Neutral)",
                  OUT_DIR / "topo_saliency_diff.png",
                  cmap='RdBu_r')

    print("\nDone! All improved topography plots saved.")


if __name__ == "__main__":
    main()
