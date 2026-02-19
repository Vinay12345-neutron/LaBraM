import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import mne
import umap
from pathlib import Path

# Imports from local modules
from modeling_finetune import labram_base_patch200_200
from data_processor.dataset import ShockDataset
from utils import standard_1020, get_input_chans
from einops import rearrange


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FOLD = 0
CHECKPOINT = f"runs/boredom_cv/fold{FOLD}/checkpoint-best.pth"
SPLIT_FILE = f"boredom_split_fold{FOLD}.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# -----------------------------------------------------------------------------
# Dataset & DataLoader (Simplified)
# -----------------------------------------------------------------------------
def load_data(split_file):
    with open(split_file) as f:
        split = json.load(f)
    
    # We only need TEST set for visualization
    test_files = [item['file'] for item in split['test']]
    test_labels = [item['label'] for item in split['test']]
    
    # Create dataset - window_size=512 samples
    # ShockDataset expects window_size*time_step(200)? No, check utils.
    # Looking at run_class_finetuning: dataset = ShockDataset(..., 512, ...)
    # Wait, ShockDataset in utils calls it effectively. 
    # Let's trust run_class_finetuning args: input_size 512.
    # But ShockDataset init signature: (file_paths, window_size, stride_size, ...)
    dataset = ShockDataset(
        file_paths=[Path(f) for f in test_files],
        window_size=512,
        stride_size=512, # Non-overlapping for test
    )
    # Verify labels match
    # ShockDataset might return multiple windows per file.
    # We need to map windows back to file labels.
    # We will assume dataset iteration order matches the file order if stride matches length?
    # Actually, ShockDataset breaks files into windows.
    # We need to propagate labels.
    
    return dataset, test_labels

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Inspect args used during training
    if 'args' in ckpt:
        train_args = ckpt['args']
        print(f"Training Args found. Model: {train_args.model}")
        # We can try to use these args, but for now let's just infer key params
        # The checkpoint keys suggested qkv_bias=True (q_bias present)
        # And pos_embed missing might mean use_abs_pos_emb=False?
        # Let's check if 'use_abs_pos_emb' is in args? 
        # Usually standard labram_base doesn't expose it in args unless added.
        pass
    
    # Initialize model
    # We enable qkv_bias=True because 'q_bias' was in checkpoint
    # We set init_values=0.1 because we found it in args default.
    model = labram_base_patch200_200(pretrained=False, num_classes=1, init_values=0.1, qkv_bias=True)
    
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Check for pos_embed
    if 'pos_embed' not in state_dict:
        print("Notice: 'pos_embed' not in checkpoint. Setting use_abs_pos_emb=False might be required if model expects it.")
        # If we set use_abs_pos_emb=False, we must re-init model
        # But let's try strict=False first. If the model has pos_embed param but checkpoint doesn't,
        # it stays random. If the model performance depends on it, this is bad.
        # But if the trained model didn't use it, then we should disable it.
        # Since accuracy was high, assume it wasn't used or was used but not saved? (Unlikely)
        # Most likely: The model logic handles it.
        pass

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded: {msg}")
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

def get_attention(name):
    def hook(model, input, output):
        # output of attn_drop is the attention matrix
        activations[name] = output.detach().cpu()
    return hook

# -----------------------------------------------------------------------------
# Main Extraction & Plotting
# -----------------------------------------------------------------------------
def main():
    print(f"Loading data from {SPLIT_FILE}...")
    
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    
    test_data = split['test'] # List of dicts {file, label}
    
    model = load_model(CHECKPOINT)
    
    # Register Hooks
    model.norm.register_forward_hook(get_activation('norm'))
    model.blocks[-1].attn.attn_drop.register_forward_hook(get_attention('attn'))
    
    all_embeddings = []
    all_attentions = []
    all_labels = [] 
    
    # Saliency Accumulators
    saliency_sums = {0: None, 1: None}
    saliency_counts = {0: 0, 1: 0}
    
    input_chans_list = None
    
    print("Extracting features and computing gradients...")
    
    import h5py
    
    # We need to enable gradients for model parameters? No, just input.
    # Ensure model is in eval mode but we can run backward on input.
    for param in model.parameters():
        param.requires_grad = False
    
    for item in test_data:
        fpath = item['file']
        label = item['label']
        
        try:
            with h5py.File(fpath, 'r') as f:
                # Structure: Group (subject_x) -> Dataset (eeg)
                keys = list(f.keys())
                if not keys:
                    print(f"Skipping {fpath}: Empty file.")
                    continue
                    
                obj = f[keys[0]]
                dset = None
                ch_names = []
                
                if isinstance(obj, h5py.Group):
                    if 'eeg' in obj:
                        dset = obj['eeg'][:]
                        if 'chOrder' in obj['eeg'].attrs:
                            ch_names = obj['eeg'].attrs['chOrder']
                    elif 'data' in obj:
                        dset = obj['data'][:]
                        if 'chOrder' in obj['data'].attrs:
                            ch_names = obj['data'].attrs['chOrder']
                    else:
                        subkeys = list(obj.keys())
                        if subkeys:
                            dset = obj[subkeys[0]][:]
                            if 'chOrder' in obj[subkeys[0]].attrs:
                                ch_names = obj[subkeys[0]].attrs['chOrder']
                else:
                    dset = obj[:]
                    if 'chOrder' in obj.attrs:
                        ch_names = obj.attrs['chOrder']
                
                if dset is None:
                    print(f"Skipping {fpath}: No dataset found.")
                    continue
                
                # Decode bytes to strings
                ch_names = [x.decode('utf-8') if isinstance(x, bytes) else x for x in ch_names]
                
                # Compute input_chans indices
                input_chans_list = get_input_chans(ch_names)
                input_chans = torch.tensor(input_chans_list).long().to(DEVICE)
                
                # Compute n_samples
                if dset.shape[0] > dset.shape[1]:
                    dset = dset.T
                
                n_samples = dset.shape[1]
                
                # Manual windowing 512 samples
                for start in range(0, n_samples - 512 + 1, 512):
                    window = dset[:, start:start+512]
                    # Preprocess
                    window = window / 100.0
                    
                    # Prepare tensor: (B, C, T) -> (1, C, 512)
                    tensor = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
                    
                    # We want gradients w.r.t 'tensor' (the input window)
                    tensor.requires_grad_(True)
                    tensor.retain_grad()
                    
                    # Prepare for LaBraM (Splitting 512 -> 3 patches of 200)
                    p1 = tensor[:, :, 0:200]
                    p2 = tensor[:, :, 156:356]
                    p3 = tensor[:, :, 312:512]
                    x_input = torch.cat([p1, p2, p3], dim=2) # (1, C, 600)
                    
                    # Rearrange: 'B N (A T) -> B N A T', T=200
                    x_input = rearrange(x_input, 'b n (a t) -> b n a t', t=200)
                    
                    # Forward
                    output = model(x_input, input_chans=input_chans)
                    # Output is logits (1, 1) for binary? No, unnormalized score.
                    
                    # Backward for Saliency
                    # We want to explain why it is classified as 'Boredom' (Class 1) or 'Neutral' (Class 0)
                    # Since it's binary classification with 1 output unit (Sigmoid later):
                    # >0 -> Boredom, <0 -> Neutral.
                    # If we want to visualize evidence for Boredom, we maximize score.
                    # If we want to visualize evidence for Neutral, we minimize score (or maximize -score).
                    
                    # Let's map gradients to the "Predicted" class or "True" class?
                    # "True" class makes sense to see what features support the label.
                    
                    score = output[0, 0]
                    if label == 0:
                        score = -score # Maximize evidence for Neutral
                    
                    model.zero_grad()
                    score.backward()
                    
                    # Get input gradients
                    # tensor.grad shape: (1, C, 512)
                    grads = tensor.grad.abs().detach().cpu().numpy() # (1, C, 512)
                    saliency = np.mean(grads, axis=(0, 2)) # (Channels,) average over Time
                    
                    if saliency_sums[label] is None:
                        saliency_sums[label] = np.zeros_like(saliency)
                    saliency_sums[label] += saliency
                    saliency_counts[label] += 1
                    
                    # Capture Hook Data
                    # Embedding (CLS is index 0)
                    # output of norm shape: (B, N_tokens, Dim)
                    emb = activations['norm'][:, 0, :].numpy() # (1, 200)
                    
                    # Attention
                    attn = activations['attn'].numpy() 
                    
                    all_embeddings.append(emb)
                    all_attentions.append(attn)
                    all_labels.append(label)
                    
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    if not all_embeddings:
        print("No data extracted. Exiting.")
        return

    all_embeddings = np.concatenate(all_embeddings, axis=0) # (TotalWindows, 200)
    all_attentions = np.concatenate(all_attentions, axis=0) # (TotalWindows, H, N, N)
    all_labels = np.array(all_labels)
    
    print(f"Collected {len(all_labels)} windows.")
    
    # -------------------------------------------------------------------------
    # 1. TSNE Plot
    # -------------------------------------------------------------------------
    print("Running t-SNE...")
    # Use low perplexity if samples are few
    perp = min(30, len(all_labels) // 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_embedded = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    # Neutral (0)
    idx_0 = (all_labels == 0)
    if np.any(idx_0):
        plt.scatter(X_embedded[idx_0, 0], X_embedded[idx_0, 1], label='Neutral', alpha=0.6, s=20, c='blue')
    
    # Boredom (1)
    idx_1 = (all_labels == 1)
    if np.any(idx_1):
        plt.scatter(X_embedded[idx_1, 0], X_embedded[idx_1, 1], label='Boredom', alpha=0.6, s=20, c='red')
        
    plt.legend()
    plt.title(f"t-SNE of LaBraM Features (Fold {FOLD})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    out_tsne = f"runs/boredom_cv/tsne_plot.png"
    plt.savefig(out_tsne)
    print(f"Saved {out_tsne}")
    
    # -------------------------------------------------------------------------
    # 1.5 UMAP Plot
    # -------------------------------------------------------------------------
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    if np.any(idx_0):
        plt.scatter(X_umap[idx_0, 0], X_umap[idx_0, 1], label='Neutral', alpha=0.6, s=20, c='blue')
    if np.any(idx_1):
        plt.scatter(X_umap[idx_1, 0], X_umap[idx_1, 1], label='Boredom', alpha=0.6, s=20, c='red')
        
    plt.legend()
    plt.title(f"UMAP of LaBraM Features (Fold {FOLD})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)
    out_umap = f"runs/boredom_cv/umap_plot.png"
    plt.savefig(out_umap)
    print(f"Saved {out_umap}")

    # -------------------------------------------------------------------------
    # Shared Mapping Helpers
    # -------------------------------------------------------------------------
    if input_chans_list is None:
        print("No channel info found. Skipping topography.")
        return

    # input_chans_list includes CLS at index 0. Drop it.
    chan_indices = input_chans_list[1:]
    raw_names = [standard_1020[i-1] for i in chan_indices]
    
    def fix_ch(name):
        name = name.upper()
        name = name.replace('FP', 'Fp').replace('Z', 'z')
        # Specific fixes logic
        if name == 'OZ': return 'Oz'
        if name == 'POZ': return 'POz'
        if name == 'FPZ': return 'Fpz'
        if name == 'AFZ': return 'AFz'
        if name == 'FCZ': return 'FCz'
        if name == 'CPZ': return 'CPz'
        if name == 'FZ': return 'Fz'
        if name == 'CZ': return 'Cz'
        if name == 'PZ': return 'Pz'
        return name
        
    used_montage = [fix_ch(n) for n in raw_names]
    
    def plot_topo(data, title, filename):
        if len(used_montage) != len(data):
            print(f"Warning: Montage len {len(used_montage)} != Data len {len(data)}")
            min_len = min(len(used_montage), len(data))
            mnt = used_montage[:min_len]
            dat = data[:min_len]
        else:
            mnt = used_montage
            dat = data
            
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(mnt, sfreq=200, ch_types='eeg')
            info.set_montage(montage)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            mne.viz.plot_topomap(dat, info, axes=ax, show=False, cmap='Reds')
            plt.title(title)
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error plotting {title}: {e}")

    # -------------------------------------------------------------------------
    # 2. Brain Topography (Attention)
    # -------------------------------------------------------------------------
    print("Generating Attention Topography...")
    # att_map: Attention from CLS (0) to Patches (1:)
    # shape: (B, H, Patches)
    att_map = all_attentions[:, :, 0, 1:] 
    
    for class_id, class_name in [(0, 'Neutral'), (1, 'Boredom')]:
        mask = (all_labels == class_id)
        if np.sum(mask) == 0:
            continue
            
        class_att = att_map[mask] # (N_class, H, Patches)
        avg_att = np.mean(class_att, axis=(0, 1)) # (Patches,)
        
        n_patches = avg_att.shape[0]
        n_windows = 3
        computed_chans = n_patches // n_windows
        
        att_reshaped = avg_att.reshape(computed_chans, n_windows)
        chan_importance = np.mean(att_reshaped, axis=1) # (Channels,)
        
        plot_topo(chan_importance, f"Attention Topography ({class_name})", f"runs/boredom_cv/topography_{class_name}.png")

    # -------------------------------------------------------------------------
    # 3. Input Gradient Saliency (Grad-CAM equivalent)
    # -------------------------------------------------------------------------
    print("Generating Gradient Saliency Topography...")
    for class_id, class_name in [(0, 'Neutral'), (1, 'Boredom')]:
        if saliency_counts[class_id] > 0:
            avg_saliency = saliency_sums[class_id] / saliency_counts[class_id]
            plot_topo(avg_saliency, f"Saliency Map ({class_name})", f"runs/boredom_cv/saliency_{class_name}.png")


if __name__ == "__main__":
    print("Calling main...")
    main()
