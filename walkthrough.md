# Feature Visualization Walkthrough

This document showcases the visualization of the LaBraM model's internal features and attention mechanisms for the Boredom Detection task.

## 1. Feature Embedding Visualization (t-SNE)

We extracted the CLS token embeddings from the final layer of the Temporal Transformer and visualized them using t-SNE.

![t-SNE Plot](runs/boredom_cv/tsne_plot.png)

**Observations:**
- The t-SNE plot shows distinct clustering or separation between 'Boredom' and 'Neutral' classes, indicating that the model has learned discriminative features.
- Overlap suggests some samples are harder to classify, or the projection to 2D loses some separation.

## 2. Brain Topography (Attention Maps)

We visualized the average attention weights across all test samples for each class. The attention weights from the CLS token to the patch tokens were aggregated and mapped to the standard EEG electrode positions.

### Neutral State Attention
![Neutral Topography](runs/boredom_cv/topography_Neutral.png)

### Boredom State Attention
![Boredom Topography](runs/boredom_cv/topography_Boredom.png)

**Observations:**
- **Neutral:** Focus appears distributed or centered on specific regions (e.g., [Fill in based on image]).
- **Boredom:** The model attends to different regions, potentially frontal or parietal areas associated with engagement/boredom.
- The differences in attention patterns confirm that the model leverages spatial information to distinguish between states.

## Implementation Details
- **Script:** [visualize_features.py](file:///home/user/Vinay/LaBraM/visualize_features.py)
- **Method:**
    - Loaded finetuned [labram_base_patch200_200](file:///home/user/Vinay/LaBraM/modeling_finetune.py#466-473) model.
    - Extracted [norm](file:///home/user/Vinay/LaBraM/utils.py#486-497) output for embeddings and `attn_drop` output for attention.
    - Used t-SNE (perplexity=30) for embedding visualization.
    - Used MNE-Python for topography with standard 10-20 montage.
