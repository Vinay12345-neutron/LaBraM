# 🧠 LaBraM EEG Boredom Detection: Progress & Handover Ground Truth

> **Lead Research Engineer Note:** This file acts as the absolute ground truth for our project state. Because we are developing across a local laptop and a remote RTX 4090 server, **always read this document first upon login**, and mandate an update to this document upon logout.

## 1. Project Context & Objectives
We are fine-tuning the foundational **LaBraM (Large Brain Model)** for a binary cognitive state classification task: **Boredom vs. Neutral baseline**. 
* **Adviser:** Professor Yuvaraj (BITS Goa).
* **Target:** Submission to a top-tier neuroscience/machine learning journal.
* **Core Pipeline:** Processing raw EEG signals into 2-second windows (512 samples at 256Hz). These are processed into overlapping temporal patches (3x200) fed into the Transformers. 
* **Validation Standard:** A strict 73-subject Leave-One-Subject-Out (LOSO) cross-validation and a 5-fold GroupKFold baseline.

---

## 2. Current State & Assets Summary

### Existing Datasets
* **`boredom_hdf5/` & `neutral_hdf5/`**: Raw preprocessed EEG data chunks.
* **`loso_splits/`**: The generated JSON split tracking files for 73 folds ensuring absolute disjointness.

### Scripts & Codebase
* `run_class_finetuning.py`: The core LaBraM fine-tuning loop (supports distributed training).
* `run_class_finetuning_es.py`: Version with explicit Early Stopping implemented.
* `run_boredom_loso.py`: The automated 73-subject LOSO cross-validation wrapper.
* `run_epoch_experiment.py`: Evaluator testing convergence rates across 5, 10, 25, 50 epochs.
* `plot_topography_improved.py`: Generates MNE 10-20 system attention and saliency plots.
* `plot_training_curves.py`: Generates the matplotlib learning convergence metrics.

### Environment
* **Conda Environment**: `labram` (Requires `torch`, `mne`, `timm`, `matplotlib`).

### Final Current Metrics
* **5-Fold Average Pipeline:** Accuracy: 98.62% | ROC-AUC: 99.73% | F1: 0.985
* **73-Subject LOSO Pipeline:** Accuracy: 96.97% | ROC-AUC: 98.10% | F1: 0.966
*(Convergence verified optimally at 5 epochs).*

---

## 3. Action Protocol & Checklist

### ✅ Completed
- [x] Initial binary mapping of Boredom vs Neutral tags.
- [x] 5-Fold GroupKFold baseline proving 98%+ accuracy.
- [x] 73-Subject strict LOSO proving 96.9% zero-shot accuracy across unseen humans.
- [x] Epoch experimentation proving convergence at Epoch 5.
- [x] Initial visual Topographies (Attention and gradient Saliency) with difference maps.

### ⏳ Pending Immediate Tasks (Publication Prep)
- [ ] **Strict LOSO Refactor:** Modify the training script to run hyperparameter tuning strictly *outside* the LOSO loop (on a static validation set), then lock those parameters for all 73 LOSO iterations. This categorically proves zero data or tuning leakage.
- [ ] **Computational Time Profiling:** Implement `time.time()` wrappers to calculate average microsecond inference time broken down by: Data Loading -> Segmentation -> Patchification -> Model Forward Pass.
- [ ] **Raw Input Visualization:** Generate a plot showing a 2-second raw EEG waveform (comparing Boredom vs Neutral state), visually highlighting the frontal lobe amplitude/frequency differences.
- [ ] **Standardized Metrics Visuals:** Generate academic-standard heat-mapped Confusion Matrices (2x2) and overlaid ROC / Precision-Recall curves.
- [ ] **Interpretability Topographies (4-Case Analysis):** Generate 10-20 system brain maps for Attention & Saliency explicitly broken down into True Positives, False Positives, True Negatives, and False Negatives.
- [ ] **Baseline Execution:** Run a traditional feature-engineering baseline (e.g., EEGNet or SVM with PSD features) on the exact same LOSO split to provide comparison tables in the final paper.

---

## 4. Handover & Next Steps
**Current Immediate Goal:** Knock out the **Strict LOSO Refactor** and **Computational Time Profiling**.
**Action to take:** Open `run_boredom_loso.py` and extract the tuning logic, then inject timed blocks into `run_class_finetuning.py` inference stages.

*(Last Updated: 2026-04-09)*
