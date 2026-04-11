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
- [x] **Strict LOSO Refactor:** `run_boredom_loso_bestparam.py` — Two-phase protocol.
  - **Phase 1:** 9-config grid (`lr` × `epochs`) on a fixed 7-subject pilot set. Best config saved to `runs/boredom_loso_bestparam/best_params.json`.
  - **Phase 2:** All 73 LOSO folds with identical locked params. Val subjects chosen by the same seed every fold (no test-identity leakage).
  - Resume: `--skip_tuning`, `--loso_start_from N`, `--tuning_only`, `--dry_run`.
- [x] **Computational Time Profiling:** `profile_inference_time.py` — Results in `runs/profiling/`.
  - Data Loading: **4035.7 μs** (54.0%), Segmentation: **6.1 μs** (0.1%), Patchification: **19.0 μs** (0.3%), Model Forward: **3418.9 μs** (45.7%). Total: **~7.5 ms/window**.
- [x] **Raw Input Visualization:** `plot_raw_eeg_comparison.py` — Outputs in `runs/publication_figures/raw_eeg_comparison/`.
  - `raw_eeg_waveform_comparison.png` (multi-region + PSD), `raw_eeg_frontal_closeup.png` (per-channel waveform + PSD).
- [x] **Standardized Metrics Visuals:** `plot_metrics_visuals.py` — Outputs in `runs/publication_figures/metrics/`.
  - `confusion_matrices.png`, `roc_curves.png`, `pr_curves.png`, `combined_metrics_panel.png`.
  - Reads from CV folds by default; use `--source loso_bestparam` once LOSO run completes.
- [x] **Interpretability Topographies (4-Case Analysis):** `plot_topography_4case.py` — Outputs in `runs/publication_figures/topography_4case/`.
  - 2×2 grid for Attention and Saliency. Individual maps per case (TP/FP/TN/FN). Difference maps.
- [ ] **Baseline Execution:** Run a traditional feature-engineering baseline (e.g., EEGNet or SVM with PSD features) on the exact same LOSO split to provide comparison tables in the final paper.

---

## 4. Handover & Next Steps
**Current Immediate Goal:** Run the strict LOSO experiment, then proceed with **Computational Time Profiling**.
**Action to take:**
```bash
# Step 1 — Phase 1 only (fast sanity check, ~20-30 min):
python run_boredom_loso_bestparam.py --tuning_only

# Step 2 — Full run:
python run_boredom_loso_bestparam.py --skip_tuning   # if Phase 1 already done
# OR
python run_boredom_loso_bestparam.py                 # Phase 1 + Phase 2 together
```

*(Last Updated: 2026-04-11)*
