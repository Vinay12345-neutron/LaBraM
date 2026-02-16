import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(preds, trues, threshold):
    preds_bin = (np.array(preds) >= threshold).astype(int)
    trues = np.array(trues).astype(int)
    
    acc = accuracy_score(trues, preds_bin)
    prec = precision_score(trues, preds_bin, zero_division=0)
    rec = recall_score(trues, preds_bin, zero_division=0)
    f1 = f1_score(trues, preds_bin, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(trues, preds_bin, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity
    }

def main():
    folds = glob.glob("runs/boredom_cv/fold*")
    all_preds = []
    all_trues = []
    
    # Collect data from all folds (using the last epoch presumably, or all epochs?)
    # The user asked for "how would the results look", usually implying the final best model validation.
    # But since we save every epoch in test_predictions.json, we should pick the last one or the best one.
    # For simplicity and robustness, let's take the last available epoch for each fold.
    
    for fold_dir in folds:
        pred_file = f"{fold_dir}/test_predictions.json"
        try:
            with open(pred_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    continue
                # Get last line (last epoch)
                last_epoch_data = json.loads(lines[-1])
                all_preds.extend(last_epoch_data["preds"])
                all_trues.extend(last_epoch_data["trues"])
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Skipping {fold_dir}, no valid predictions found.")
            continue

    if not all_preds:
        print("No prediction data found.")
        return

    thresholds = [0.5, 0.6, 0.7, 0.8]
    results = {}
    
    print(f"Analyzed {len(all_preds)} samples across {len(folds)} folds.")
    print("-" * 60)
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<10}")
    print("-" * 60)
    
    metrics_map = {k: [] for k in ["accuracy", "f1", "precision", "recall", "specificity"]}
    
    for thresh in thresholds:
        m = compute_metrics(all_preds, all_trues, thresh)
        results[thresh] = m
        print(f"{thresh:<10.1f} | {m['accuracy']:<10.4f} | {m['f1']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['specificity']:<10.4f}")
        
        for k in metrics_map:
            metrics_map[k].append(m[k])

    # Plotting
    plt.figure(figsize=(10, 6))
    for k, v in metrics_map.items():
        plt.plot(thresholds, v, marker='o', label=k.capitalize())
        
    plt.title("Performance Metrics vs. Probability Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig("runs/boredom_cv/threshold_analysis.png")
    print("\nPlot saved to runs/boredom_cv/threshold_analysis.png")

if __name__ == "__main__":
    main()
