import json
import os
from pathlib import Path
import csv
import numpy as np

def main():
    out_root = Path("/home/user/Vinay/LaBraM/runs/boredom_loso")
    all_metrics = []

    for subject_dir in out_root.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("S"):
            continue
        
        test_subject = int(subject_dir.name[1:])
        log_file = subject_dir / "log.txt"
        
        if log_file.exists():
            with open(log_file) as f:
                lines = f.readlines()
            if lines:
                try:
                    last = json.loads(lines[-1].strip())
                    fold_metrics = {
                        "test_subject": int(test_subject),
                        "test_acc": last.get("test_accuracy"),
                        "test_roc_auc": last.get("test_roc_auc"),
                        "test_pr_auc": last.get("test_pr_auc"),
                        "test_precision": last.get("test_precision"),
                        "test_recall": last.get("test_recall"),
                        "test_f1": last.get("test_f1"),
                        "test_specificity": last.get("test_specificity"),
                    }
                    all_metrics.append(fold_metrics)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse metrics for S{test_subject}: {e}")

    # Sort by subject
    all_metrics.sort(key=lambda x: x["test_subject"])

    if all_metrics:
        accs = [m["test_acc"] for m in all_metrics if m["test_acc"] is not None]
        aucs = [m["test_roc_auc"] for m in all_metrics if m["test_roc_auc"] is not None]
        f1s = [m["test_f1"] for m in all_metrics if m["test_f1"] is not None]

        print(f"Found {len(all_metrics)} completed subjects.")
        if accs:
            print(f"Accuracy:  {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        if aucs:
            print(f"ROC AUC:   {np.mean(aucs)*100:.2f}% ± {np.std(aucs)*100:.2f}%")
        if f1s:
            print(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        summary_path = out_root / "loso_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"JSON saved to {summary_path}")

        csv_path = out_root / "loso_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
