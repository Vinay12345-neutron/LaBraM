import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

with open("runs/boredom_loso/loso_summary.json") as f:
    data = json.load(f)

accs = np.array([d["test_acc"] * 100 for d in data])
accs_sorted = np.sort(accs)
mean_acc = np.mean(accs)
median_acc = np.median(accs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#F44336' if a < 80 else '#FF9800' if a < 90 else '#4CAF50' for a in accs_sorted]
axes[0].bar(range(len(accs_sorted)), accs_sorted, color=colors, width=0.8)
axes[0].axhline(y=mean_acc, color='navy', linestyle='--', linewidth=2, label=f"Mean={mean_acc:.2f}%")
axes[0].set_ylim([0, 105])
axes[0].set_xlabel("Subject Index (sorted)")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title(f"Per-Subject LOSO Accuracy (N={len(accs)})")
axes[0].legend()

# Histogram
axes[1].hist(accs, bins=10, color='#4CAF50', edgecolor='white')
axes[1].axvline(x=mean_acc, color='navy', linestyle='--', linewidth=2, label=f"Mean={mean_acc:.2f}%")
axes[1].axvline(x=median_acc, color='red', linestyle='-.', linewidth=2, label=f"Median={median_acc:.2f}%")
axes[1].set_xlabel("Accuracy (%)")
axes[1].set_ylabel("Number of Subjects")
axes[1].set_title("Distribution of LOSO Accuracy")
axes[1].legend()

plt.tight_layout()
out_path = Path("runs/boredom_loso/loso_performance_correct.png")
plt.savefig(out_path, dpi=150)
print(f"Saved correct plot to {out_path}")
