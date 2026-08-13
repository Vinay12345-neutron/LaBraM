import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

log_path = Path("runs/epoch_exp/ep50/log.txt")
epochs = {}
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except: continue
        if "epoch" in d:
            epochs[d["epoch"]] = d

sorted_data = [v for _, v in sorted(epochs.items())]
ep_nums = [d["epoch"] for d in sorted_data]

fig, ax = plt.subplots(figsize=(7, 5))
train_acc = [d.get("train_window_acc", d.get("train_class_acc", np.nan)) for d in sorted_data]
val_acc = [d.get("val_accuracy", np.nan) for d in sorted_data]
test_acc = [d.get("test_accuracy", np.nan) for d in sorted_data]

ax.plot(ep_nums, train_acc, 'o-', label="Train Acc", color="#2196F3", markersize=4)
ax.plot(ep_nums, val_acc, 's-', label="Val Acc", color="#FF9800", markersize=4)
if any(x is not None and not np.isnan(x) for x in test_acc):
    ax.plot(ep_nums, test_acc, '^-', label="Test Acc", color="#F44336", markersize=4)

ax.axvline(x=5, color='black', linestyle='--', label="Epoch 5 (Selected)")

ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Epoch Convergence Analysis (50 Epochs)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])
ax.set_xlim([0, 51])

out_path = Path("runs/epoch_exp/training_curves_50epochs.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
