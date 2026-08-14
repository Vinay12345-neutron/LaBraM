#!/env/bin/python3
"""
Generate publication-quality LOSO Subject Accuracy Bar Plot.
- X-axis: Participant IDs (in natural ID order, unsorted by accuracy).
- Y-axis: Accuracy (%).
- Thin space between adjacent participant bars (width=0.65).
- 350 DPI, Arial font (16-18pt), Titleless format matching IEEE specifications.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18

def main():
    json_path = Path("runs/boredom_loso/loso_summary.json")
    if not json_path.exists():
        json_path = Path("runs/boredom_loso_bestparam/loso_summary.json")
        with open(json_path) as f:
            raw = json.load(f)
            data = raw.get("folds", raw)
    else:
        with open(json_path) as f:
            data = json.load(f)

    # Keep participant ID order without sorting by accuracy
    subj_ids = [f"S{d['test_subject']}" for d in data]
    accs = [d['test_acc'] * 100 for d in data]
    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(18, 6.5), dpi=350)

    x = np.arange(len(subj_ids))
    # Bar width 0.65 leaves a thin space between each participant bar
    ax.bar(x, accs, width=0.65, color="#2b5c8f", edgecolor="none", alpha=0.9)

    # Horizontal mean accuracy line
    ax.axhline(y=mean_acc, color="#c0392b", linestyle="--", linewidth=2.0, label=f"Mean Acc = {mean_acc:.2f}%")

    ax.set_xlabel("Participant ID", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight="bold", labelpad=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(subj_ids, rotation=90, fontsize=10, fontweight="bold")
    
    # Reduce whitespace on left and right borders
    ax.set_xlim([-0.6, len(subj_ids) - 0.4])
    ax.set_ylim([0, 105])
    
    # Major and minor axis ticks configuration
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    ax.tick_params(axis='y', which='major', length=6, width=1.2, labelsize=16)
    ax.tick_params(axis='y', which='minor', length=3.5, width=0.8)
    ax.tick_params(axis='x', which='major', length=4, width=1.0)
    
    # Remove top and right box spines for an open, modern aesthetic
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=14, frameon=True)

    plt.tight_layout()

    out_dir = Path("runs/boredom_loso")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loso_subject_accuracy_bar.png"
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    
    # Also save copy in runs/boredom_cv for convenience
    out_cv = Path("runs/boredom_cv") / "loso_subject_accuracy_bar.png"
    plt.savefig(out_cv, dpi=350, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path}")
    print(f"Saved: {out_cv}")

if __name__ == "__main__":
    main()
