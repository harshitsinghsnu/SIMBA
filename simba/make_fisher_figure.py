"""Two figures from `eval_fisher_intra_target.py`'s output:
  (1) overall (pooled) Pearson vs. Fisher-z-averaged intra-target Pearson
  (2) scatter plot of predicted vs. true dG
"""
from __future__ import annotations
import json
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simba.bootstrap_utils import bootstrap_metrics_ci
from simba.figure_style import RCPARAMS, sequential_blues
IN_JSON = "figures/fisher_intra_target_results.json"
plt.rcParams.update(RCPARAMS)

def make_bar_figure(d):
    labels = ["Overall\n(pooled)", "Intra-target\nPearson\n(naive mean)",
              "Intra-target\nPearson\n(Fisher-z mean)", "Intra-target\nSpearman\n(naive mean)",
              "Intra-target\nSpearman\n(Fisher-z mean)"]
    values = [d["overall_pearson"], d["naive_mean_pearson"], d["fisher_mean_pearson"],
              d["naive_mean_spearman"], d["fisher_mean_spearman"]]
    colors = sequential_blues(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 7))
    bars = ax.bar(labels, values, width=0.55, color=colors, edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Correlation coefficient")
    ax.set_ylim(0, max(values) + 0.13)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("figures/fisher_intra_target_bars.png", dpi=300)

def make_scatter_figure(d):
    true = np.array(d["scatter_true"])
    pred = np.array(d["scatter_pred"])
    ci = bootstrap_metrics_ci(true.tolist(), pred.tolist(), n_boot=1000)
    print(f"[fisher scatter] Pearson={ci['pearson']:.3f} [{ci['pearson_ci_lo']:.3f},{ci['pearson_ci_hi']:.3f}]  "
          f"Spearman={ci['spearman']:.3f}  RMSE={ci['rmse']:.2f} kcal/mol")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    
    # EXACT CHANGE: Changed scatter color to a vibrant deep blue 
    ax.scatter(true, pred, s=10, alpha=0.4, color="#0077b6", edgecolors="none")
    
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="perfect prediction")
    ax.set_xlabel("true ΔG (kcal/mol)")
    ax.set_ylabel("predicted ΔG (kcal/mol)")
    ax.legend(fontsize=15)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig("figures/scatter_random_split.png", dpi=300)

def main():
    with open(IN_JSON) as f:
        d = json.load(f)
    make_bar_figure(d)
    make_scatter_figure(d)

if __name__ == "__main__":
    main()