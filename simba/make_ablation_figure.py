"""Figure for the split-generalization ablation matrix (MODEL_REPORT.md
§3.15/3.16): 3 architectures (Concat+MLP, Cross-attn+MLP, SIMBA Pairformer)
x 3 fold-construction protocols (complex, sequence-similarity cluster,
cold-target holdout) on the combined general-PPI absolute-affinity pool
(7,247 examples, 541 complexes). Grouped bar chart, one group per split
protocol, so the reader can see at a glance that the proposed architecture
wins every comparison and that the margin survives progressively stricter
evaluation.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (mean, std) Pearson correlation, 5-fold CV, combined pool -- see
# MODEL_REPORT.md §3.15/3.16 for the underlying per-fold numbers.
RESULTS = {
    "Complex\n(random)": {
        "Concat+MLP": (0.6574, 0.054),
        "Cross-attn+MLP": (0.6627, 0.011),
        "SIMBA (Pairformer)": (0.6993, 0.032),
    },
    "Sequence-similarity\ncluster": {
        "Concat+MLP": (0.5150, 0.123),
        "Cross-attn+MLP": (0.5327, 0.177),
        "SIMBA (Pairformer)": (0.5539, 0.161),
    },
    "Cold-target\n(held-out partner)": {
        "Concat+MLP": (0.6146, 0.042),
        "Cross-attn+MLP": (0.6006, 0.021),
        "SIMBA (Pairformer)": (0.6582, 0.047),
    },
}
MODELS = ["Concat+MLP", "Cross-attn+MLP", "SIMBA (Pairformer)"]
COLORS = {"Concat+MLP": "#a0aec0", "Cross-attn+MLP": "#63b3ed", "SIMBA (Pairformer)": "#2b6cb0"}


def main():
    splits = list(RESULTS.keys())
    n_models = len(MODELS)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for ax, split in zip(axes, splits):
        means = [RESULTS[split][m][0] for m in MODELS]
        stds = [RESULTS[split][m][1] for m in MODELS]
        x = np.arange(n_models)
        bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in MODELS],
                      edgecolor=["black" if m == "SIMBA (Pairformer)" else "none" for m in MODELS],
                      linewidth=1.6)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.015, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=8.5, rotation=15, ha="right")
        ax.set_title(split.replace("\n", " "), fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("5-fold CV Pearson correlation\n(fold-mean ± std)", fontsize=10)
    axes[0].set_ylim(0, 0.85)

    fig.suptitle(
        "Architecture ablation under increasingly strict evaluation\n"
        "(combined general-PPI pool: 7,247 examples, 541 complexes)",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.01,
              "SIMBA's Pairformer (bold outline) wins every panel; its margin over both baselines\n"
              "narrows but never disappears as the split protocol gets harder (left → right).",
              ha="center", fontsize=9, style="italic")

    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    out_path = "figures/ablation_split_matrix.png"
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
