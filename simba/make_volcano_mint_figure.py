"""Volcano-plot view (cnsplots-style: https://github.com/faridrashidi/
cnsplots) of the SIMBA-vs-MINT paired permutation test (Section 5.3): each
point is one metric (Pearson, Spearman, RMSE), x-axis is SIMBA's advantage
(observed gap), y-axis is -log10(p) from the paired permutation test (5,000
resamples). All three land far into the "large effect, high significance"
region -- a compact, single-glance summary of how solid this comparison is,
using a plot type not otherwise in the paper. Recomputes the permutation
test from the already-saved raw predictions (no retraining).
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import permutation_test_paired_metric  # noqa: E402
from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

plt.rcParams.update(RCPARAMS)


def main():
    simba_p = json.load(open("simba_vs_mint_skempi_split_results.json"))["mint_split"]["predictions"]
    mint_p = json.load(open("mint_own_method_skempi_results.json"))["predictions"]

    metrics = [("pearson", "Pearson"), ("spearman", "Spearman"), ("rmse", "RMSE")]
    colors = sequential_blues(len(metrics))
    # Pearson and Spearman's observed gaps land very close together (~0.145-0.150),
    # so annotations are staggered at different vertical offsets (not all
    # centered on the point) to keep them legible instead of overlapping.
    label_y_offsets = [28, -28, 0]

    fig, ax = plt.subplots(figsize=(9, 8))
    points = []
    for (metric, label), color in zip(metrics, colors):
        pt = permutation_test_paired_metric(simba_p["true"], simba_p["pred"], mint_p["pred"],
                                             metric=metric, n_perm=5000)
        gap = pt["observed_gap"]
        p = max(pt["p_value"], 1e-4)  # avoid log10(0) when p rounds to 0.0000 at 5000 resamples
        neg_log10_p = -np.log10(p)
        points.append((gap, neg_log10_p, label, color))

    for (gap, neg_log10_p, label, color), y_off in zip(points, label_y_offsets):
        ax.scatter(gap, neg_log10_p, s=260, color=color, edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(f"{label}: gap={gap:+.3f}, p<0.001", (gap, neg_log10_p),
                    xytext=(18, y_off), textcoords="offset points", fontsize=13.5, va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8,
                                     shrinkA=8, shrinkB=3))

    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.axhline(-np.log10(0.05), color="#d62728", linewidth=1, linestyle=":",
               label="p = 0.05 threshold")
    ax.set_xlabel("SIMBA advantage over MINT (observed gap)")
    ax.set_ylabel(r"$-\log_{10}(p)$, paired permutation test")
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(1.0, 4.9)
    xmax = max(gap for gap, _, _, _ in points)
    ax.set_xlim(-0.1, xmax + 0.22)

    out_path = "figures/volcano_mint_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    fig.savefig("figures/volcano_mint_comparison.pdf", bbox_inches="tight")
    print("saved figures/volcano_mint_comparison.pdf")


if __name__ == "__main__":
    main()
