"""Ridge plot (joyplot) alternative to the dataset-analysis figure's box
plot (Figure panel_dataset_analysis, panel D): overlapping KDE curves per
source, offset vertically (cnsplots-style: https://github.com/faridrashidi/
cnsplots), showing the actual distribution *shape* the box plot's five
summary numbers compress away -- e.g. whether a source is unimodal,
skewed, or has a distinct second mode. Reads from the same real data
loaders as make_dataset_analysis_figure.py (no separate snapshot).
"""

from __future__ import annotations

import matplotlib
import numpy as np
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba import env  # noqa: E402

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

plt.rcParams.update(RCPARAMS)

SOURCE_LABELS = {"skempi2": "SKEMPI2", "benchmark46": "Kastritis-lineage\n(benchmark46)",
                  "atlas": "ATLAS\nTCR:pMHC"}


def main():
    combined = build_combined_examples()
    by_source: dict[str, list[float]] = {}
    for e in combined:
        by_source.setdefault(e.meta["dataset"], []).append(e.dG)

    order = [k for k in ("atlas", "benchmark46", "skempi2") if k in by_source]  # top-to-bottom in plot
    colors = sequential_blues(len(order))

    all_vals = np.concatenate([by_source[k] for k in order])
    x_grid = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 400)

    fig, ax = plt.subplots(figsize=(11, 7))
    overlap = 1.6  # vertical offset between ridges; >1 lets adjacent curves overlap slightly
    for i, (key, color) in enumerate(zip(order, colors)):
        vals = np.array(by_source[key])
        kde = gaussian_kde(vals)
        density = kde(x_grid)
        density = density / density.max()  # normalize peak height for comparable ridges
        baseline = i * overlap
        ax.fill_between(x_grid, baseline, baseline + density, color=color, alpha=0.85,
                        edgecolor="black", linewidth=0.8, zorder=len(order) - i)
        ax.plot(x_grid, baseline + density, color="black", linewidth=0.8, zorder=len(order) - i)
        ax.text(x_grid[0], baseline + 0.15, f"{SOURCE_LABELS[key]}  (n={len(vals)})",
                fontsize=14, va="bottom", ha="left")

    ax.set_yticks([])
    ax.set_xlabel("Binding free energy, dG (kcal/mol)")
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    out_path = "figures/ridge_dataset_by_source.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    fig.savefig("figures/ridge_dataset_by_source.pdf", bbox_inches="tight")
    print("saved figures/ridge_dataset_by_source.pdf")


if __name__ == "__main__":
    main()
