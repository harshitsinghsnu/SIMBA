"""Heatmap alternative view of the architecture-ablation matrix (Table 3):
architecture x split-protocol grid, annotated with pooled Pearson, in the
cnsplots (https://github.com/faridrashidi/cnsplots) clustered-heatmap style
-- a single glance showing SIMBA (bottom row) is the darkest cell in every
column, i.e. wins under every split protocol tested, not just on average.
Reads directly from the already-saved skempi2_5fold_combined_*.json files
(no retraining).
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.figure_style import PANEL_LABEL_KW, RCPARAMS  # noqa: E402

plt.rcParams.update(RCPARAMS)

SPLITS = [("Complex\n(random)", ""), ("Sequence-similarity\ncluster", "seqsim_"),
          ("Cold-target\n(held-out partner)", "coldtarget_")]
MODELS = [("concat_mlp", "Concat + MLP"), ("cross_attn_mlp", "Cross-attn + MLP"),
          ("pairformer", "SIMBA\n(Pairformer)")]


def load_grid(metric="fold_pcc_mean"):
    grid = np.zeros((len(MODELS), len(SPLITS)))
    for gi, (_, suffix) in enumerate(SPLITS):
        for mi, (model_key, _) in enumerate(MODELS):
            with open(f"skempi2_5fold_combined_{suffix}{model_key}.json") as f:
                grid[mi, gi] = json.load(f)[metric]
    return grid


def main():
    grid = load_grid("fold_pcc_mean")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(grid, cmap="Blues", vmin=0.45, vmax=0.75, aspect="auto")

    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([label for label, _ in SPLITS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([label for _, label in MODELS])

    for mi in range(len(MODELS)):
        for gi in range(len(SPLITS)):
            val = grid[mi, gi]
            is_best = val == grid[:, gi].max()
            color = "white" if val > 0.62 else "black"
            ax.text(gi, mi, f"{val:.3f}" + ("\n(best)" if is_best else ""), ha="center", va="center",
                    fontsize=15, color=color, fontweight="bold" if is_best else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Pearson correlation (5-fold CV)")

    out_path = "figures/alt_heatmap_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    fig.savefig("figures/alt_heatmap_ablation.pdf", bbox_inches="tight")
    print("saved figures/alt_heatmap_ablation.pdf")


if __name__ == "__main__":
    main()
