"""Alternative visualization of the architecture-ablation matrix (Table 3 /
Figure panel1 A-C): instead of a single mean+/-std bar per (architecture,
split) cell, shows the actual 5 per-fold Pearson values as a strip+violin
plot (cnsplots-style: https://github.com/faridrashidi/cnsplots), which
reveals fold-to-fold spread the bar chart's error bar collapses into one
number -- most visible on the sequence-similarity split, where one fold's
correlation collapses to ~0.22-0.28 for every architecture while the other
four stay above 0.5, a real bimodal pattern a symmetric error bar hides.

Reads directly from the already-saved skempi2_5fold_combined_*.json files
(no retraining).
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.figure_style import PANEL_LABEL_KW, RCPARAMS, sequential_blues  # noqa: E402

plt.rcParams.update(RCPARAMS)

SPLITS = [("Complex\n(random)", ""), ("Sequence-similarity\ncluster", "seqsim_"),
          ("Cold-target\n(held-out partner)", "coldtarget_")]
MODELS = [("concat_mlp", "Concat + MLP"), ("cross_attn_mlp", "Cross-attn + MLP"),
          ("pairformer", "SIMBA (Pairformer)")]
COLORS = dict(zip((m for m, _ in MODELS), sequential_blues(len(MODELS))))


def load_fold_pccs():
    out = {}
    for _, suffix in SPLITS:
        for model_key, _ in MODELS:
            with open(f"skempi2_5fold_combined_{suffix}{model_key}.json") as f:
                out[(suffix, model_key)] = json.load(f)["fold_pccs"]
    return out


def main():
    data = load_fold_pccs()

    fig, ax = plt.subplots(figsize=(15, 8))
    n_models = len(MODELS)
    group_width = 0.75
    width = group_width / n_models
    positions, violin_data, colors = [], [], []

    for gi, (_, suffix) in enumerate(SPLITS):
        for mi, (model_key, _) in enumerate(MODELS):
            pos = gi + (mi - (n_models - 1) / 2) * width
            positions.append(pos)
            violin_data.append(data[(suffix, model_key)])
            colors.append(COLORS[model_key])

    parts = ax.violinplot(violin_data, positions=positions, widths=width * 0.9,
                            showmeans=False, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, violin_data, colors):
        jitter = rng.uniform(-width * 0.12, width * 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, color=color, edgecolor="black",
                   linewidth=0.5, s=45, zorder=3)
        ax.plot([pos - width * 0.22, pos + width * 0.22], [np.mean(vals)] * 2,
                color="black", linewidth=2.0, zorder=4)

    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([label for label, _ in SPLITS])
    ax.set_ylabel("Pearson correlation (per fold)")
    ax.set_ylim(0.1, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor="black", alpha=0.7)
               for k, _ in MODELS]
    ax.legend(handles, [label for _, label in MODELS], loc="lower center", ncol=3,
              framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    out_path = "figures/alt_violin_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    fig.savefig("figures/alt_violin_ablation.pdf", bbox_inches="tight")
    print("saved figures/alt_violin_ablation.pdf")


if __name__ == "__main__":
    main()
