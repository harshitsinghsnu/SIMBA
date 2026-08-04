"""Heatmap/profile figure for the 1CSE Integrated Gradients case study
(`interpretability_1cse.py`) -- same layout as the 1BRS figure, plus a
marker for the fixed attribution > 0.4 threshold requested alongside the
top-K/top-15% convention already used for 1BRS.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IN_JSON = "figures/ig_1cse_attribution.json"
OUT_PNG = "figures/ig_1cse_heatmap.png"


def panel(ax_attr, ax_dist, attr_norm, dist, cutoff, threshold, title):
    attr_norm = np.asarray(attr_norm)
    dist = np.asarray(dist)
    is_interface = dist <= cutoff

    ax_attr.imshow(attr_norm[None, :], aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax_attr.set_yticks([])
    ax_attr.set_xticks([])
    ax_attr.set_title(f"{title}: IG attribution (normalized 0-1; threshold={threshold})", fontsize=10)
    for i, (a, iface) in enumerate(zip(attr_norm, is_interface)):
        if a > threshold:
            ax_attr.plot(i, 0.42, marker="v", color="lime" if iface else "red", markersize=6)
        if iface:
            ax_attr.plot(i, -0.42, marker="^", color="cyan", markersize=5)

    colors = ["#2ca02c" if iface else "#888888" for iface in is_interface]
    ax_dist.bar(np.arange(len(dist)), dist, color=colors, width=1.0)
    ax_dist.axhline(cutoff, color="red", linestyle="--", linewidth=0.8, label=f"interface cutoff ({cutoff:.0f}Å)")
    ax_dist.set_ylabel("CA dist. to\npartner (Å)", fontsize=8)
    ax_dist.set_xlabel("residue index")
    ax_dist.set_xlim(-0.5, len(dist) - 0.5)
    ax_dist.legend(fontsize=7, loc="upper right")
    ax_dist.tick_params(labelsize=7)


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), height_ratios=[1, 2, 1, 2])
    panel(axes[0], axes[1], d["self_attribution_norm"], d["self_distance_to_partner"],
          d["interface_cutoff_angstrom"], d["attribution_threshold"], "Eglin c (self, in training)")
    panel(axes[2], axes[3], d["partner_attribution_norm"], d["partner_distance_to_self"],
          d["interface_cutoff_angstrom"], d["attribution_threshold"], "Subtilisin (partner)")

    fig.suptitle(
        f"1CSE (eglin c-subtilisin, wildtype row WAS in the training split) — "
        f"predicted ΔG = {d['predicted_dG']:.2f} kcal/mol\n"
        f"corr(distance, attribution): eglin c Pearson={d['self_corr_pearson']:.2f} "
        f"Spearman={d['self_corr_spearman']:.2f}  |  "
        f"subtilisin Pearson={d['partner_corr_pearson']:.2f} Spearman={d['partner_corr_spearman']:.2f}\n"
        f"attribution>{d['attribution_threshold']}: eglin c {d['self_n_above_threshold']}/63 residues, "
        f"precision={d['self_precision_above_threshold']:.2f} (random={d['self_baseline']:.2f})  |  "
        f"subtilisin {d['partner_n_above_threshold']}/274, "
        f"precision={d['partner_precision_above_threshold']:.2f} (random={d['partner_baseline']:.2f})\n"
        f"markers: ▽ green=above-threshold & true interface, ▽ red=above-threshold but not interface, △ cyan=true interface",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(OUT_PNG, dpi=200)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
