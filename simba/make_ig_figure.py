"""Heatmap/profile figure for the 1BRS Integrated Gradients case study
(`interpretability_1brs.py`'s output JSON): per-residue attribution score
alongside real CA-distance-to-partner-interface, for both barnase (self)
and barstar (partner), so the visual correspondence between "what the model
attends to" and "what is structurally at the interface" is inspectable
directly, not just summarized as a single correlation number.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IN_JSON = "figures/ig_1brs_attribution.json"
OUT_PNG = "figures/ig_1brs_heatmap.png"


def panel(ax_attr, ax_dist, seq, attr, dist, cutoff, title):
    attr = np.asarray(attr)
    dist = np.asarray(dist)
    norm_attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-9)
    is_interface = dist <= cutoff

    ax_attr.imshow(norm_attr[None, :], aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax_attr.set_yticks([])
    ax_attr.set_xticks([])
    ax_attr.set_title(f"{title}: Integrated Gradients attribution (per residue)", fontsize=10)
    for i, iface in enumerate(is_interface):
        if iface:
            ax_attr.plot(i, 0, marker="|", color="cyan", markersize=8, markeredgewidth=1.2)

    colors = ["#2ca02c" if iface else "#888888" for iface in is_interface]
    ax_dist.bar(np.arange(len(dist)), dist, color=colors, width=1.0)
    ax_dist.axhline(cutoff, color="red", linestyle="--", linewidth=0.8,
                     label=f"interface cutoff ({cutoff:.0f}Å)")
    ax_dist.set_ylabel("CA dist. to\npartner (Å)", fontsize=8)
    ax_dist.set_xlabel("residue index")
    ax_dist.set_xlim(-0.5, len(dist) - 0.5)
    ax_dist.legend(fontsize=7, loc="upper right")
    ax_dist.tick_params(labelsize=7)


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), height_ratios=[1, 2, 1, 2])
    panel(axes[0], axes[1], d["self_sequence"], d["self_attribution"],
          d["self_distance_to_partner"], d["interface_cutoff_angstrom"],
          "Barnase (self chain)")
    panel(axes[2], axes[3], d["partner_sequence"], d["partner_attribution"],
          d["partner_distance_to_self"], d["interface_cutoff_angstrom"],
          "Barstar (partner chain)")

    fig.suptitle(
        f"1BRS (barnase-barstar, held out of training) — predicted ΔG = "
        f"{d['predicted_dG']:.2f} kcal/mol\n"
        f"corr(distance, attribution): barnase Pearson={d['self_corr_pearson']:.2f} "
        f"Spearman={d['self_corr_spearman']:.2f}  |  "
        f"barstar Pearson={d['partner_corr_pearson']:.2f} Spearman={d['partner_corr_spearman']:.2f}\n"
        f"precision@top15%: barnase {d['self_precision_at_k']:.2f} "
        f"(random={d['self_precision_baseline']:.2f}), "
        f"barstar {d['partner_precision_at_k']:.2f} (random={d['partner_precision_baseline']:.2f})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_PNG, dpi=200)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
