"""Heatmap figures for the 5 named-complex interpretability case studies
(`interpretability_case_studies.py`), same layout convention as
`make_ig_figure_1cse.py`.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASES = ["ras_raf", "trypsin_bpti", "colicin_im9", "fab2d1_1918ha", "hyhel10_lysozyme"]


def panel(ax_attr, ax_dist, attr_norm, dist, cutoff, threshold, title):
    attr_norm = np.asarray(attr_norm)
    dist = np.asarray(dist)
    is_interface = dist <= cutoff

    ax_attr.imshow(attr_norm[None, :], aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax_attr.set_yticks([]); ax_attr.set_xticks([])
    ax_attr.set_title(f"{title}: IG attribution (normalized; threshold={threshold})", fontsize=9.5)
    for i, (a, iface) in enumerate(zip(attr_norm, is_interface)):
        if a > threshold:
            ax_attr.plot(i, 0.42, marker="v", color="lime" if iface else "red", markersize=5)
        if iface:
            ax_attr.plot(i, -0.42, marker="^", color="cyan", markersize=4)

    colors = ["#2ca02c" if iface else "#888888" for iface in is_interface]
    ax_dist.bar(np.arange(len(dist)), dist, color=colors, width=1.0)
    ax_dist.axhline(cutoff, color="red", linestyle="--", linewidth=0.7)
    ax_dist.set_ylabel("CA dist. (Å)", fontsize=7.5)
    ax_dist.set_xlim(-0.5, len(dist) - 0.5)
    ax_dist.tick_params(labelsize=6.5)


def make_figure(case_name):
    with open(f"figures/ig_{case_name}_attribution.json") as f:
        d = json.load(f)

    fig, axes = plt.subplots(4, 1, figsize=(11, 6.5), height_ratios=[1, 1.6, 1, 1.6])
    self_label = f"Self ({'+'.join(d['self_chains'])})"
    partner_label = f"Partner ({'+'.join(d['partner_chains'])})"
    panel(axes[0], axes[1], d["self_attribution_norm"], d["self_distance_to_partner"],
          d["interface_cutoff_angstrom"], d["attribution_threshold"], self_label)
    panel(axes[2], axes[3], d["partner_attribution_norm"], d["partner_distance_to_self"],
          d["interface_cutoff_angstrom"], d["attribution_threshold"], partner_label)

    fig.suptitle(
        f"{d['label']} ({d['complex_id']}, wildtype row WAS in training) — "
        f"predicted ΔG = {d['predicted_dG']:.2f} kcal/mol\n"
        f"corr(dist,attr): self P={d['self_corr_pearson']:.2f} S={d['self_corr_spearman']:.2f}  |  "
        f"partner P={d['partner_corr_pearson']:.2f} S={d['partner_corr_spearman']:.2f}\n"
        f"precision@attr>{d['attribution_threshold']}: self "
        f"{'%.2f' % d['self_precision_above_threshold'] if d['self_precision_above_threshold'] is not None else 'n/a'} "
        f"(random={d['self_baseline']:.2f}, n_above={d['self_n_above_threshold']})  |  partner "
        f"{'%.2f' % d['partner_precision_above_threshold'] if d['partner_precision_above_threshold'] is not None else 'n/a'} "
        f"(random={d['partner_baseline']:.2f}, n_above={d['partner_n_above_threshold']})",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    out_path = f"figures/ig_{case_name}_heatmap.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    for case_name in CASES:
        make_figure(case_name)


if __name__ == "__main__":
    main()
