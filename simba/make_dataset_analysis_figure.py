"""New Panel: dataset analysis (binding-affinity distributions, dataset
sizes, sequence-length spread) for the Datasets section. Loads directly
from the same builders every training/eval script uses -- no separate
data snapshot to go stale.

  (A) Histogram of absolute binding free energy (dG) across the combined
      general-PPI ablation pool.
  (B) Histogram of mutation-induced binding-energy change (ddG) across
      the RDE-PPI mutation-effect benchmark.
  (C) Dataset sizes (number of examples) used across the paper, log scale
      given the ~150x spread between the smallest and largest.
  (D) dG distribution by source within the combined pool (box plot),
      showing the three constituent datasets are not interchangeable in
      scale even though they're pooled for training.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba import env  # noqa: E402

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.data.dataset import load_all_mutations, load_rde_skempi  # noqa: E402
from simba.figure_style import PANEL_LABEL_KW, RCPARAMS, sequential_blues  # noqa: E402

plt.rcParams.update(RCPARAMS)


def main():
    combined = build_combined_examples()
    rde = load_rde_skempi()

    dG_vals = np.array([e.dG for e in combined])
    ddG_vals = np.array([e.ddG for e in rde if e.ddG is not None])

    by_source: dict[str, list[float]] = {}
    for e in combined:
        by_source.setdefault(e.meta["dataset"], []).append(e.dG)

    # Dataset sizes referenced throughout the paper. Combined pool,
    # SKEMPI2-only pool, and RDE-PPI are counted live from the same
    # loaders used everywhere else; AbDesign, benchmark46, and the eLife
    # Test 1 reproduction are reported as their published row/complex
    # counts (Sec. 4.2/5.1) rather than re-derived here, since
    # `load_all_mutations()` returns a broader combined training+eval
    # pool (SKEMPI2+AB-Bind+AbDesign together, 8,878 rows) rather than
    # AbDesign's own 1,303-row benchmark specifically.
    sizes = {
        "SKEMPI2-only pool": len([e for e in combined if e.meta["dataset"] == "skempi2"]),
        "Combined pool": len(combined),
        "RDE-PPI\n(3-fold split)": len(rde),
        "AbDesign": 1303,
        "Kastritis-lineage\n(benchmark46)": 45,
        "eLife Test 1\nreproduction": 66,
    }

    colors_5 = sequential_blues(5)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28, top=0.96, bottom=0.07, left=0.07, right=0.98)

    # --- A: dG histogram, combined pool ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(dG_vals, bins=40, color=colors_5[3], edgecolor="black", linewidth=0.5)
    ax_a.set_xlabel("Binding free energy, dG (kcal/mol)")
    ax_a.set_ylabel("Count")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.set_axisbelow(True)

    # --- B: ddG histogram, RDE-PPI ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(ddG_vals, bins=40, color=colors_5[1], edgecolor="black", linewidth=0.5)
    ax_b.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax_b.set_xlabel("Mutation-induced binding-energy change, ddG (kcal/mol)")
    ax_b.set_ylabel("Count")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.set_axisbelow(True)

    # --- C: dataset sizes, log scale ---
    ax_c = fig.add_subplot(gs[1, 0])
    labels = list(sizes.keys())
    vals = [sizes[k] for k in labels]
    bar_colors = sequential_blues(len(labels))
    bars = ax_c.barh(labels, vals, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Number of examples (log scale)")
    for b, v in zip(bars, vals):
        ax_c.text(b.get_width() * 1.08, b.get_y() + b.get_height() / 2, f"{v:,}",
                  va="center", fontsize=14)
    ax_c.grid(axis="x", alpha=0.25, which="both")
    ax_c.set_axisbelow(True)

    # --- D: dG by source, box plot ---
    ax_d = fig.add_subplot(gs[1, 1])
    source_labels = {"skempi2": "SKEMPI2", "benchmark46": "Kastritis\n(benchmark46)", "atlas": "ATLAS\nTCR:pMHC"}
    order = [k for k in ("skempi2", "benchmark46", "atlas") if k in by_source]
    data = [by_source[k] for k in order]
    bp = ax_d.boxplot(data, labels=[source_labels[k] for k in order], patch_artist=True,
                      widths=0.55, medianprops=dict(color="black", linewidth=1.6))
    for patch, color in zip(bp["boxes"], sequential_blues(len(order))):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    ax_d.set_ylabel("Binding free energy, dG (kcal/mol)")
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.set_axisbelow(True)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    out_path = "figures/panel_dataset_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    pdf_path = "figures/panel_dataset_analysis.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"saved {pdf_path}")

    print(f"\ncombined pool: n={len(combined)}, dG range [{dG_vals.min():.2f}, {dG_vals.max():.2f}]")
    print(f"RDE-PPI: n={len(rde)}, ddG range [{ddG_vals.min():.2f}, {ddG_vals.max():.2f}]")
    for k, v in sizes.items():
        print(f"  {k.splitlines()[0]}: {v}")


if __name__ == "__main__":
    main()
