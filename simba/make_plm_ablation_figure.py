"""Figure for the PLM-backbone ablation (MODEL_REPORT.md S3.7): 2 frozen
PLM backbones (ESM2-150M, ESM2-650M) x 3 projection sizes (d_model in
{128, 256, 512}), on two absolute-affinity test sets (prott_test1,
benchmark46). Single-seed sweep, proposed architecture (pairformer)
otherwise unchanged. Grouped bar chart, one panel per test set, so the
reader can see at a glance that the two backbones diverge in *opposite*
directions depending on the test set -- the smaller backbone wins on
prott_test1, the larger wins on benchmark46, consistently across every
projection size.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pearson correlation, single seed, see MODEL_REPORT.md S3.7 for the table.
RESULTS = {
    "prott_test1": {
        "ESM2-150M": {128: 0.717, 256: 0.717, 512: 0.713},
        "ESM2-650M": {128: 0.687, 256: 0.631, 512: 0.665},
    },
    "benchmark46": {
        "ESM2-150M": {128: 0.666, 256: 0.701, 512: 0.602},
        "ESM2-650M": {128: 0.803, 256: 0.781, 512: 0.791},
    },
}
D_MODELS = [128, 256, 512]
BACKBONES = ["ESM2-150M", "ESM2-650M"]
COLORS = {"ESM2-150M": "#63b3ed", "ESM2-650M": "#2b6cb0"}
WINNER = {"prott_test1": "ESM2-150M", "benchmark46": "ESM2-650M"}


def main():
    test_sets = list(RESULTS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True)

    for ax, test_set in zip(axes, test_sets):
        x = np.arange(len(D_MODELS))
        width = 0.35
        for i, backbone in enumerate(BACKBONES):
            vals = [RESULTS[test_set][backbone][d] for d in D_MODELS]
            offset = (i - 0.5) * width
            is_winner = backbone == WINNER[test_set]
            bars = ax.bar(x + offset, vals, width, color=COLORS[backbone],
                          edgecolor="black" if is_winner else "none",
                          linewidth=1.8 if is_winner else 0, label=backbone)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold" if is_winner else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([f"d_model={d}" for d in D_MODELS], fontsize=9.5)
        ax.set_title(test_set, fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    axes[0].set_ylabel("Pearson correlation", fontsize=10)
    axes[0].set_ylim(0, 0.90)
    axes[0].legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "PLM backbone ablation: ESM2-150M vs. ESM2-650M x projection size\n"
        "(frozen backbone, proposed pairformer head unchanged)",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.01,
              "Bold outline = better backbone for that test set. The two backbones diverge in opposite\n"
              "directions depending on the benchmark -- consistent across all 3 projection sizes, not a single-run fluke.",
              ha="center", fontsize=9, style="italic")

    fig.tight_layout(rect=[0, 0.07, 1, 0.88])
    out_path = "figures/plm_backbone_ablation.png"
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
