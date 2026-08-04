"""One figure for the biological error-analysis dimensions (§3.13):
distance-to-interface vs. error, conservation vs. error, flexibility
(B-factor) vs. error, and the full 5-category Core/Rim/Support/Surface
/Interior MAE breakdown. Reads `biological_analysis_results.json` (produced
by `simba/eval_biological_analysis.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CATEGORY_ORDER = ["COR", "RIM", "SUP", "SUR", "INT"]
CATEGORY_LABELS = {"COR": "Core", "RIM": "Rim", "SUP": "Support", "SUR": "Surface", "INT": "Interior"}
CATEGORY_COLORS = {"COR": "#c0392b", "RIM": "#e67e22", "SUP": "#f1c40f", "SUR": "#27ae60", "INT": "#2980b9"}


def main():
    with open(ROOT / "biological_analysis_results.json") as f:
        d = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Core/Rim/Support/Surface/Interior MAE bar chart
    ax = axes[0, 0]
    cats = [c for c in CATEGORY_ORDER if c in d["by_interface_location_full"]]
    maes = [d["by_interface_location_full"][c]["mae"] for c in cats]
    ns = [d["by_interface_location_full"][c]["n"] for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]
    bars = ax.bar([CATEGORY_LABELS[c] for c in cats], maes, color=colors)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}",
                ha="center", fontsize=8)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("(a) Full structural classification (SKEMPI2 iMutation_Location)\nCore = interface core, "
                  "Rim = interface periphery, Support = near-interface,\nSurface/Interior = non-interface")

    def panel_scatter(ax, key, xlabel, title, color):
        entry = d[key]
        bins = entry["bins"]
        x = [b[f"mean_{key.replace('_vs_error','')}"] for b in bins]
        y = [b["mean_abs_error"] for b in bins]
        ax.plot(x, y, "-o", color=color, markersize=8, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean |error| (kcal/mol)")
        r, s = entry["pearson"], entry["spearman"]
        ax.set_title(f"{title}\nPearson={r:.3f}, Spearman={s:.3f} (n={entry['n']})")

    panel_scatter(axes[0, 1], "distance_to_partner_vs_error",
                  "CA distance to nearest partner-chain residue (Å)",
                  "(b) Distance from interface vs. error\n(binned means, quartiles)", "#8e44ad")
    panel_scatter(axes[1, 0], "conservation_vs_error",
                  "ESM2 wildtype-marginal log P(wt aa) [conservation proxy]",
                  "(c) Conservation vs. error\n(binned means, quartiles)", "#16a085")
    panel_scatter(axes[1, 1], "bfactor_vs_error",
                  "crystal structure CA B-factor [flexibility proxy]",
                  "(d) Flexibility vs. error\n(binned means, quartiles)", "#d35400")

    fig.suptitle("SIMBA (RDE-PPI): does prediction error correlate with structural/evolutionary "
                  "properties of the mutated residue?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = ROOT / "figures" / "biological_error_analysis.png"
    fig.savefig(out_path, dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
