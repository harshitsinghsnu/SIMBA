"""Figure for the Pairformer-depth ablation (paper Sec. 3.4): single-seed
runs on the prott_test1 development set with 1, 2, and 3 stacked
row/column-attention layers, motivating the choice of a single Pairformer
-lite layer everywhere else in the paper. Reads the per-example predictions
saved by `eval_affinity.py --out ...` and reports Pearson, Spearman, and
RMSE (the paper text only quotes Pearson; Spearman/RMSE included here for
the same reason every other panel in this project reports all three).
Single seed, so no error bars -- these are point estimates, not CIs.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import bootstrap_metrics_ci  # noqa: E402
from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

FILES = {
    1: "pairformer_depth_1.json",
    2: "pairformer_depth_2.json",
    3: "pairformer_depth_3.json",
}
N_BOOT = 1000

plt.rcParams.update(RCPARAMS)


def main():
    results = {}
    for depth, path in FILES.items():
        with open(path) as f:
            d = json.load(f)
        p = d["predictions"]
        ci = bootstrap_metrics_ci(p["true"], p["pred"], n_boot=N_BOOT)
        results[depth] = ci
        print(f"  depth={depth}: n={ci['n']} Pearson={ci['pearson']:.3f} "
              f"[{ci['pearson_ci_lo']:.3f},{ci['pearson_ci_hi']:.3f}]  "
              f"Spearman={ci['spearman']:.3f}  RMSE={ci['rmse']:.3f}")

    depths = sorted(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2))
    metric_specs = [("pearson", "Pearson correlation", "{:.3f}"),
                    ("spearman", "Spearman correlation", "{:.3f}"),
                    ("rmse", "RMSE, dG (kcal/mol)", "{:.2f}")]
    colors = sequential_blues(len(depths))
    for ax, (key, ylabel, fmt) in zip(axes, metric_specs):
        vals = [results[d][key] for d in depths]
        lo = [results[d][key] - results[d][f"{key}_ci_lo"] for d in depths]
        hi = [results[d][f"{key}_ci_hi"] - results[d][key] for d in depths]
        bars = ax.bar([str(d) for d in depths], vals, 0.5, yerr=[lo, hi], capsize=5,
                      color=colors, edgecolor="black", linewidth=0.8)
        for b, v, h in zip(bars, vals, hi):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + h + 0.015, fmt.format(v),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_xlabel("Pairformer-lite layers")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = "figures/pairformer_depth_ablation.png"
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
