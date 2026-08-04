"""Figure for the SKEMPI2-only pool ablation (paper Sec. 6.2, closing
paragraph): the same 3-architecture comparison as Table 3/Panel 1, but on
the smaller 7,046-example SKEMPI2-only pool (no benchmark46/ATLAS TCR:pMHC
rows) under the complex (random) split only -- confirms the same
architecture ordering holds on a pool without the combined-pool additions.
Reads each run's saved per-example predictions and reports Pearson,
Spearman, and RMSE with 95% bootstrap CIs.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import bootstrap_metrics_ci  # noqa: E402
from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

_KEYS = ["concat_mlp", "cross_attn_mlp", "pairformer"]
_LABELS = ["Concat + MLP", "Cross-attn + MLP", "SIMBA (Pairformer)"]
_BLUES = sequential_blues(len(_KEYS))
FILES = {k: (label, color) for k, label, color in zip(_KEYS, _LABELS, _BLUES)}
N_BOOT = 1000

plt.rcParams.update(RCPARAMS)


def main():
    results = {}
    for model_key in FILES:
        path = f"skempi2_5fold_{model_key}.predictions.json"
        with open(path) as f:
            recs = json.load(f)
        true = [r["dG_true"] for r in recs]
        pred = [r["dG_pred"] for r in recs]
        ci = bootstrap_metrics_ci(true, pred, n_boot=N_BOOT)
        results[model_key] = ci
        print(f"  {model_key}: n={ci['n']} Pearson={ci['pearson']:.3f} "
              f"[{ci['pearson_ci_lo']:.3f},{ci['pearson_ci_hi']:.3f}]  "
              f"Spearman={ci['spearman']:.3f}  RMSE={ci['rmse']:.3f}")

    labels = list(FILES.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2))
    metric_specs = [("pearson", "Pearson correlation", "{:.3f}"),
                    ("spearman", "Spearman correlation", "{:.3f}"),
                    ("rmse", "RMSE, dG (kcal/mol)", "{:.2f}")]
    for ax, (key, ylabel, fmt) in zip(axes, metric_specs):
        vals = [results[l][key] for l in labels]
        lo = [results[l][key] - results[l][f"{key}_ci_lo"] for l in labels]
        hi = [results[l][f"{key}_ci_hi"] - results[l][key] for l in labels]
        colors = [FILES[l][1] for l in labels]
        display_labels = [FILES[l][0] for l in labels]
        bars = ax.bar(display_labels, vals, 0.5, yerr=[lo, hi], capsize=5,
                      color=colors, edgecolor="black", linewidth=0.8)
        for b, v, h in zip(bars, vals, hi):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + h + 0.015, fmt.format(v),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=10.5, rotation=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = "figures/skempi2_only_ablation.png"
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
