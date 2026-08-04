"""Zero-shot vs. few-shot (10% support) on genuinely novel complexes, from
`eval_novel_generalization_fewshot_scaling.py`'s output -- Pearson,
Spearman, and RMSE, with 95% bootstrap CIs computed from the saved raw
per-example predictions (79-row held-out test set). Embedded as Panel 2D
(moved out of the old Panel 3, which now holds only the two
interpretability sub-panels).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import bootstrap_metrics_ci  # noqa: E402
from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

IN_JSON = "novel_generalization_fewshot_scaling_results.json"
OUT_PNG = "figures/fewshot_scaling_curve.png"
N_BOOT = 1000

plt.rcParams.update(RCPARAMS)


def main():
    with open(IN_JSON) as f:
        d = json.load(f)
    p = d["predictions"]
    true = p["true"]
    zero_pred = p["zeroshot_pred"]
    few_pred = p["fewshot_pred_by_fraction"]["0.10"]
    n_support = [row["n_support_used"] for row in d["curve"] if row["fraction"] == 0.10][0]

    zero_ci = bootstrap_metrics_ci(true, zero_pred, n_boot=N_BOOT)
    few_ci = bootstrap_metrics_ci(true, few_pred, n_boot=N_BOOT)
    print(f"[fewshot_scale_fig] zero-shot  n={zero_ci['n']} P={zero_ci['pearson']:.3f} "
          f"S={zero_ci['spearman']:.3f} RMSE={zero_ci['rmse']:.3f}")
    print(f"[fewshot_scale_fig] few-shot   n={few_ci['n']} P={few_ci['pearson']:.3f} "
          f"S={few_ci['spearman']:.3f} RMSE={few_ci['rmse']:.3f}")

    groups = ["Zero-shot\n(no fine-tuning)", f"Few-shot, 10% support\n({n_support} examples, {d['epochs']} epochs)"]
    colors = sequential_blues(2)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8))
    metric_specs = [
        ("pearson", "Pearson correlation", "{:.3f}"),
        ("spearman", "Spearman correlation", "{:.3f}"),
        ("rmse", "RMSE, ddG (kcal/mol)", "{:.2f}"),
    ]
    for ax, (key, label, fmt) in zip(axes, metric_specs):
        vals = [zero_ci[key], few_ci[key]]
        lo = [zero_ci[key] - zero_ci[f"{key}_ci_lo"], few_ci[key] - few_ci[f"{key}_ci_lo"]]
        hi = [zero_ci[f"{key}_ci_hi"] - zero_ci[key], few_ci[f"{key}_ci_hi"] - few_ci[key]]
        ax.bar(groups, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.5,
              yerr=[lo, hi], capsize=5, error_kw=dict(linewidth=1.2))
        ax.set_ylabel(label)
        top = max(zero_ci[f"{key}_ci_hi"], few_ci[f"{key}_ci_hi"])
        ax.set_ylim(0, top * 1.3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    print(f"[fewshot_scale_fig] saved {OUT_PNG}")


if __name__ == "__main__":
    main()
