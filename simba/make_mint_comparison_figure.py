"""Figure for MODEL_REPORT.md S3.21: SIMBA vs. MINT's own reported method."""
from __future__ import annotations
import json
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simba.bootstrap_utils import (
    bootstrap_metrics_ci, bootstrap_per_complex_metrics_ci, permutation_test_paired_metric
)
from simba.figure_style import RCPARAMS

N_BOOT = 1000
N_PERM = 5000

plt.rcParams.update(RCPARAMS)

# EXACT CHANGE: Beautiful custom contrasting blues
_COLOR_MINT = "#90e0ef"  # Light Sky Blue
_COLOR_SIMBA = "#023e8a" # Strong Navy Blue

def load_predictions():
    with open("simba_vs_mint_skempi_split_results.json") as f:
        simba_p = json.load(f)["mint_split"]["predictions"]
    with open("mint_own_method_skempi_results.json") as f:
        mint_p = json.load(f)["predictions"]
    return simba_p, mint_p

def main():
    simba_p, mint_p = load_predictions()
    simba_true, simba_pred, simba_ids = simba_p["true"], simba_p["pred"], simba_p["ids"]
    mint_true, mint_pred, mint_ids = mint_p["true"], mint_p["pred"], mint_p["ids"]

    simba_pool = bootstrap_metrics_ci(simba_true, simba_pred, n_boot=N_BOOT)
    mint_pool = bootstrap_metrics_ci(mint_true, mint_pred, n_boot=N_BOOT)
    simba_struct = bootstrap_per_complex_metrics_ci(simba_ids, simba_true, simba_pred, min_n=10, n_boot=N_BOOT)
    mint_struct = bootstrap_per_complex_metrics_ci(mint_ids, mint_true, mint_pred, min_n=10, n_boot=N_BOOT)

    aligned = (len(simba_true) == len(mint_true) and np.allclose(simba_true, mint_true, atol=1e-6))
    perm_results = {}
    if aligned:
        for metric in ("pearson", "spearman", "rmse"):
            perm_results[metric] = permutation_test_paired_metric(
                simba_true, simba_pred, mint_pred, metric=metric, n_perm=N_PERM)

    groups = ["Overall\nPearson", "Overall\nSpearman", "Overall\nRMSE (kcal/mol)",
              "Per-Structure\nPearson", "Per-Structure\nSpearman"]
              
    simba_vals = [simba_pool["pearson"], simba_pool["spearman"], simba_pool["rmse"],
                  simba_struct["mean_pearson"], simba_struct["mean_spearman"]]
    simba_err = [
        [simba_pool["pearson"] - simba_pool["pearson_ci_lo"], simba_pool["spearman"] - simba_pool["spearman_ci_lo"],
         simba_pool["rmse"] - simba_pool["rmse_ci_lo"],
         simba_struct["mean_pearson"] - simba_struct["mean_pearson_ci_lo"],
         simba_struct["mean_spearman"] - simba_struct["mean_spearman_ci_lo"]],
        [simba_pool["pearson_ci_hi"] - simba_pool["pearson"], simba_pool["spearman_ci_hi"] - simba_pool["spearman"],
         simba_pool["rmse_ci_hi"] - simba_pool["rmse"],
         simba_struct["mean_pearson_ci_hi"] - simba_struct["mean_pearson"],
         simba_struct["mean_spearman_ci_hi"] - simba_struct["mean_spearman"]],
    ]
    
    mint_vals = [mint_pool["pearson"], mint_pool["spearman"], mint_pool["rmse"],
                 mint_struct["mean_pearson"], mint_struct["mean_spearman"]]
    mint_err = [
        [mint_pool["pearson"] - mint_pool["pearson_ci_lo"], mint_pool["spearman"] - mint_pool["spearman_ci_lo"],
         mint_pool["rmse"] - mint_pool["rmse_ci_lo"],
         mint_struct["mean_pearson"] - mint_struct["mean_pearson_ci_lo"],
         mint_struct["mean_spearman"] - mint_struct["mean_spearman_ci_lo"]],
        [mint_pool["pearson_ci_hi"] - mint_pool["pearson"], mint_pool["spearman_ci_hi"] - mint_pool["spearman"],
         mint_pool["rmse_ci_hi"] - mint_pool["rmse"],
         mint_struct["mean_pearson_ci_hi"] - mint_struct["mean_pearson"],
         mint_struct["mean_spearman_ci_hi"] - mint_struct["mean_spearman"]],
    ]

    x = np.arange(len(groups))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2 - 0.02, simba_vals, width, yerr=simba_err, capsize=4,
          color=_COLOR_SIMBA, edgecolor="black", linewidth=1.1,
          label="SIMBA (pairformer, ESM2-650M)", error_kw=dict(linewidth=1.2))
    ax.bar(x + width / 2 + 0.02, mint_vals, width, yerr=mint_err, capsize=4,
          color=_COLOR_MINT, edgecolor="black", linewidth=0.9,
          label="MINT (their own method: embeddings + Ridge)", error_kw=dict(linewidth=1.2))

    if aligned:
        for i, metric in enumerate(("pearson", "spearman", "rmse")):
            p = perm_results[metric]["p_value"]
            p_label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            y = max(simba_vals[i] + simba_err[1][i], mint_vals[i] + mint_err[1][i]) + 0.28
            ax.text(x[i], y, p_label, ha="center", fontsize=11, style="italic", color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Correlation  /  RMSE (kcal/mol)")
    ax.set_ylim(0, max(max(simba_vals), max(mint_vals)) * 1.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = "figures/simba_vs_mint_skempi.png"
    fig.savefig(out_path, dpi=300)

if __name__ == "__main__":
    main()