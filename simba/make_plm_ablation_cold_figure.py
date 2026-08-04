"""Figure for the PLM-backbone ablation on the cold_target split (Panel 2C)"""
from __future__ import annotations
import json
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simba.bootstrap_utils import bootstrap_metrics_ci
from simba.figure_style import RCPARAMS

N_BOOT = 1000

FILES = {
    "ESM2-650M\n(default)": "plm_ablation_cold_esm2_results.json",
    "MINT\n(PPI-pretrained)": "plm_ablation_cold_mint_results.json",
    "AbLang2\n(antibody-specific)": "plm_ablation_cold_ablang2_results.json",
}

# EXACT CHANGE: Hand-picked sequence of blues (Darkest for ESM2, mid for MINT, light for AbLang)
_BLUES = ["#005b96", "#00b4d8", "#caf0f8"] 
COLORS = dict(zip(FILES.keys(), _BLUES))

plt.rcParams.update(RCPARAMS)

def load_metrics():
    out = {}
    for label, path in FILES.items():
        with open(path) as f:
            d = json.load(f)
        if "predictions" in d:
            p = d["predictions"]
            m = bootstrap_metrics_ci(p["true"], p["pred"], n_boot=N_BOOT)
            m["has_ci"] = True
        else:
            m = {
                "pearson": d["fold_pcc_mean"], "pearson_ci_lo": d["fold_pcc_mean"] - d["fold_pcc_std"],
                "pearson_ci_hi": d["fold_pcc_mean"] + d["fold_pcc_std"],
                "spearman": float("nan"), "spearman_ci_lo": float("nan"), "spearman_ci_hi": float("nan"),
                "rmse": d["fold_mae_mean"], "rmse_ci_lo": d["fold_mae_mean"] - d["fold_mae_std"],
                "rmse_ci_hi": d["fold_mae_mean"] + d["fold_mae_std"],
                "has_ci": False, "n": d.get("n_folds_run", 5),
            }
        out[label] = m
    return out

def main():
    results = load_metrics()
    labels = list(FILES.keys())
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    metric_specs = [
        ("pearson", "Pearson correlation", "{:.3f}", 0.85),
        ("spearman", "Spearman correlation", "{:.3f}", 0.85),
        ("rmse", "RMSE / MAE, dG (kcal/mol)", "{:.2f}", 3.2),
    ]
    
    for ax, (key, ylabel, fmt, ylim) in zip(axes, metric_specs):
        vals = [results[l][key] for l in labels]
        lo = [results[l][key] - results[l][f"{key}_ci_lo"] for l in labels]
        hi = [results[l][f"{key}_ci_hi"] - results[l][key] for l in labels]
        
        colors = [COLORS[l] for l in labels]
        ax.bar(x, vals, 0.5, yerr=[lo, hi], capsize=5, color=colors,
              edgecolor="black", linewidth=0.8, error_kw=dict(linewidth=1.2))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(0, ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = "figures/plm_ablation_cold_split.png"
    fig.savefig(out_path, dpi=300)

if __name__ == "__main__":
    main()