"""Publication-style Panel 1: split-robustness matrix (A-C) combined with
the statistical summary of confirmed wins (D-E) -- merged into one panel
per request, since both are "how much do we trust these numbers" views of
the same underlying ablation/benchmark data.



No in-image titles/headings: panel identity is carried by the A-E letters
alone, full description lives in the LaTeX caption.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import bootstrap_metrics_ci, bootstrap_per_complex_metrics_ci  # noqa: E402
from simba.figure_style import PANEL_LABEL_KW, RCPARAMS, sequential_blues  # noqa: E402

N_BOOT = 1000

plt.rcParams.update(RCPARAMS)

# --- A-C: split-robustness matrix ---------------------------------------
SPLITS = [
    ("Complex\n(random)", ""),
    ("Sequence-similarity\ncluster", "seqsim_"),
    ("Cold-target\n(held-out partner)", "coldtarget_"),
]
MODELS = [("concat_mlp", "Concat + MLP"), ("cross_attn_mlp", "Cross-attn + MLP"),
          ("pairformer", "SIMBA (Pairformer)")]
_BLUES = sequential_blues(len(MODELS))
COLORS = {key: _BLUES[i] for i, (key, _) in enumerate(MODELS)}
METRICS = [("pearson", "Pearson correlation"), ("spearman", "Spearman correlation"), ("rmse", "RMSE, dG (kcal/mol)")]

_COLOR_SIMBA = sequential_blues(1)[0]


def load_split_robustness():
    out = {}
    for _, suffix in SPLITS:
        for model_key, _ in MODELS:
            path = f"skempi2_5fold_combined_{suffix}{model_key}.predictions.json"
            with open(path) as f:
                recs = json.load(f)
            true = [r["dG_true"] for r in recs]
            pred = [r["dG_pred"] for r in recs]
            out[(suffix, model_key)] = bootstrap_metrics_ci(true, pred, n_boot=N_BOOT)
    return out


def panel_row(ax, results, metric_key, metric_label, show_legend=False):
    n_groups = len(SPLITS)
    n_bars = len(MODELS)
    width = 0.65 / n_bars
    x = np.arange(n_groups)

    for i, (model_key, model_label) in enumerate(MODELS):
        vals, lo_err, hi_err = [], [], []
        for _, suffix in SPLITS:
            m = results[(suffix, model_key)]
            vals.append(m[metric_key])
            lo_err.append(m[metric_key] - m[f"{metric_key}_ci_lo"])
            hi_err.append(m[f"{metric_key}_ci_hi"] - m[metric_key])
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=[lo_err, hi_err], capsize=3.5,
               color=COLORS[model_key], edgecolor="black", linewidth=0.7,
               label=model_label, error_kw=dict(linewidth=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in SPLITS])
    ax.set_ylabel(metric_label)
    if metric_key == "rmse":
        ax.set_ylim(0, max(results[k]["rmse_ci_hi"] for k in results) * 1.28)
    else:
        ax.set_ylim(0, min(1.0, max(results[k][f"{metric_key}_ci_hi"] for k in results) * 1.3))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="lower center", ncol=3, framealpha=0.85, bbox_to_anchor=(0.5, 1.16))


# --- D-E: statistical summary --------------------------------------------
def _pool_seeds(predictions_by_seed):
    ids, true, pred = [], [], []
    for s in predictions_by_seed:
        ids.extend(s.get("ids", [None] * len(s["true"])))
        true.extend(s["true"])
        pred.extend(s["pred"])
    return ids, true, pred


def compute_forest_rows():
    rows = []

    d = json.load(open("affinity_extended_results.json"))
    _, true, pred = _pool_seeds(d["benchmark46"]["predictions_by_seed"])
    ci = bootstrap_metrics_ci(true, pred, n_boot=N_BOOT)
    rows.append(("Absolute affinity\n(benchmark46)", "SIMBA (3 seeds pooled)", ci["pearson"],
                 ci["pearson_ci_lo"], ci["pearson_ci_hi"], "PPI-Graphomer\n(structure)", 0.641))

    d = json.load(open("ddg_multiseed_results.json"))
    seed0 = d["abdesign_ddg"]["predictions_by_seed"][0]
    ci = bootstrap_per_complex_metrics_ci(seed0["ids"], seed0["true"], seed0["pred"], min_n=3, n_boot=N_BOOT)
    rows.append(("Mutation effect\n(AbDesign holdout,\nper-complex mean)", "SIMBA (seed 0)",
                 ci["mean_pearson"], ci["mean_pearson_ci_lo"], ci["mean_pearson_ci_hi"],
                 "DSMBind / Binding-DDG-pred. /\nRDE-PPI (reported, ~0-0.1)", 0.05))

    simba_p = json.load(open("simba_vs_mint_skempi_split_results.json"))["mint_split"]["predictions"]
    mint_p = json.load(open("mint_own_method_skempi_results.json"))["predictions"]
    ci = bootstrap_metrics_ci(simba_p["true"], simba_p["pred"], n_boot=N_BOOT)
    mint_ci = bootstrap_metrics_ci(mint_p["true"], mint_p["pred"], n_boot=N_BOOT)
    rows.append(("Mutation effect\n(SIMBA vs. MINT split)", "SIMBA (single run)", ci["pearson"],
                 ci["pearson_ci_lo"], ci["pearson_ci_hi"], "MINT\n(own method)", mint_ci["pearson"]))

    return rows


def panel_forest(ax, rows):
    y = np.arange(len(rows))[::-1]
    for yi, (label, simba_label, val, lo, hi, base_label, base_val) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=_COLOR_SIMBA, linewidth=2.4, solid_capstyle="round")
        ax.plot(val, yi, "o", color=_COLOR_SIMBA, markersize=11, zorder=3,
                label="SIMBA (95% bootstrap CI)" if yi == y[0] else None)
        ax.plot(base_val, yi, "o", color="#d62728", markersize=9, markeredgecolor="black",
                markeredgewidth=0.6, zorder=3,
                label="baseline (reported value)" if yi == y[0] else None)
        ax.text(base_val, yi - 0.26, base_label, va="top", ha="center", fontsize=12.5,
                color="#555555", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-0.15, 1.02)
    ax.set_ylim(y.min() - 0.8, y.max() + 0.55)
    ax.set_xlabel("Pearson correlation (higher is better)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.95)


def load_trimmed(path, pad=15):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    non_white = np.any(arr < 250, axis=-1)
    rows_ = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    r0, r1 = np.argmax(rows_), len(rows_) - np.argmax(rows_[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, arr.shape[0]), min(c1 + pad, arr.shape[1])
    return arr[r0:r1, c0:c1]


def main():
    print("[panel1] loading split-robustness predictions + computing bootstrap CIs...")
    split_results = load_split_robustness()
    print("[panel1] computing pooled/per-complex bootstrap CIs for the forest plot...")
    forest_rows = compute_forest_rows()
    for label, simba_label, val, lo, hi, base_label, base_val in forest_rows:
        print(f"  {label.splitlines()[0]}: SIMBA={val:.3f} [{lo:.3f},{hi:.3f}]  "
              f"{base_label.splitlines()[0]}={base_val:.3f}")

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.4, wspace=0.9,
                          top=0.93, bottom=0.06, left=0.045, right=0.99)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    for ax, (metric_key, metric_label) in zip((ax_a, ax_b, ax_c), METRICS):
        panel_row(ax, split_results, metric_key, metric_label, show_legend=(ax is ax_b))

    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_d.imshow(load_trimmed("figures/fisher_intra_target_bars.png"), aspect="auto")
    ax_d.axis("off")

    ax_e = fig.add_subplot(gs[1, 2:6])
    panel_forest(ax_e, forest_rows)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d, ax_e), "ABCDE"):
        ax.text(-0.02, 1.05, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    out_path = "figures/panel1_split_robustness.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[panel1] saved {out_path}")
    pdf_path = "figures/panel1_split_robustness.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[panel1] saved {pdf_path}")


if __name__ == "__main__":
    main()
