"""Camera-ready versions of the four paper figures, drawn natively at final
print size (7.2 in wide, 7-9 pt text) instead of embedding large rasters.

  Panel1.pdf               split robustness + statistical summary
  panel 2.pdf              accuracy, backbone choice, transfer
  reviewer_experiments.pdf robustness controls
  Panel 4.pdf              dataset characteristics

Run from the repository root (where the *_results.json files live):
  PYTHONPATH=simba/src python -m simba.make_camera_ready_figures [panel1 panel2 controls datasets]
Output goes to figures/camera_ready/ (PDF with embedded TrueType fonts + 300 dpi PNG).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import (  # noqa: E402
    bootstrap_metrics_ci, bootstrap_per_complex_metrics_ci, permutation_test_paired_metric,
)
from simba.figure_style import sequential_blues, sequential_purples  # noqa: E402

OUT_DIR = "figures/camera_ready"
N_BOOT = 1000
N_PERM = 2000
W = 7.2  # full text width (inches), matches figure* in acmart sigconf

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
LETTER_KW = dict(fontsize=11, fontweight="bold", va="bottom", ha="left")
DG, DDG = "ΔG", "ΔΔG"
ERR = dict(linewidth=0.9, capthick=0.9)


def cached(key, fn):
    """Memoise slow bootstrap / permutation results on disk so layout-only edits re-render instantly."""
    import pickle
    path = f"{OUT_DIR}/.cache/{key}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    val = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(val, f)
    return val


def new_fig(height):
    plt.rcParams.update(STYLE)
    return plt.figure(figsize=(W, height))


def letter(ax, s, dx=0.0, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, **LETTER_KW)


def tidy(ax, grid="y"):
    ax.grid(axis=grid, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    print(f"saved {OUT_DIR}/{name}.pdf/.png")
    plt.close(fig)


def lohi(m, key):
    return (m[key] - m[f"{key}_ci_lo"], m[f"{key}_ci_hi"] - m[key])


def grouped_bars(ax, groups, series, colors, labels, width, ylim=None, errs=None):
    """series: per-legend-entry lists of per-group values; errs: same shape of (lo, hi) tuples."""
    n = len(series)
    x = np.arange(len(groups))
    for i, vals in enumerate(series):
        off = (i - (n - 1) / 2) * width
        kw = {}
        if errs is not None:
            kw = dict(yerr=np.array([[e[0] for e in errs[i]], [e[1] for e in errs[i]]]), capsize=2.2,
                      error_kw=ERR)
        ax.bar(x + off, vals, width, color=colors[i], edgecolor="black", linewidth=0.5,
               label=labels[i], **kw)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)
    tidy(ax)


# --------------------------------------------------------------------------
# Panel 1
# --------------------------------------------------------------------------

def panel1():
    from simba.make_panel1_figure import MODELS, SPLITS, compute_forest_rows, load_split_robustness
    split_results = cached("p1_split", load_split_robustness)
    forest = cached("p1_forest", compute_forest_rows)
    fisher = json.load(open("figures/fisher_intra_target_results.json"))

    fig = new_fig(5.6)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.62, wspace=1.15,
                          top=0.88, bottom=0.09, left=0.07, right=0.99)
    axes_top = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    colors = sequential_blues(len(MODELS))
    split_labels = ["Complex\n(random)", "Sequence\nsimilarity", "Cold target\n(held-out\npartner)"]
    metrics = [("pearson", "Pearson correlation"), ("spearman", "Spearman correlation"),
               ("rmse", f"RMSE, {DG} (kcal/mol)")]
    for ax, (key, ylabel) in zip(axes_top, metrics):
        series, errs = [], []
        for model_key, _ in MODELS:
            ms = [split_results[(suffix, model_key)] for _, suffix in SPLITS]
            series.append([m[key] for m in ms])
            errs.append([lohi(m, key) for m in ms])
        top = max(split_results[k][f"{key}_ci_hi"] for k in split_results)
        grouped_bars(ax, split_labels, series, colors, [l for _, l in MODELS], 0.26,
                     ylim=(0, top * 1.12), errs=errs)
        ax.set_ylabel(ylabel)
    h, l = axes_top[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))

    ax_d = fig.add_subplot(gs[1, 0:3])
    labels = ["Overall (pooled)", "Pearson (naive)", "Pearson (Fisher-z)", "Spearman (naive)",
              "Spearman (Fisher-z)"]
    vals = [fisher["overall_pearson"], fisher["naive_mean_pearson"], fisher["fisher_mean_pearson"],
            fisher["naive_mean_spearman"], fisher["fisher_mean_spearman"]]
    bars = ax_d.bar(labels, vals, width=0.82, color=sequential_blues(5), edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax_d.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax_d.set_ylabel("Correlation")
    ax_d.set_ylim(0, 0.8)
    ax_d.tick_params(axis="x", labelsize=7.2)
    plt.setp(ax_d.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    tidy(ax_d)
    pos = ax_d.get_position()
    ax_d.set_position([pos.x0, pos.y0, pos.width * 0.78, pos.height])

    ax_e = fig.add_subplot(gs[1, 3:6])
    y_labels = ["Absolute affinity\n(benchmark46)", "Mutation effect\n(AbDesign holdout)",
                "Mutation effect\n(MINT split)"]
    base_labels = ["PPI-Graphomer\n(structure)", "DSMBind, Binding-DDG,\nRDE-PPI (reported)", "MINT\n(own method)"]
    y = np.arange(len(forest))[::-1]
    blue = sequential_blues(1)[0]
    for yi, row, base_label in zip(y, forest, base_labels):
        val, lo, hi, base_val = row[2], row[3], row[4], row[6]
        ax_e.plot([lo, hi], [yi, yi], color=blue, linewidth=1.6, solid_capstyle="round")
        ax_e.plot(val, yi, "o", color=blue, markersize=5.5, zorder=3,
                  label="SIMBA (95% bootstrap CI)" if yi == y[0] else None)
        ax_e.plot(base_val, yi, "o", color="#d62728", markersize=5, markeredgecolor="black",
                  markeredgewidth=0.4, zorder=3, label="Baseline (reported)" if yi == y[0] else None)
        ax_e.text(base_val - (0.03 if base_val < 0.15 else 0), yi + 0.17, base_label, va="bottom",
                  ha="left" if base_val < 0.15 else "center", fontsize=6.5,
                  color="#555555", style="italic")
    ax_e.set_yticks(y)
    ax_e.set_yticklabels(y_labels)
    ax_e.axvline(0, color="black", linewidth=0.5)
    ax_e.set_xlim(-0.12, 1.0)
    ax_e.set_ylim(y.min() - 0.55, y.max() + 0.95)
    ax_e.set_xlabel("Pearson correlation")
    ax_e.legend(loc="upper left", frameon=False)
    tidy(ax_e, grid="x")

    letter(axes_top[0], "A", dy=1.04)
    letter(axes_top[1], "B", dy=1.04)
    letter(axes_top[2], "C", dy=1.04)
    letter(ax_d, "D")
    letter(ax_e, "E", dx=-0.12)
    save(fig, "Panel1")


# --------------------------------------------------------------------------
# Panel 2
# --------------------------------------------------------------------------

def _fold_mean_metrics(d):
    rm = d.get("fold_rmse_mean", d["fold_mae_mean"])
    rs = d.get("fold_rmse_std", d["fold_mae_std"])
    sc, ss = d.get("fold_scc_mean", np.nan), d.get("fold_scc_std", 0.0)
    return {"pearson": d["fold_pcc_mean"], "pearson_ci_lo": d["fold_pcc_mean"] - d["fold_pcc_std"],
            "pearson_ci_hi": d["fold_pcc_mean"] + d["fold_pcc_std"],
            "spearman": sc, "spearman_ci_lo": sc - ss, "spearman_ci_hi": sc + ss,
            "rmse": rm, "rmse_ci_lo": rm - rs, "rmse_ci_hi": rm + rs}


def panel2():
    fisher = json.load(open("figures/fisher_intra_target_results.json"))
    simba_p = json.load(open("simba_vs_mint_skempi_split_results.json"))["mint_split"]["predictions"]
    mint_p = json.load(open("mint_own_method_skempi_results.json"))["predictions"]
    nov = json.load(open("novel_generalization_fewshot_scaling_results.json"))

    fig = new_fig(5.4)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.15, 1], hspace=0.85, wspace=2.6,
                          top=0.93, bottom=0.12, left=0.075, right=0.99)
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b1 = fig.add_subplot(gs[0, 5:10])
    ax_b2 = fig.add_subplot(gs[0, 10:12])

    true, pred = np.array(fisher["scatter_true"]), np.array(fisher["scatter_pred"])
    ax_a.scatter(true, pred, s=2.5, alpha=0.35, color="#0077b6", edgecolors="none", rasterized=True)
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax_a.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, label="Perfect prediction")
    ax_a.set_xlabel(f"True {DG} (kcal/mol)")
    ax_a.set_ylabel(f"Predicted {DG} (kcal/mol)")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", frameon=False)
    tidy(ax_a, grid="both")

    sp = cached("p2_simba_pool", lambda: bootstrap_metrics_ci(simba_p["true"], simba_p["pred"], n_boot=N_BOOT))
    mp = cached("p2_mint_pool", lambda: bootstrap_metrics_ci(mint_p["true"], mint_p["pred"], n_boot=N_BOOT))
    ss = cached("p2_simba_struct", lambda: bootstrap_per_complex_metrics_ci(
        simba_p["ids"], simba_p["true"], simba_p["pred"], min_n=10, n_boot=N_BOOT))
    ms = cached("p2_mint_struct", lambda: bootstrap_per_complex_metrics_ci(
        mint_p["ids"], mint_p["true"], mint_p["pred"], min_n=10, n_boot=N_BOOT))
    aligned = len(simba_p["true"]) == len(mint_p["true"]) and np.allclose(simba_p["true"], mint_p["true"], atol=1e-6)
    pvals = {}
    if aligned:
        for m in ("pearson", "spearman", "rmse"):
            pvals[m] = cached(f"p2_perm_{m}", lambda m=m: permutation_test_paired_metric(
                simba_p["true"], simba_p["pred"], mint_p["pred"], metric=m, n_perm=N_PERM)["p_value"])
    c_simba, c_mint = "#023e8a", "#90e0ef"
    groups = ["Overall\nPearson", "Overall\nSpearman", "Per-struct.\nPearson", "Per-struct.\nSpearman"]

    def entries(pool, st):
        return [(pool["pearson"], lohi(pool, "pearson")), (pool["spearman"], lohi(pool, "spearman")),
                (st["mean_pearson"], lohi(st, "mean_pearson")), (st["mean_spearman"], lohi(st, "mean_spearman"))]
    es, em = entries(sp, ss), entries(mp, ms)
    grouped_bars(ax_b1, groups, [[e[0] for e in es], [e[0] for e in em]], [c_simba, c_mint],
                 ["SIMBA", "MINT (embeddings + ridge)"], 0.36, ylim=(0, 0.7),
                 errs=[[e[1] for e in es], [e[1] for e in em]])
    ax_b1.set_ylabel("Correlation")
    ax_b1.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", frameon=False, ncol=2, columnspacing=1.0, handlelength=1.2)
    for i, m in enumerate(("pearson", "spearman")):
        if m in pvals:
            p = pvals[m]
            ax_b1.text(i, max(es[i][0] + es[i][1][1], em[i][0] + em[i][1][1]) + 0.03,
                       "p<0.001" if p < 0.001 else f"p={p:.3f}", ha="center", fontsize=6.5, style="italic",
                       color="#444444")
    grouped_bars(ax_b2, ["Overall\nRMSE"], [[sp["rmse"]], [mp["rmse"]]], [c_simba, c_mint], ["", ""], 0.36,
                 ylim=(0, 2.9), errs=[[lohi(sp, "rmse")], [lohi(mp, "rmse")]])
    ax_b2.set_ylabel("RMSE (kcal/mol)")
    if "rmse" in pvals:
        p = pvals["rmse"]
        ax_b2.text(0, max(sp["rmse_ci_hi"], mp["rmse_ci_hi"]) + 0.08, "p<0.001" if p < 0.001 else f"p={p:.3f}",
                   ha="center", fontsize=6.5, style="italic", color="#444444")

    files = [("ESM-2", "plm_ablation_cold_esm2_results.json"), ("MINT", "plm_ablation_cold_mint_results.json"),
             ("AbLang2", "plm_ablation_cold_ablang2_results.json")]
    bb = {}
    for name, path in files:
        d = json.load(open(path))
        if "predictions" in d:
            bb[name] = cached(f"p2_bb_{name}", lambda d=d: bootstrap_metrics_ci(
                d["predictions"]["true"], d["predictions"]["pred"], n_boot=N_BOOT))
        else:
            print(f"[warn] {path} has no per-example predictions; plotting fold mean +/- s.d.")
            bb[name] = _fold_mean_metrics(d)
    names = [n for n, _ in files]
    ax_c1 = fig.add_subplot(gs[1, 0:3])
    ax_c2 = fig.add_subplot(gs[1, 3:5])
    cb = ["#005b96", "#00b4d8", "#caf0f8"]
    grouped_bars(ax_c1, ["Pearson", "Spearman"],
                 [[bb[n]["pearson"], bb[n]["spearman"]] for n in names], cb, names, 0.26, ylim=(0, 0.8),
                 errs=[[lohi(bb[n], "pearson"), lohi(bb[n], "spearman")] for n in names])
    ax_c1.set_ylabel("Correlation")
    ax_c1.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", frameon=False, ncol=3, columnspacing=0.8, handlelength=1.0)
    errs = [lohi(bb[n], "rmse") for n in names]
    ax_c2.bar(names, [bb[n]["rmse"] for n in names], 0.6, color=cb, edgecolor="black", linewidth=0.5,
              yerr=[[e[0] for e in errs], [e[1] for e in errs]], capsize=2.2, error_kw=ERR)
    ax_c2.set_ylabel("RMSE (kcal/mol)")
    ax_c2.set_ylim(0, 3.4)
    ax_c2.tick_params(axis="x", labelrotation=30)
    tidy(ax_c2)

    p = nov["predictions"]
    zc = cached("p2_zero", lambda: bootstrap_metrics_ci(p["true"], p["zeroshot_pred"], n_boot=N_BOOT))
    fc = cached("p2_few", lambda: bootstrap_metrics_ci(p["true"], p["fewshot_pred_by_fraction"]["0.10"], n_boot=N_BOOT))
    ax_d1 = fig.add_subplot(gs[1, 6:9])
    ax_d2 = fig.add_subplot(gs[1, 9:12])
    cd = sequential_blues(2)
    grouped_bars(ax_d1, ["Pearson", "Spearman"], [[zc["pearson"], zc["spearman"]], [fc["pearson"], fc["spearman"]]],
                 cd, ["Zero-shot", "Few-shot"], 0.34, ylim=(0, 0.75),
                 errs=[[lohi(zc, "pearson"), lohi(zc, "spearman")], [lohi(fc, "pearson"), lohi(fc, "spearman")]])
    ax_d1.set_ylabel("Correlation")
    ax_d1.legend(bbox_to_anchor=(0.5, 1.02), loc="lower center", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.0)
    ze, fe = lohi(zc, "rmse"), lohi(fc, "rmse")
    ax_d2.bar(["Zero-shot", "Few-shot"], [zc["rmse"], fc["rmse"]], 0.55, color=cd, edgecolor="black", linewidth=0.5,
              yerr=[[ze[0], fe[0]], [ze[1], fe[1]]], capsize=2.2, error_kw=ERR)
    ax_d2.set_ylabel(f"RMSE, {DDG} (kcal/mol)")
    ax_d2.set_ylim(0, 1.8)
    ax_d2.tick_params(axis="x", labelrotation=30)
    tidy(ax_d2)

    letter(ax_a, "A", dx=-0.2, dy=1.12)
    letter(ax_b1, "B", dx=-0.17, dy=1.12)
    letter(ax_c1, "C", dx=-0.2, dy=1.2)
    letter(ax_d1, "D", dx=-0.2, dy=1.2)
    save(fig, "panel 2")


# --------------------------------------------------------------------------
# Robustness controls
# --------------------------------------------------------------------------

def _corr_bars(entries):
    series, errs = [], []
    for preds, trues in entries:
        ci = bootstrap_metrics_ci(trues, preds, n_boot=N_BOOT)
        series.append([ci["pearson"], ci["spearman"]])
        errs.append([lohi(ci, "pearson"), lohi(ci, "spearman")])
    return series, errs


def controls():
    symm = json.load(open("symmetrization_test_results.json"))
    gate = json.load(open("gate_ablation_results.json"))
    direct = json.load(open("direct_ddg_head_results.json"))
    cross = json.load(open("mint_protocol_cross_results.json"))
    pur = sequential_purples(3)
    pal2 = sequential_purples(2)

    fig = new_fig(4.7)
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.2, top=0.95, bottom=0.1, left=0.07, right=0.93)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3))
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    s = symm["summary"]
    keys = ["pcc_forward", "pcc_swapped", "pcc_averaged"]
    ax_a.bar(["f(A,B)", "f(B,A)", "Average"], [s[k]["mean"] for k in keys], 0.6,
             yerr=[s[k]["std"] for k in keys], capsize=2.2, error_kw=ERR,
             color=[pur[2], pur[0], pur[1]], edgecolor="black", linewidth=0.5)
    ax_a.set_ylabel("Pearson correlation")
    ax_a.set_ylim(0, 1.0)
    tidy(ax_a)

    seed0 = symm["per_seed"][0]
    fwd, swp = np.array(seed0["pred_forward"]), np.array(seed0["pred_swapped"])
    lo, hi = min(fwd.min(), swp.min()), max(fwd.max(), swp.max())
    ax_b.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, label="Perfect symmetry")
    ax_b.scatter(fwd, swp, s=9, alpha=0.75, color=pur[2], edgecolors="black", linewidth=0.3)
    ax_b.set_xlabel(f"f(A, B) predicted {DG} (kcal/mol)")
    ax_b.set_ylabel(f"f(B, A) predicted {DG} (kcal/mol)")
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.legend(loc="upper left", frameon=False)
    tidy(ax_b, grid="both")

    g = {r["gated"]: r for r in gate["results"]}
    ser, err = _corr_bars([(g[True]["predictions"]["pred"], g[True]["predictions"]["true"]),
                           (g[False]["predictions"]["pred"], g[False]["predictions"]["true"])])
    grouped_bars(ax_c, ["Pearson", "Spearman"], ser, [pal2[1], pal2[0]], ["Gated (SIMBA)", "Ungated"], 0.34,
                 ylim=(0, 0.98), errs=err)
    ax_c.set_ylabel("Correlation")
    ax_c.legend(loc="upper center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.0)

    ds, dh = direct["subtraction"]["predictions"], direct["direct_head"]["predictions"]
    ser, err = _corr_bars([(ds["pred"], ds["true"]), (dh["pred"], dh["true"])])
    grouped_bars(ax_d, ["Overall Pearson", "Overall Spearman"], ser, [pal2[1], pal2[0]],
                 ["Two-pass subtraction (SIMBA)", f"Direct joint {DDG} head"], 0.34, ylim=(0, 0.75), errs=err)
    ax_d.set_ylabel("Correlation")
    ax_d.legend(loc="upper right", frameon=False, handlelength=1.0)

    grid = np.array([[0.528, cross["esm2_ridge"]["overall"]["pearson"]],
                     [cross["simba_mint_backbone"]["overall"]["pearson"], 0.375]])
    im = ax_e.imshow(grid, cmap=plt.get_cmap("Purples"), vmin=0.3, vmax=0.6, aspect="auto")
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(["Pairformer head", "Ridge head"])
    ax_e.set_yticks([0, 1])
    ax_e.set_yticklabels(["ESM-2", "MINT"])
    ax_e.set_ylabel("Backbone")
    for i in range(2):
        for j in range(2):
            ax_e.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                      color="white" if grid[i, j] > 0.47 else "black")
    for side in ax_e.spines.values():
        side.set_visible(False)
    ax_e.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax_e, fraction=0.05, pad=0.03)
    cbar.set_label("Overall Pearson")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    letter(ax_a, "A", dx=-0.25, dy=1.08)
    letter(ax_b, "B", dx=-0.25, dy=1.08)
    letter(ax_c, "C", dx=-0.25, dy=1.08)
    letter(ax_d, "D", dx=-0.12, dy=1.06)
    letter(ax_e, "E", dx=-0.2, dy=1.06)
    save(fig, "reviewer_experiments")


# --------------------------------------------------------------------------
# Dataset characteristics
# --------------------------------------------------------------------------

def datasets():
    from simba import env
    env.setup()
    from simba.data.build_combined_affinity import build_combined_examples
    from simba.data.dataset import load_rde_skempi

    combined = build_combined_examples()
    rde = load_rde_skempi()
    dg = np.array([e.dG for e in combined])
    ddg = np.array([e.ddG for e in rde if e.ddG is not None])
    by_source = {}
    for e in combined:
        by_source.setdefault(e.meta["dataset"], []).append(e.dG)
    sizes = {
        "SKEMPI2-only pool": len([e for e in combined if e.meta["dataset"] == "skempi2"]),
        "Combined pool": len(combined),
        "RDE-PPI (3-fold)": len(rde),
        "AbDesign": 1303,
        "Kastritis-lineage": 45,
        "eLife Test 1": 66,
    }
    fig = new_fig(4.6)
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34, top=0.96, bottom=0.1, left=0.09, right=0.99)
    c5 = sequential_blues(5)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
    ax_a.hist(dg, bins=40, color=c5[3], edgecolor="black", linewidth=0.3)
    ax_a.set_xlabel(f"Binding free energy, {DG} (kcal/mol)")
    ax_a.set_ylabel("Count")
    tidy(ax_a)
    ax_b.hist(ddg, bins=40, color=c5[1], edgecolor="black", linewidth=0.3)
    ax_b.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_b.set_xlabel(f"Binding-energy change, {DDG} (kcal/mol)")
    ax_b.set_ylabel("Count")
    tidy(ax_b)
    labels = list(sizes)
    vals = [sizes[k] for k in labels]
    bars = ax_c.barh(labels, vals, color=sequential_blues(len(labels)), edgecolor="black", linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_xlim(30, 4e4)
    ax_c.set_xlabel("Number of examples (log scale)")
    for b, v in zip(bars, vals):
        ax_c.text(b.get_width() * 1.1, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=7)
    tidy(ax_c, grid="x")
    names = {"skempi2": "SKEMPI2", "benchmark46": "Kastritis-\nlineage", "atlas": "ATLAS\nTCR:pMHC"}
    order = [k for k in ("skempi2", "benchmark46", "atlas") if k in by_source]
    bp = ax_d.boxplot([by_source[k] for k in order], patch_artist=True, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.2), flierprops=dict(markersize=2.5),
                      whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0.7), boxprops=dict(linewidth=0.7))
    ax_d.set_xticks(range(1, len(order) + 1))
    ax_d.set_xticklabels([names[k] for k in order])
    for patch, color in zip(bp["boxes"], sequential_blues(len(order))):
        patch.set_facecolor(color)
    ax_d.set_ylabel(f"Binding free energy, {DG} (kcal/mol)")
    tidy(ax_d)
    for ax, s in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        letter(ax, s, dx=-0.2, dy=1.05)
    save(fig, "Panel 4")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["panel1", "panel2", "controls", "datasets"]
    for name in todo:
        {"panel1": panel1, "panel2": panel2, "controls": controls, "datasets": datasets}[name]()
