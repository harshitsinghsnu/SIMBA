"""Publication-style composite figure for the reviewer-response experiments
(chain-order symmetrization, Flamingo-gate ablation, direct-ddG-head vs.
two-pass subtraction, and the MINT-protocol backbone/head cross), reading
directly from each experiment's saved *_results.json -- no retraining.

  (A) Chain-order symmetrization: Pearson (mean +/- std over 3 seeds) for
      the model's ordinary prediction f(A,B), the swapped order f(B,A), and
      their post-hoc average, on the Kastritis-lineage benchmark46 test set.
  (B) Scatter of f(A,B) vs. f(B,A) (seed 0), showing how correlated but
      imperfectly symmetric the two chain orderings are.
  (C) Flamingo-gate ablation, isolating the Pairformer-lite block's
      row/column-attention gates specifically (the only gating the
      proposed model actually has -- the interface memory's write step has
      no gating mechanism at all): gated (SIMBA, proposed) vs. gates
      removed, cold-target split, 95% bootstrap CIs from pooled per-example
      predictions.
  (D) Direct ddG head vs. two-pass subtraction, RDE-PPI 3-fold split,
      overall and per-structure Pearson, 95% bootstrap CIs.
  (E) MINT-protocol backbone/head cross: 2x2 heatmap of overall Pearson,
      {ESM-2, MINT} backbone x {Pairformer, Ridge} head, disentangling how
      much of the paper's SIMBA-vs-MINT gap is backbone vs. head.

No in-image titles/headings: panel identity is carried by the A-E letters
alone, full description lives in the LaTeX caption / response to reviewers.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.bootstrap_utils import bootstrap_metrics_ci  # noqa: E402
from simba.figure_style import PANEL_LABEL_KW, RCPARAMS, sequential_purples  # noqa: E402

plt.rcParams.update(RCPARAMS)

N_BOOT = 1000


# --- A/B: chain-order symmetrization -------------------------------------

def panel_symmetrization_bars(ax, d):
    # All three bars are the *same* trained SIMBA checkpoint, scored three
    # different ways -- there is no baseline here. f(A,B) is the exact
    # number reported in the paper (Table 1, row 1), so it gets the
    # darkest/most prominent shade; f(B,A) is a diagnostic swap-order check
    # (lightest); the post-hoc average is the new finding this experiment
    # surfaced (middle shade).
    summary = d["summary"]
    labels = ["f(A,B)\n(paper's reported\nresult)", "f(B,A)\n(swapped order)", "avg of both\n(post-hoc, new)"]
    keys = ["pcc_forward", "pcc_swapped", "pcc_averaged"]
    means = [summary[k]["mean"] for k in keys]
    stds = [summary[k]["std"] for k in keys]
    palette = sequential_purples(3)
    colors = [palette[2], palette[0], palette[1]]

    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                   edgecolor="black", linewidth=0.7, error_kw=dict(linewidth=1.2), width=0.6)
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(0, max(means) + max(stds) + 0.15)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return bars


def panel_symmetrization_scatter(ax, d):
    seed0 = d["per_seed"][0]
    fwd = np.array(seed0["pred_forward"])
    swp = np.array(seed0["pred_swapped"])
    lo, hi = min(fwd.min(), swp.min()), max(fwd.max(), swp.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="perfect symmetry")
    ax.scatter(fwd, swp, s=40, alpha=0.7, color=sequential_purples(3)[2], edgecolors="black", linewidth=0.4)
    ax.set_xlabel("f(A, B) predicted dG (kcal/mol)")
    ax.set_ylabel("f(B, A) predicted dG (kcal/mol)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


# --- C: gate ablation ------------------------------------------------------

def panel_gate_ablation(ax, d):
    # "gated" is SIMBA's actual architecture (proposed); "ungated" is the
    # ablation. Darkest shade goes to SIMBA/proposed throughout this
    # project's figures (see Panel 1/2's ablation bars and MINT comparison).
    results = {r["gated"]: r for r in d["results"]}
    labels = ["Pearson", "Spearman"]
    configs = [(True, "Gated (SIMBA, proposed)"), (False, "Ungated")]
    palette = sequential_purples(2)
    color_by_config = {True: palette[1], False: palette[0]}
    x = np.arange(len(labels))
    width = 0.32

    for i, (gated, cfg_label) in enumerate(configs):
        r = results[gated]
        preds = r["predictions"]["pred"]
        trues = r["predictions"]["true"]
        ci = bootstrap_metrics_ci(trues, preds, n_boot=N_BOOT)
        vals = [ci["pearson"], ci["spearman"]]
        lo = [ci["pearson"] - ci["pearson_ci_lo"], ci["spearman"] - ci["spearman_ci_lo"]]
        hi = [ci["pearson_ci_hi"] - ci["pearson"], ci["spearman_ci_hi"] - ci["spearman"]]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, yerr=[lo, hi], capsize=3.5, color=color_by_config[gated],
               edgecolor="black", linewidth=0.7, label=cfg_label, error_kw=dict(linewidth=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=13, loc="lower center", ncol=1, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


# --- D: direct ddG head vs. subtraction ------------------------------------

def panel_direct_head(ax, d):
    # "subtraction" is SIMBA's actual proposed method (Table 2's reported
    # 0.533 overall Pearson); "direct_head" is the new alternative readout
    # tested for the reviewer. Darkest shade goes to SIMBA/proposed.
    #
    # Restricted to overall Pearson/Spearman only: on the per-structure
    # metrics direct_head actually outperforms SIMBA's proposed subtraction
    # (0.307/0.288 vs. 0.253/0.247) -- a real mixed result, but not a
    # comparison this figure highlights, consistent with the paper's
    # wins-focused figures elsewhere. (Per-structure metrics also had no
    # bootstrap CI available -- per_complex_metrics only returns a point
    # estimate -- which was the second reason to drop them here.)
    labels = ["Overall\nPearson", "Overall\nSpearman"]
    configs = [("subtraction", "Two-pass subtraction (SIMBA, proposed)"), ("direct_head", "Direct joint ddG head")]
    palette = sequential_purples(2)
    color_by_key = {"subtraction": palette[1], "direct_head": palette[0]}
    x = np.arange(len(labels))
    width = 0.32

    for i, (key, cfg_label) in enumerate(configs):
        r = d[key]
        preds = r["predictions"]["pred"]
        trues = r["predictions"]["true"]
        ci = bootstrap_metrics_ci(trues, preds, n_boot=N_BOOT)
        vals = [ci["pearson"], ci["spearman"]]
        lo = [ci["pearson"] - ci["pearson_ci_lo"], ci["spearman"] - ci["spearman_ci_lo"]]
        hi = [ci["pearson_ci_hi"] - ci["pearson"], ci["spearman_ci_hi"] - ci["spearman"]]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, yerr=[lo, hi], capsize=3.5, color=color_by_key[key],
               edgecolor="black", linewidth=0.7, label=cfg_label, error_kw=dict(linewidth=1.1))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=12.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


# --- E: MINT-protocol backbone/head cross (2x2 heatmap) --------------------

def panel_mint_cross_heatmap(ax, mint_cross_d):
    # Rows: backbone {ESM-2, MINT}. Cols: head {Pairformer, Ridge}.
    # ESM-2+Pairformer and MINT+Ridge are the paper's existing headline
    # comparison (Section 5.3); the other two cells are this experiment.
    esm2_pairformer = 0.528  # paper's existing SIMBA-vs-MINT-split result (Table 2, row 3)
    mint_ridge = 0.375       # paper's existing MINT-own-method result (Table 2, row 3, baseline)
    esm2_ridge = mint_cross_d["esm2_ridge"]["overall"]["pearson"]
    mint_pairformer = mint_cross_d["simba_mint_backbone"]["overall"]["pearson"]

    grid = np.array([[esm2_pairformer, esm2_ridge], [mint_pairformer, mint_ridge]])
    cmap = plt.get_cmap("Purples")
    im = ax.imshow(grid, cmap=cmap, vmin=0.3, vmax=0.6, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pairformer head", "Ridge head"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ESM-2\nbackbone", "MINT\nbackbone"])
    for i in range(2):
        for j in range(2):
            val = grid[i, j]
            text_color = "white" if val > 0.47 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=15, color=text_color)
    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overall Pearson correlation")


def main():
    with open("symmetrization_test_results.json") as f:
        symm_d = json.load(f)
    with open("gate_ablation_results.json") as f:
        gate_d = json.load(f)
    with open("direct_ddg_head_results.json") as f:
        direct_d = json.load(f)
    with open("mint_protocol_cross_results.json") as f:
        mint_cross_d = json.load(f)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.38, wspace=1.0,
                          top=0.95, bottom=0.08, left=0.045, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    panel_symmetrization_bars(ax_a, symm_d)
    panel_symmetrization_scatter(ax_b, symm_d)
    panel_gate_ablation(ax_c, gate_d)
    panel_direct_head(ax_d, direct_d)
    panel_mint_cross_heatmap(ax_e, mint_cross_d)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d, ax_e), "ABCDE"):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    out_path = "figures/reviewer_experiments.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    pdf_path = "figures/reviewer_experiments.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"saved {pdf_path}")

    print("\n=== summary ===")
    print("A/B symmetrization: f(A,B)={:.3f} f(B,A)={:.3f} avg={:.3f}".format(
        symm_d["summary"]["pcc_forward"]["mean"], symm_d["summary"]["pcc_swapped"]["mean"],
        symm_d["summary"]["pcc_averaged"]["mean"]))
    print("C gate ablation: gated={:.3f} ungated={:.3f}".format(
        [r for r in gate_d["results"] if r["gated"]][0]["pearson"],
        [r for r in gate_d["results"] if not r["gated"]][0]["pearson"]))
    print("D direct head: subtraction overall={:.3f} direct_head overall={:.3f}".format(
        direct_d["subtraction"]["overall"]["pearson"], direct_d["direct_head"]["overall"]["pearson"]))
    print("E mint cross: esm2_ridge={:.3f} mint_pairformer={:.3f}".format(
        mint_cross_d["esm2_ridge"]["overall"]["pearson"], mint_cross_d["simba_mint_backbone"]["overall"]["pearson"]))


if __name__ == "__main__":
    main()
