"""Post-hoc statistical analysis over every predictions file produced by
`run_all_experiments.sh`: computes Pearson/Spearman/RMSE with bootstrap
confidence intervals for every reported result, plus a paired permutation
test for the one same-rows/same-folds head-to-head comparison in the suite
(SIMBA vs. MINT's own method on the MINT-protocol SKEMPI split).

Reads only already-saved JSON files -- no GPU, no retraining, no PLM
embedding cache touched. Run this after `run_all_experiments.sh` (or any
subset of it) has produced the files it looks for; missing files are
reported and skipped, not treated as fatal.

Writes one consolidated `all_results_with_ci.json` plus a plain-text
summary table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simba.bootstrap_utils import (
    bootstrap_metrics_ci,
    bootstrap_per_complex_metrics_ci,
    permutation_test_paired_metric,
)

N_BOOT = 2000
N_PERM = 5000


def _load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _report_pooled(label, ids, true, pred, out, min_n=3):
    if not true or len(true) < 2:
        print(f"  [{label}] SKIPPED (no usable predictions, n={len(true) if true else 0})")
        return
    m = bootstrap_metrics_ci(true, pred, n_boot=N_BOOT)
    print(f"  [{label}] pooled  n={m['n']}  "
          f"Pearson={m['pearson']:.4f} [{m['pearson_ci_lo']:.4f},{m['pearson_ci_hi']:.4f}]  "
          f"Spearman={m['spearman']:.4f} [{m['spearman_ci_lo']:.4f},{m['spearman_ci_hi']:.4f}]  "
          f"RMSE={m['rmse']:.4f} [{m['rmse_ci_lo']:.4f},{m['rmse_ci_hi']:.4f}]")
    out[label] = {"pooled": m}
    if ids:
        pc = bootstrap_per_complex_metrics_ci(ids, true, pred, min_n=min_n, n_boot=N_BOOT)
        print(f"  [{label}] per-complex ({pc['n_complexes_scored']}/{pc['n_complexes_total']} scored, "
              f">={min_n} rows)  "
              f"meanPearson={pc['mean_pearson']:.4f} [{pc['mean_pearson_ci_lo']:.4f},{pc['mean_pearson_ci_hi']:.4f}]  "
              f"meanSpearman={pc['mean_spearman']:.4f} "
              f"[{pc['mean_spearman_ci_lo']:.4f},{pc['mean_spearman_ci_hi']:.4f}]")
        out[label]["per_complex"] = pc


def analyze_skempi2_5fold_matrix(out):
    print("\n=== Stage 2: 3-architecture x 3-split ablation matrix ===")
    for suffix in ("", "seqsim_", "coldtarget_"):
        for model in ("pairformer", "concat_mlp", "cross_attn_mlp"):
            path = f"skempi2_5fold_combined_{suffix}{model}.predictions.json"
            records = _load(path)
            if records is None:
                print(f"  [{path}] MISSING, skipped")
                continue
            ids = [r["complex_id"] for r in records]
            true = [r["dG_true"] for r in records]
            pred = [r["dG_pred"] for r in records]
            label = f"skempi2_5fold_combined_{suffix}{model}"
            _report_pooled(label, ids, true, pred, out)


def analyze_fisher(out):
    print("\n=== Stage 3: Fisher intra-target (pairformer, combined pool, complex split) ===")
    d = _load("figures/fisher_intra_target_results.json")
    if d is None:
        print("  [fisher_intra_target] MISSING, skipped")
        return
    _report_pooled("fisher_intra_target", d.get("scatter_ids"), d["scatter_true"], d["scatter_pred"], out)


def _analyze_predictions_by_seed(label, entry, out, min_n=3):
    seeds = entry.get("predictions_by_seed", [])
    if not seeds:
        print(f"  [{label}] no per-seed predictions saved, skipped")
        return
    per_seed = {}
    for s in seeds:
        seed_label = f"{label}_seed{s['seed']}"
        m = bootstrap_metrics_ci(s["true"], s["pred"], n_boot=N_BOOT)
        print(f"  [{seed_label}] n={m['n']}  Pearson={m['pearson']:.4f} "
              f"[{m['pearson_ci_lo']:.4f},{m['pearson_ci_hi']:.4f}]  "
              f"Spearman={m['spearman']:.4f} [{m['spearman_ci_lo']:.4f},{m['spearman_ci_hi']:.4f}]  "
              f"RMSE={m['rmse']:.4f} [{m['rmse_ci_lo']:.4f},{m['rmse_ci_hi']:.4f}]")
        per_seed[str(s["seed"])] = {"pooled": m}
        if s.get("ids"):
            pc = bootstrap_per_complex_metrics_ci(s["ids"], s["true"], s["pred"], min_n=min_n, n_boot=N_BOOT)
            per_seed[str(s["seed"])]["per_complex"] = pc
    out[label] = per_seed


def analyze_affinity_extended(out):
    print("\n=== Stage 4: absolute affinity vs. PPI-Graphomer (multiseed) ===")
    d = _load("affinity_extended_results.json")
    if d is None:
        print("  [affinity_extended_results.json] MISSING, skipped")
        return
    for test_name, entry in d.items():
        _analyze_predictions_by_seed(f"affinity_extended_{test_name}", entry, out)


def analyze_ppi_graphomer_test1(out):
    print("\n=== Stage 5: PPI-Graphomer Test Set 1 (eLife reproduction, multiseed) ===")
    rows = _load("ppi_graphomer_test1_per_row.json")
    if rows is None:
        print("  [ppi_graphomer_test1_per_row.json] MISSING, skipped")
        return
    by_seed: dict[int, list] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    per_seed = {}
    for seed, rs in sorted(by_seed.items()):
        ids = [r["complex_id"] for r in rs]
        true = [r["true"] for r in rs]
        pred = [r["pred"] for r in rs]
        m = bootstrap_metrics_ci(true, pred, n_boot=N_BOOT)
        print(f"  [ppi_graphomer_test1_seed{seed}] n={m['n']}  Pearson={m['pearson']:.4f} "
              f"[{m['pearson_ci_lo']:.4f},{m['pearson_ci_hi']:.4f}]  RMSE={m['rmse']:.4f}")
        per_seed[str(seed)] = {"pooled": m}
    out["ppi_graphomer_test1"] = per_seed


def analyze_ppi_affinity_test2(out):
    print("\n=== Stage 6: PPI-Affinity Test Set 2 (multiseed) ===")
    d = _load("ppi_affinity_test2_results.json")
    if d is None:
        print("  [ppi_affinity_test2_results.json] MISSING, skipped")
        return
    _analyze_predictions_by_seed("ppi_affinity_test2", d, out)


def analyze_ddg_multiseed(out):
    print("\n=== Stage 7: RDE-PPI + AbDesign ddG (multiseed) ===")
    d = _load("ddg_multiseed_results.json")
    if d is None:
        print("  [ddg_multiseed_results.json] MISSING, skipped")
        return
    if "rde" in d:
        _analyze_predictions_by_seed("ddg_multiseed_rde", d["rde"], out, min_n=10)
    if "abdesign_ddg" in d:
        _analyze_predictions_by_seed("ddg_multiseed_abdesign", d["abdesign_ddg"], out, min_n=3)


def analyze_mint_comparison(out):
    print("\n=== Stage 8: SIMBA vs. MINT's own method (same rows/folds) ===")
    simba_d = _load("simba_vs_mint_skempi_split_results.json")
    mint_d = _load("mint_own_method_skempi_results.json")
    if simba_d is None or "mint_split" not in simba_d:
        print("  [simba_vs_mint_skempi_split_results.json] MISSING or missing 'mint_split' key, skipped")
        simba_p = None
    else:
        simba_p = simba_d["mint_split"]["predictions"]
        _report_pooled("simba_mint_split", simba_p["ids"], simba_p["true"], simba_p["pred"], out, min_n=10)
    if mint_d is None or "predictions" not in mint_d:
        print("  [mint_own_method_skempi_results.json] MISSING or stale (no 'predictions' key), skipped")
        mint_p = None
    else:
        mint_p = mint_d["predictions"]
        _report_pooled("mint_own_method", mint_p["ids"], mint_p["true"], mint_p["pred"], out, min_n=10)

    if simba_p is None or mint_p is None:
        return
    if len(simba_p["true"]) != len(mint_p["true"]):
        print(f"  [simba_vs_mint permutation test] SKIPPED -- row count mismatch "
              f"({len(simba_p['true'])} vs {len(mint_p['true'])})")
        return
    import numpy as np
    if not np.allclose(simba_p["true"], mint_p["true"], atol=1e-6):
        print("  [simba_vs_mint permutation test] SKIPPED -- 'true' arrays don't align row-for-row "
              "(different iteration order); rerun both from the same environment to align them")
        return
    for metric in ("pearson", "spearman", "rmse"):
        pt = permutation_test_paired_metric(simba_p["true"], simba_p["pred"], mint_p["pred"],
                                             metric=metric, n_perm=N_PERM)
        print(f"  [simba_vs_mint permutation test, {metric}] observed_gap={pt['observed_gap']:.4f} "
              f"p={pt['p_value']:.4f} (n={pt['n']}, n_perm={pt['n_perm']})")
        out.setdefault("simba_vs_mint_permutation_test", {})[metric] = pt


def analyze_plm_ablation_cold(out):
    print("\n=== Stage 9: PLM backbone ablation (cold_target split) ===")
    for backbone in ("esm2", "ablang2", "mint"):
        d = _load(f"plm_ablation_cold_{backbone}_results.json")
        if d is None or "predictions" not in d:
            print(f"  [plm_ablation_cold_{backbone}] MISSING or no predictions, skipped")
            continue
        p = d["predictions"]
        _report_pooled(f"plm_ablation_cold_{backbone}", p["ids"], p["true"], p["pred"], out)


def analyze_novel_generalization(out):
    print("\n=== Stage 10: zero-shot / few-shot generalization to novel complexes ===")
    d = _load("novel_generalization_results.json")
    if d is not None and "zero_shot_preds" in d:
        true = d["true"]
        for key, label in (("zero_shot_preds", "novel_gen_zero_shot"), ("few_shot_preds", "novel_gen_few_shot")):
            m = bootstrap_metrics_ci(true, d[key], n_boot=N_BOOT)
            print(f"  [{label}] n={m['n']}  Pearson={m['pearson']:.4f} "
                  f"[{m['pearson_ci_lo']:.4f},{m['pearson_ci_hi']:.4f}]  RMSE={m['rmse']:.4f}")
            out[label] = {"pooled": m}
    else:
        print("  [novel_generalization_results.json] MISSING, skipped")

    d2 = _load("novel_generalization_fewshot_scaling_results.json")
    if d2 is not None and "predictions" in d2:
        p = d2["predictions"]
        true = p["true"]
        m = bootstrap_metrics_ci(true, p["zeroshot_pred"], n_boot=N_BOOT)
        print(f"  [novel_gen_scaling_zeroshot] n={m['n']}  Pearson={m['pearson']:.4f} RMSE={m['rmse']:.4f}")
        out["novel_gen_scaling_zeroshot"] = {"pooled": m}
        for frac, preds in p["fewshot_pred_by_fraction"].items():
            m = bootstrap_metrics_ci(true, preds, n_boot=N_BOOT)
            print(f"  [novel_gen_scaling_frac{frac}] n={m['n']}  Pearson={m['pearson']:.4f} RMSE={m['rmse']:.4f}")
            out[f"novel_gen_scaling_frac{frac}"] = {"pooled": m}
    else:
        print("  [novel_generalization_fewshot_scaling_results.json] MISSING, skipped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="all_results_with_ci.json")
    parser.parse_args()

    out: dict = {}
    analyze_skempi2_5fold_matrix(out)
    analyze_fisher(out)
    analyze_affinity_extended(out)
    analyze_ppi_graphomer_test1(out)
    analyze_ppi_affinity_test2(out)
    analyze_ddg_multiseed(out)
    analyze_mint_comparison(out)
    analyze_plm_ablation_cold(out)
    analyze_novel_generalization(out)

    with open("all_results_with_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved consolidated bootstrap-CI/permutation-test summary to all_results_with_ci.json")


if __name__ == "__main__":
    main()
