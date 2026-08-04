"""Fisher-z / intra-target correlation analysis for SIMBA's pairformer on
the combined pool, complex ("random") split -- 5-fold CV, pooling every
fold's held-out predictions so every one of the 7,247 examples gets exactly
one out-of-fold prediction.

Motivation: a single pooled (overall) Pearson correlation over all examples
can be dominated by *between-target* variance (some binding targets simply
have much higher/lower absolute affinity than others) rather than reflecting
whether the model correctly ranks *within* a given target's own variants --
the same "pooled vs. per-complex" distinction this project has already
established matters for RDE-PPI (S3.5's "overall" vs. "per-structure"
metrics). Here "target" = the partner sequence (the binding target a set of
self-chain variants is measured against), grouping is at that level rather
than at the complex level, giving one intra-target correlation per unique
partner sequence with enough repeated measurements to compute one.

Averaging correlation coefficients naively (arithmetic mean of r) is a known
statistical bias (Fisher, 1915): the sampling distribution of r is skewed,
especially for r far from 0 or with small n. The unbiased approach is
Fisher's z-transformation: z = arctanh(r), average z across targets, then
back-transform r_fisher = tanh(mean(z)). We report both the naive mean and
the Fisher-corrected mean for direct comparison. Spearman has no exact
analogue of Fisher's transform, but the same z=arctanh/tanh procedure is a
standard, widely-used approximation for averaging Spearman coefficients
too, and is reported the same way here for consistency.

This is a pure post-hoc analysis over already-trained predictions -- it
used to retrain the exact same pairformer/combined-pool/complex-split model
from scratch (identical protocol to `eval_skempi2_5fold.py --pool combined
--split complex --model pairformer`), which is redundant GPU work now that
that script always dumps per-example predictions. Reads
`skempi2_5fold_combined_pairformer.predictions.json` instead; run that
script first if it isn't there yet.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

MIN_TARGET_N = 5  # minimum examples sharing a partner sequence to compute an intra-target correlation
OUT_JSON = "figures/fisher_intra_target_results.json"
DEFAULT_PREDICTIONS = "skempi2_5fold_combined_pairformer.predictions.json"


def fisher_average(rs: list[float]) -> float:
    zs = [math.atanh(max(-0.999999, min(0.999999, r))) for r in rs]
    return math.tanh(float(np.mean(zs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                         help="Per-example predictions dump from eval_skempi2_5fold.py "
                              "(--pool combined --split complex --model pairformer).")
    parser.add_argument("--out", default=OUT_JSON)
    args = parser.parse_args()

    with open(args.predictions) as f:
        records = json.load(f)
    print(f"[fisher] loaded {len(records)} records from {args.predictions}", flush=True)

    all_records = [(r["partner_sequence"], r["complex_id"].upper(), r["dG_true"], r["dG_pred"])
                   for r in records]

    all_true = np.array([r[2] for r in all_records])
    all_pred = np.array([r[3] for r in all_records])
    all_ids = [r[1] for r in all_records]
    overall_pearson = float(pearsonr(all_true, all_pred)[0])
    overall_spearman = float(spearmanr(all_true, all_pred)[0])
    overall_rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
    print(f"\n[fisher] overall pooled Pearson={overall_pearson:.4f} Spearman={overall_spearman:.4f} "
          f"RMSE={overall_rmse:.4f}", flush=True)

    by_target: dict[str, list[tuple[float, float]]] = {}
    for partner_seq, _cid, t, p in all_records:
        by_target.setdefault(partner_seq, []).append((t, p))

    target_pearsons, target_spearmans, target_ns = [], [], []
    for partner_seq, pairs in by_target.items():
        if len(pairs) < MIN_TARGET_N:
            continue
        t = np.array([x[0] for x in pairs])
        p = np.array([x[1] for x in pairs])
        if np.std(t) == 0 or np.std(p) == 0:
            continue
        r_p = pearsonr(t, p)[0]
        r_s = spearmanr(t, p)[0]
        if math.isnan(r_p) or math.isnan(r_s):
            continue
        target_pearsons.append(float(r_p))
        target_spearmans.append(float(r_s))
        target_ns.append(len(pairs))

    naive_mean_pearson = float(np.mean(target_pearsons))
    fisher_pearson = fisher_average(target_pearsons)
    naive_mean_spearman = float(np.mean(target_spearmans))
    fisher_spearman = fisher_average(target_spearmans)

    print(f"[fisher] {len(target_pearsons)} targets with >= {MIN_TARGET_N} examples "
          f"(out of {len(by_target)} unique partner sequences)", flush=True)
    print(f"[fisher] intra-target Pearson: naive mean={naive_mean_pearson:.4f}, "
          f"Fisher-z mean={fisher_pearson:.4f}", flush=True)
    print(f"[fisher] intra-target Spearman: naive mean={naive_mean_spearman:.4f}, "
          f"Fisher-z mean={fisher_spearman:.4f}", flush=True)

    result = {
        "overall_pearson": overall_pearson, "overall_spearman": overall_spearman,
        "overall_rmse": overall_rmse,
        "n_total": len(all_records),
        "n_targets_total": len(by_target), "n_targets_used": len(target_pearsons),
        "min_target_n": MIN_TARGET_N,
        "target_pearsons": target_pearsons, "target_spearmans": target_spearmans, "target_ns": target_ns,
        "naive_mean_pearson": naive_mean_pearson, "fisher_mean_pearson": fisher_pearson,
        "naive_mean_spearman": naive_mean_spearman, "fisher_mean_spearman": fisher_spearman,
        "scatter_true": all_true.tolist(), "scatter_pred": all_pred.tolist(), "scatter_ids": all_ids,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fisher] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
