"""Shared bootstrap-CI and permutation-test utilities for post-hoc analysis
of saved per-example predictions (true/pred pairs, optionally grouped by
complex_id). Deliberately kept separate from `simba/eval_metrics.py`
(point-estimate metrics only, used inline during training/eval loops) --
resampling is only ever done once, after the fact, from saved predictions
JSON, never during training.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": float("nan")}
    return {
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def bootstrap_metrics_ci(
    y_true: list[float], y_pred: list[float], *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict:
    """Pearson/Spearman/RMSE point estimates plus percentile bootstrap CIs,
    resampling examples with replacement. Use this for any pooled metric
    reported with an "n=" in the paper/report -- it directly answers "how
    much would this number move under a different sample of the same
    population."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    point = _point_metrics(y_true, y_pred)

    rng = np.random.default_rng(seed)
    boot = {"pearson": [], "spearman": [], "rmse": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _point_metrics(y_true[idx], y_pred[idx])
        for k in boot:
            if not np.isnan(m[k]):
                boot[k].append(m[k])

    alpha = (1 - ci) / 2
    out = {"n": n, "n_boot": n_boot, "ci": ci}
    for k in ("pearson", "spearman", "rmse"):
        out[k] = point[k]
        if boot[k]:
            lo, hi = np.percentile(boot[k], [100 * alpha, 100 * (1 - alpha)])
            out[f"{k}_ci_lo"] = float(lo)
            out[f"{k}_ci_hi"] = float(hi)
            out[f"{k}_std_boot"] = float(np.std(boot[k]))
        else:
            out[f"{k}_ci_lo"] = out[f"{k}_ci_hi"] = out[f"{k}_std_boot"] = float("nan")
    return out


def bootstrap_per_complex_metrics_ci(
    complex_ids: list[str], y_true: list[float], y_pred: list[float], *,
    min_n: int = 3, n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict:
    """Per-complex mean Pearson/Spearman with a bootstrap CI built by
    resampling *complexes* (not raw rows) with replacement -- the correct
    resampling unit for a "mean over complexes" statistic, since rows within
    one complex aren't independent draws."""
    from collections import defaultdict

    by_complex: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for cid, yt, yp in zip(complex_ids, y_true, y_pred):
        by_complex[cid][0].append(yt)
        by_complex[cid][1].append(yp)

    def per_complex_means(cids):
        pearsons, spearmans = [], []
        for cid in cids:
            yt, yp = by_complex[cid]
            yt, yp = np.asarray(yt), np.asarray(yp)
            if len(yt) < min_n or np.std(yt) == 0 or np.std(yp) == 0:
                continue
            r_p, r_s = pearsonr(yt, yp)[0], spearmanr(yt, yp)[0]
            # Guards against near-constant rows that pass the std==0 check
            # exactly (float noise) but still yield an undefined/NaN
            # correlation -- a single NaN here would otherwise silently
            # poison the mean via np.mean.
            if np.isnan(r_p) or np.isnan(r_s):
                continue
            pearsons.append(r_p)
            spearmans.append(r_s)
        mean_p = float(np.mean(pearsons)) if pearsons else float("nan")
        mean_s = float(np.mean(spearmans)) if spearmans else float("nan")
        return mean_p, mean_s, len(pearsons)

    all_cids = sorted(by_complex.keys())
    point_p, point_s, n_scored = per_complex_means(all_cids)

    rng = np.random.default_rng(seed)
    boot_p, boot_s = [], []
    for _ in range(n_boot):
        sampled = rng.choice(all_cids, size=len(all_cids), replace=True)
        mp, ms, _ = per_complex_means(sampled)
        if not np.isnan(mp):
            boot_p.append(mp)
        if not np.isnan(ms):
            boot_s.append(ms)

    alpha = (1 - ci) / 2
    out = {
        "n_complexes_total": len(all_cids), "n_complexes_scored": n_scored,
        "min_n": min_n, "n_boot": n_boot, "ci": ci,
        "mean_pearson": point_p, "mean_spearman": point_s,
    }
    for name, boot in (("mean_pearson", boot_p), ("mean_spearman", boot_s)):
        if boot:
            lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
            out[f"{name}_ci_lo"] = float(lo)
            out[f"{name}_ci_hi"] = float(hi)
        else:
            out[f"{name}_ci_lo"] = out[f"{name}_ci_hi"] = float("nan")
    return out


def permutation_test_paired_metric(
    y_true: list[float], pred_a: list[float], pred_b: list[float], *,
    metric: str = "pearson", n_perm: int = 5000, seed: int = 0,
) -> dict:
    """Paired permutation test for "does model A actually beat model B on
    the same rows" (e.g. SIMBA vs. MINT's own method, or SIMBA vs. a
    reported baseline column, on identical examples). Null hypothesis:
    A and B are exchangeable per-example predictors of the same truth, so
    randomly relabeling which of (pred_a[i], pred_b[i]) is "A" and which is
    "B" for each example shouldn't systematically change the metric gap.
    Two-sided p-value = fraction of permuted gaps with |gap| >= observed."""
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)

    def score(preds):
        if metric == "pearson":
            return pearsonr(y_true, preds)[0] if np.std(preds) > 0 else float("nan")
        if metric == "spearman":
            return spearmanr(y_true, preds)[0] if np.std(preds) > 0 else float("nan")
        if metric == "rmse":
            return -np.sqrt(np.mean((y_true - preds) ** 2))  # negate: higher = A better, for consistent sign
        raise ValueError(metric)

    observed_gap = score(pred_a) - score(pred_b)

    rng = np.random.default_rng(seed)
    perm_gaps = np.empty(n_perm)
    swap_mask = np.empty((n, 2), dtype=bool)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        a_i = np.where(swap, pred_b, pred_a)
        b_i = np.where(swap, pred_a, pred_b)
        perm_gaps[i] = score(a_i) - score(b_i)

    p_value = float(np.mean(np.abs(perm_gaps) >= abs(observed_gap)))
    return {
        "metric": metric, "n": n, "n_perm": n_perm,
        "observed_gap": float(observed_gap), "p_value": p_value,
    }
