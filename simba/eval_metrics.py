"""Evaluation metrics shared by all training/eval scripts.

Per-complex Spearman (averaged across complexes) is reported alongside the
pooled correlation because pooled correlation on SKEMPI2-style benchmarks is
known to be dominated by the handful of complexes with the most mutations --
a per-complex average is the more honest generalization signal.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pooled_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": float("nan"), "n": len(y_true)}
    return {
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": len(y_true),
    }


def per_complex_metrics(
    complex_ids: list[str], y_true: list[float], y_pred: list[float], min_n: int = 3
) -> dict:
    by_complex: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for cid, yt, yp in zip(complex_ids, y_true, y_pred):
        by_complex[cid][0].append(yt)
        by_complex[cid][1].append(yp)

    spearmans, pearsons = [], []
    for cid, (yt, yp) in by_complex.items():
        if len(yt) < min_n or np.std(yt) == 0 or np.std(yp) == 0:
            continue
        r_s, r_p = spearmanr(yt, yp)[0], pearsonr(yt, yp)[0]
        # A near-constant array (e.g. floating-point noise leaving std
        # ~1e-16 instead of exactly 0) slips past the check above but still
        # yields an undefined/NaN correlation from scipy -- guard against
        # a single such complex silently poisoning the mean via np.mean.
        if np.isnan(r_s) or np.isnan(r_p):
            continue
        spearmans.append(r_s)
        pearsons.append(r_p)

    return {
        "mean_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "mean_pearson": float(np.mean(pearsons)) if pearsons else float("nan"),
        "n_complexes_scored": len(spearmans),
        "n_complexes_total": len(by_complex),
    }


def generalization_gap(in_dist_spearman: float, out_dist_spearman: float) -> float:
    return in_dist_spearman - out_dist_spearman


def full_report(
    complex_ids: list[str], y_true: list[float], y_pred: list[float], min_complex_n: int = 3
) -> dict:
    return {
        "pooled": pooled_metrics(y_true, y_pred),
        "per_complex": per_complex_metrics(complex_ids, y_true, y_pred, min_n=min_complex_n),
    }
