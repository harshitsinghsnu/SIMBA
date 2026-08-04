"""Reproduces MINT's *own* reported SKEMPI ddG-prediction method (Ullanat et
al.) -- not SIMBA-with-a-MINT-backbone (that's the separate cold_target
ablation, MODEL_REPORT.md S3.20) -- for a direct SIMBA-vs-MINT comparison on
the same rows (MODEL_REPORT.md S3.21).

Their method, reverse-engineered from `third_party/mint/downstream/
GeneralPPI/{tasks.py,embeddings_mint.py,finetune_general.py}`:
  1. Frozen MINT embeddings, mean-pooled per chain (self, partner) and
     concatenated (`sep_chains=True` -- their own published usage example),
     for the wildtype complex and the mutant complex separately.
  2. Feature = wt_embedding - mut_embedding (their `SKEMPI_ESMMultimerWrapper
     .forward`, default `cat=False`).
  3. Ridge regression (`return_ridge_model` in their `finetune_general.py`),
     alpha tuned via inner 3-fold GridSearchCV, on a pre-defined complex
     -level train/test split (their `pre_defined_cv`).

Split: MINT's own `SkempiDataset` protocol (3-fold, complex-level,
split_seed=2023) is used here rather than their actual headline-number split
(built from an *unseeded* `np.random.shuffle` in their notebook, and
therefore not reproducible from their published code at all -- see
simba/data/build_mint_skempi_split.py). Same rows/complexes SIMBA is
evaluated on via `eval_ddg_affinity.py --benchmarks mint_split`.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_mint_skempi  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.features.mint_plm import MINTPairScorer, MintPairCache  # noqa: E402

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def pooled_complex_embedding(cache, self_seq, partner_seq, memo):
    """Mean-pooled self + partner embeddings, concatenated
    (sep_chains=True convention) -- MINT's own published usage pattern."""
    key = (self_seq, partner_seq)
    if key not in memo:
        se, pe = cache.get(self_seq, partner_seq)
        self_mean = se.astype(np.float32).mean(axis=0)
        partner_mean = pe.astype(np.float32).mean(axis=0)
        memo[key] = np.concatenate([self_mean, partner_mean])
    return memo[key]


def build_features(examples, cache, memo):
    X, y, ids = [], [], []
    for ex in examples:
        if ex.ddG is None:
            continue
        wt_full, mut_full, partner = full_wt_mut_sequences(ex)
        wt_feat = pooled_complex_embedding(cache, wt_full, partner, memo)
        mut_feat = pooled_complex_embedding(cache, mut_full, partner, memo)
        X.append(wt_feat - mut_feat)
        y.append(ex.ddG)
        ids.append(ex.complex_id)
    return np.stack(X), np.array(y, dtype=np.float32), ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_mint_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    print(f"[mint_own_method] {len(examples)} examples, folds={folds}", flush=True)

    scorer = MINTPairScorer(device=device)
    cache = MintPairCache(scorer)
    memo: dict = {}

    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train = [e for e in examples if e.meta["cvfold"] != fold]
        print(f"\n=== MINT-own-method fold {fold}: train={len(train)} test={len(test)} ===", flush=True)

        X_train, y_train, _ = build_features(train, cache, memo)
        X_test, y_test, ids_test = build_features(test, cache, memo)

        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mu) / sigma
        X_test_n = (X_test - mu) / sigma

        grid = GridSearchCV(Ridge(), {"alpha": RIDGE_ALPHAS}, cv=3,
                             scoring="neg_mean_squared_error")
        grid.fit(X_train_n, y_train)
        best = grid.best_estimator_
        print(f"[fold {fold}] best alpha={grid.best_params_['alpha']}", flush=True)

        preds = best.predict(X_test_n)
        all_ids.extend(ids_test)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    overall = pooled_metrics(all_true, all_pred)
    per_struct = per_complex_metrics(all_ids, all_true, all_pred, min_n=10)
    print(f"\nOverall     Pearson={overall['pearson']:.4f}  Spearman={overall['spearman']:.4f}  n={overall['n']}")
    print(f"Per-Struct  Pearson={per_struct['mean_pearson']:.4f}  Spearman={per_struct['mean_spearman']:.4f} "
          f"(scored {per_struct['n_complexes_scored']}/{per_struct['n_complexes_total']} complexes, >=10 muts)")

    result = {
        "overall": overall, "per_struct": per_struct,
        "predictions": {"ids": all_ids, "true": all_true, "pred": all_pred},
    }
    with open("mint_own_method_skempi_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[mint_own_method] saved mint_own_method_skempi_results.json", flush=True)


if __name__ == "__main__":
    main()
