"""Backbone-vs-head cross comparison on the MINT-protocol split (reviewer
question): the paper's Section 5.3 comparison is SIMBA (ESM-2 backbone +
Pairformer head) vs. MINT's own reported method (MINT backbone + Ridge
head) -- a nonlinear head on one backbone vs. a linear head on a different
backbone, so the gap conflates backbone and head. This script fills in the
other two cells of that 2x2, on the *same* MINT-protocol split (343
complexes, seed 2023, simba/data/dataset.py's load_mint_skempi):

  --config simba_mint_backbone   SIMBA's Pairformer architecture, but with
                                  frozen MINT embeddings instead of ESM-2.
  --config esm2_ridge            MINT's own readout recipe (mean-pool per
                                  chain, concatenate, wildtype-minus-mutant
                                  difference, Ridge with inner-3-fold alpha
                                  search -- see eval_mint_own_method_skempi.py)
                                  but with frozen ESM-2 embeddings instead of
                                  MINT.

Existing results for the other two cells: SIMBA+ESM-2 is
simba_vs_mint_skempi_split_results.json (Section 5.3); MINT+Ridge is
mint_own_method_skempi_results.json (eval_mint_own_method_skempi.py).
"""

from __future__ import annotations

import os

# expandable_segments is a Linux-only CUDA allocator feature; leaving it set
# (e.g. inherited from a shell that previously ran eval_plm_ablation_cold.py)
# only produces a noisy UserWarning on Windows PyTorch builds. Must be set
# before `import torch` to take effect.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import argparse
import copy
import json
import math
import random

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_mint_skempi  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.mint_plm import HIDDEN as MINT_HIDDEN, MintPairCache, MINTPairScorer  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# --- config: simba_mint_backbone -----------------------------------------

def mint_pair_embed(cache, self_seq, partner_seq, memo):
    key = (self_seq, partner_seq)
    if key not in memo:
        se, pe = cache.get(self_seq, partner_seq)
        memo[key] = (torch.from_numpy(se).float(), torch.from_numpy(pe).float())
    return memo[key]


def predict_simba_mint(model, ex, cache, device, memo):
    wt_full, mut_full, partner_seq = full_wt_mut_sequences(ex)
    wt_se, wt_pe = mint_pair_embed(cache, wt_full, partner_seq, memo)
    mut_se, mut_pe = mint_pair_embed(cache, mut_full, partner_seq, memo)
    dg_wt = model(wt_se.to(device), wt_pe.to(device))
    dg_mut = model(mut_se.to(device), mut_pe.to(device))
    return dg_mut - dg_wt


def run_training_simba_mint(train, val, *, hidden, epochs, batch_size, lr, weight_decay, device, seed, cache):
    torch.manual_seed(seed)
    memo: dict = {}
    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_score, best_state, global_step = -math.inf, None, 0

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, lr)
            valid = [ex for ex in batch if ex.ddG is not None]
            if not valid:
                global_step += 1
                continue
            optimizer.zero_grad()
            n_valid = len(valid)
            for ex in valid:
                pred = predict_simba_mint(model, ex, cache, device, memo)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                ((pred - target) ** 2 / n_valid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        model.eval()
        val_ids, val_true, val_pred = [], [], []
        with torch.no_grad():
            for ex in val:
                if ex.ddG is None:
                    continue
                val_ids.append(ex.complex_id)
                val_true.append(ex.ddG)
                val_pred.append(predict_simba_mint(model, ex, cache, device, memo).item())
        val_report = per_complex_metrics(val_ids, val_true, val_pred, min_n=10)
        score = val_report["mean_spearman"]
        if math.isnan(score):
            score = pooled_metrics(val_true, val_pred)["spearman"]
        is_best = not math.isnan(score) and score > best_score
        if is_best:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(f"[simba_mint_backbone] epoch {epoch}: val_score={score:.4f}{'  <- new best' if is_best else ''}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_simba_mint_backbone(epochs, batch_size, lr, weight_decay, device):
    examples = load_mint_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    scorer = MINTPairScorer(device=device)
    cache = MintPairCache(scorer)
    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        memo: dict = {}
        print(f"\n=== [simba_mint_backbone] fold {fold} ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model = run_training_simba_mint(train, val, hidden=MINT_HIDDEN, epochs=epochs, batch_size=batch_size,
                                         lr=lr, weight_decay=weight_decay, device=device, seed=fold, cache=cache)
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pred = predict_simba_mint(model, ex, cache, device, memo).item()
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(pred)
    return {"overall": pooled_metrics(all_true, all_pred),
            "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
            "predictions": {"ids": all_ids, "true": all_true, "pred": all_pred}}


# --- config: esm2_ridge ----------------------------------------------------

def pooled_complex_embedding_esm2(cache, self_seq, partner_seq, device, memo):
    key = (self_seq, partner_seq)
    if key not in memo:
        self_score = cache.get(self_seq).embedding.astype(np.float32).mean(axis=0)
        partner_score = (cache.get(partner_seq).embedding.astype(np.float32).mean(axis=0)
                          if partner_seq else np.zeros_like(self_score))
        memo[key] = np.concatenate([self_score, partner_score])
    return memo[key]


def build_features_esm2(examples, cache, memo):
    X, y, ids = [], [], []
    for ex in examples:
        if ex.ddG is None:
            continue
        wt_full, mut_full, partner = full_wt_mut_sequences(ex)
        wt_feat = pooled_complex_embedding_esm2(cache, wt_full, partner, None, memo)
        mut_feat = pooled_complex_embedding_esm2(cache, mut_full, partner, None, memo)
        X.append(wt_feat - mut_feat)
        y.append(ex.ddG)
        ids.append(ex.complex_id)
    return np.stack(X), np.array(y, dtype=np.float32), ids


def run_esm2_ridge(device):
    examples = load_mint_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    cache = SequenceScoreCache(scorer)
    memo: dict = {}

    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        print(f"\n=== [esm2_ridge] fold {fold} ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train = [e for e in examples if e.meta["cvfold"] != fold]

        X_train, y_train, _ = build_features_esm2(train, cache, memo)
        X_test, y_test, ids_test = build_features_esm2(test, cache, memo)
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train_n, X_test_n = (X_train - mu) / sigma, (X_test - mu) / sigma

        grid = GridSearchCV(Ridge(), {"alpha": RIDGE_ALPHAS}, cv=3, scoring="neg_mean_squared_error")
        grid.fit(X_train_n, y_train)
        print(f"[fold {fold}] best alpha={grid.best_params_['alpha']}", flush=True)
        preds = grid.best_estimator_.predict(X_test_n)
        all_ids.extend(ids_test)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())

    return {"overall": pooled_metrics(all_true, all_pred),
            "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
            "predictions": {"ids": all_ids, "true": all_true, "pred": all_pred}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["simba_mint_backbone", "esm2_ridge", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="mint_protocol_cross_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    if args.config in ("simba_mint_backbone", "both"):
        r = run_simba_mint_backbone(args.epochs, args.batch_size, args.lr, args.weight_decay, device)
        print(f"\n[simba_mint_backbone] Overall P/S={r['overall']['pearson']:.4f}/{r['overall']['spearman']:.4f}  "
              f"Per-Struct P/S={r['per_struct']['mean_pearson']:.4f}/{r['per_struct']['mean_spearman']:.4f}",
              flush=True)
        results["simba_mint_backbone"] = r

    if args.config in ("esm2_ridge", "both"):
        r = run_esm2_ridge(device)
        print(f"\n[esm2_ridge] Overall P/S={r['overall']['pearson']:.4f}/{r['overall']['spearman']:.4f}  "
              f"Per-Struct P/S={r['per_struct']['mean_pearson']:.4f}/{r['per_struct']['mean_spearman']:.4f}",
              flush=True)
        results["esm2_ridge"] = r

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
