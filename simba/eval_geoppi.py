"""Evaluate SIMBA on GeoPPI's own general-PPI benchmarks (not antibody
-specific), diffed against MutaBind2's real predictions where available.

GeoPPI/MutaBind2 never released fold assignments for M1707, so this builds
its own deterministic leave-complex-out folds (same spirit as RDE-PPI's
reproduced split: shuffle the sorted unique-complex list with a fixed seed,
slice into N contiguous folds) -- not their exact split, but rigorously
leakage-free, and the MutaBind2 comparison is valid regardless of how we
split since we're diffing against their real, already-computed predictions
on whichever rows land in our test folds.
"""

from __future__ import annotations

import argparse
import random

import torch

from simba import env

env.setup()

from simba.data.dataset import load_geoppi_m1707, load_geoppi_s4169  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import run_training  # noqa: E402


def build_complex_folds(examples, n_folds: int, seed: int = 0) -> dict[str, int]:
    complexes = sorted({e.complex_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(complexes)
    fold_size = -(-len(complexes) // n_folds)  # ceil
    return {c: i // fold_size for i, c in enumerate(complexes)}


def run_benchmark(name, examples, provider, hidden, epochs, batch_size, lr, weight_decay,
                   device, n_folds, chain_mode, has_mutabind2):
    fold_of = build_complex_folds(examples, n_folds)
    for ex in examples:
        ex.meta["_fold"] = fold_of[ex.complex_id]

    print(f"\n### {name} -- {len(examples)} rows, {len(set(fold_of.values()))} folds "
          f"({len(fold_of)} unique complexes) ###")

    ids, y_true, y_pred, y_mutabind2 = [], [], [], []
    for fold in sorted(set(fold_of.values())):
        test = [e for e in examples if e.meta["_fold"] == fold]
        train_full = [e for e in examples if e.meta["_fold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        if not test or not train:
            continue
        model, *_ = run_training(
            train, val, test, provider=provider, plm_hidden_size=hidden,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            device=device, seed=fold, verbose=False, model_kwargs={"chain_mode": chain_mode},
        )
        model.eval()
        with torch.no_grad():
            for ex in test:
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                ids.append(ex.complex_id)
                y_true.append(ex.ddG)
                y_pred.append(model(sites).item())
                if has_mutabind2:
                    y_mutabind2.append(ex.meta.get("mutabind2_prediction"))

    overall = pooled_metrics(y_true, y_pred)
    per_struct = per_complex_metrics(ids, y_true, y_pred, min_n=3)
    print(f"  SIMBA  overall pearson={overall['pearson']:.4f} spearman={overall['spearman']:.4f} "
          f"| per-complex mean pearson={per_struct['mean_pearson']:.4f} "
          f"({per_struct['n_complexes_scored']}/{per_struct['n_complexes_total']} complexes) n={len(y_true)}")

    result = {"overall": overall, "per_struct": per_struct, "n": len(y_true)}
    if has_mutabind2:
        valid = [i for i, v in enumerate(y_mutabind2) if v is not None]
        mb_true = [y_true[i] for i in valid]
        mb_pred = [y_mutabind2[i] for i in valid]
        mb_ids = [ids[i] for i in valid]
        mb_overall = pooled_metrics(mb_true, mb_pred)
        mb_per_struct = per_complex_metrics(mb_ids, mb_true, mb_pred, min_n=3)
        print(f"  MutaBind2 (real, precomputed) overall pearson={mb_overall['pearson']:.4f} "
              f"| per-complex mean pearson={mb_per_struct['mean_pearson']:.4f} "
              f"({mb_per_struct['n_complexes_scored']}/{mb_per_struct['n_complexes_total']} complexes) n={len(mb_true)}")
        result["mutabind2_overall"] = mb_overall
        result["mutabind2_per_struct"] = mb_per_struct
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["m1707"], choices=["m1707", "s4169"])
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--chain-mode", default="concat", choices=["dual_memory", "concat", "self_attn", "none", "pairformer"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)
    masked_cache = MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    if "m1707" in args.benchmarks:
        examples = load_geoppi_m1707()
        run_benchmark("M1707", examples, provider, hidden, args.epochs, args.batch_size,
                      args.lr, args.weight_decay, device, args.n_folds, args.chain_mode,
                      has_mutabind2=True)
    if "s4169" in args.benchmarks:
        examples = load_geoppi_s4169()
        run_benchmark("S4169", examples, provider, hidden, args.epochs, args.batch_size,
                      args.lr, args.weight_decay, device, args.n_folds, args.chain_mode,
                      has_mutabind2=False)


if __name__ == "__main__":
    main()
