"""Evaluate SIMBA under ProtAttBA's own exact released CV folds (AB645:
10-fold, AB1101: 5-fold, S1131: 10-fold) and diff directly against their real
per-row predictions on the identical held-out rows -- the strongest possible
apples-to-apples comparison against a sequence-only peer: same data, same
folds, their actual predictions, no retraining or re-running of their code.
"""

from __future__ import annotations

import argparse

import torch

from simba import env

env.setup()

from simba.data.dataset import load_protattba  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import run_training  # noqa: E402

BENCHMARKS = ["ab645", "ab1101", "s1131"]


def run_benchmark(name, provider, scorer, epochs, batch_size, lr, weight_decay, device, model_kwargs=None):
    examples = load_protattba(name)
    folds = sorted({e.meta["fold"] for e in examples})
    print(f"\n### {name.upper()} -- {len(examples)} rows, {len(folds)} folds (ProtAttBA's own split) ###")

    pdb_ids, y_true, y_pred_simba, y_pred_protattba = [], [], [], []

    for fold in folds:
        test = [e for e in examples if e.meta["fold"] == fold]
        train_full = [e for e in examples if e.meta["fold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        print(f"  fold {fold}: train={len(train)} val={len(val)} test={len(test)}")

        model, _, _, _ = run_training(
            train, val, test,
            provider=provider, plm_hidden_size=scorer.model.config.hidden_size,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            device=device, seed=fold, verbose=False, model_kwargs=model_kwargs,
        )

        model.eval()
        with torch.no_grad():
            for ex in test:
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                pdb_ids.append(ex.meta.get("pdb_id", ex.complex_id))
                y_true.append(ex.ddG)
                y_pred_simba.append(model(sites).item())
                y_pred_protattba.append(ex.meta["protattba_prediction"])

    simba_overall = pooled_metrics(y_true, y_pred_simba)
    simba_per_struct = per_complex_metrics(pdb_ids, y_true, y_pred_simba, min_n=3)
    prot_overall = pooled_metrics(y_true, y_pred_protattba)
    prot_per_struct = per_complex_metrics(pdb_ids, y_true, y_pred_protattba, min_n=3)

    print(f"  SIMBA      overall pearson={simba_overall['pearson']:.4f} spearman={simba_overall['spearman']:.4f} "
          f"| per-complex mean spearman={simba_per_struct['mean_spearman']:.4f} "
          f"({simba_per_struct['n_complexes_scored']}/{simba_per_struct['n_complexes_total']} complexes)")
    print(f"  ProtAttBA  overall pearson={prot_overall['pearson']:.4f} spearman={prot_overall['spearman']:.4f} "
          f"| per-complex mean spearman={prot_per_struct['mean_spearman']:.4f} "
          f"({prot_per_struct['n_complexes_scored']}/{prot_per_struct['n_complexes_total']} complexes)  "
          f"[on the exact same {len(y_true)} held-out rows]")

    return {
        "n": len(y_true), "simba_overall": simba_overall, "simba_per_struct": simba_per_struct,
        "protattba_overall": prot_overall, "protattba_per_struct": prot_per_struct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    parser.add_argument("--backbone", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--core", choices=["static", "recurrent"], default="static")
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--chain-mode", choices=["dual_memory", "concat", "self_attn", "none", "pairformer"], default="dual_memory")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    cache = SequenceScoreCache(scorer)
    masked_cache = MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)
    model_kwargs = {"core": args.core, "n_rounds": args.n_rounds, "chain_mode": args.chain_mode}

    results = {}
    for name in args.benchmarks:
        results[name] = run_benchmark(
            name, provider, scorer, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
            model_kwargs=model_kwargs,
        )

    print("\n" + "=" * 78)
    print("SUMMARY -- SIMBA vs ProtAttBA's real predictions, identical held-out rows")
    print("=" * 78)
    print(f"{'Benchmark':<10}{'n':>6}{'SIMBA overall S':>18}{'ProtAttBA overall S':>22}"
          f"{'SIMBA per-cplx S':>18}{'ProtAttBA per-cplx S':>22}")
    for name, r in results.items():
        print(f"{name:<10}{r['n']:>6}{r['simba_overall']['spearman']:>18.4f}"
              f"{r['protattba_overall']['spearman']:>22.4f}"
              f"{r['simba_per_struct']['mean_spearman']:>18.4f}"
              f"{r['protattba_per_struct']['mean_spearman']:>22.4f}")


if __name__ == "__main__":
    main()
