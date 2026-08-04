"""Evaluate SIMBA under RDE-PPI's exact reproduced 3-fold complex-level
SKEMPI2 CV (see simba/data/build_rde_skempi.py), so the resulting numbers are
directly comparable to RDE-PPI's own published Table 1 -- same rows, same
folds, same metric definitions (pooled "Overall" correlation across all test
folds combined, and "Per-Structure" correlation averaged over complexes with
>=10 mutations, discarding smaller groups exactly as they do).

Published reference numbers (RDE-PPI, ICLR 2023, Table 1, verified from the
paper text) on this exact split:

                    Per-Struct.         Overall
                    Pearson  Spearman   Pearson  Spearman
  ESM-1v (0-shot)    0.0073  -0.0118    0.1921   0.1572
  FoldX              0.3789   0.3693    0.3120   0.4071
  RDE-Network        0.4448   0.4010    0.6447   0.5584   <- current best in that table
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from simba import env

env.setup()

from simba.data.dataset import load_mint_skempi, load_rde_skempi  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import run_training  # noqa: E402

REFERENCE = {
    "ESM-1v (0-shot)": {"per_struct_pearson": 0.0073, "per_struct_spearman": -0.0118,
                        "overall_pearson": 0.1921, "overall_spearman": 0.1572},
    "FoldX": {"per_struct_pearson": 0.3789, "per_struct_spearman": 0.3693,
              "overall_pearson": 0.3120, "overall_spearman": 0.4071},
    "RDE-Network": {"per_struct_pearson": 0.4448, "per_struct_spearman": 0.4010,
                    "overall_pearson": 0.6447, "overall_spearman": 0.5584},
}


def make_val_split(train_full, seed: int, val_frac: float = 0.1):
    rng = random.Random(seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val_idx = set(idx[:n_val])
    train = [e for i, e in enumerate(train_full) if i not in val_idx]
    val = [e for i, e in enumerate(train_full) if i in val_idx]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-masked-marginal", action="store_true")
    parser.add_argument("--core", choices=["static", "recurrent"], default="static")
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--split", choices=["rde", "mint"], default="rde",
                         help="rde: RDE-PPI's exact split (seed=2022, the original protocol this "
                              "script was built for). mint: same rows/complexes, refolded with "
                              "MINT's own SkempiDataset split_seed=2023, for a fair head-to-head "
                              "against MINT's own reported SKEMPI ddG protocol (MODEL_REPORT.md S3.21).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_mint_skempi() if args.split == "mint" else load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    print(f"total examples: {len(examples)}, folds: {folds}, core={args.core}")

    scorer = ESMScorer(model_name=args.backbone, device=device)
    cache = SequenceScoreCache(scorer)
    masked_cache = None if args.no_masked_marginal else MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    all_complex_ids, all_y_true, all_y_pred = [], [], []

    for fold in folds:
        print(f"\n=== fold {fold} (held out as test) ===")
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        print(f"train={len(train)} val={len(val)} test={len(test)}")

        model, best_val_score, test_report, _ = run_training(
            train, val, test,
            provider=provider, plm_hidden_size=scorer.model.config.hidden_size,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, device=device, seed=fold, verbose=True,
            model_kwargs={"core": args.core, "n_rounds": args.n_rounds},
        )

        model.eval()
        with torch.no_grad():
            for ex in test:
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                pred = model(sites).item()
                all_complex_ids.append(ex.complex_id)
                all_y_true.append(ex.ddG)
                all_y_pred.append(pred)

    print("\n" + "=" * 60)
    print("AGGREGATE OVER ALL 3 FOLDS (each mutation tested exactly once)")
    print("=" * 60)
    overall = pooled_metrics(all_y_true, all_y_pred)
    per_struct = per_complex_metrics(all_complex_ids, all_y_true, all_y_pred, min_n=10)
    print(f"Overall     Pearson={overall['pearson']:.4f}  Spearman={overall['spearman']:.4f}  n={overall['n']}")
    print(f"Per-Struct  Pearson={per_struct['mean_pearson']:.4f}  Spearman={per_struct['mean_spearman']:.4f} "
          f"(scored {per_struct['n_complexes_scored']}/{per_struct['n_complexes_total']} complexes, >=10 muts)")

    print("\n--- comparison to published RDE-PPI Table 1 ---")
    print(f"{'Method':<20}{'Per-Struct P':>14}{'Per-Struct S':>14}{'Overall P':>12}{'Overall S':>12}")
    for name, r in REFERENCE.items():
        print(f"{name:<20}{r['per_struct_pearson']:>14.4f}{r['per_struct_spearman']:>14.4f}"
              f"{r['overall_pearson']:>12.4f}{r['overall_spearman']:>12.4f}")
    print(f"{'SIMBA (ours)':<20}{per_struct['mean_pearson']:>14.4f}{per_struct['mean_spearman']:>14.4f}"
          f"{overall['pearson']:>12.4f}{overall['spearman']:>12.4f}")


if __name__ == "__main__":
    main()
