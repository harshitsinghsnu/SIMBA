"""CDR-awareness ablation: train under the AbDesign zero-shot holdout
protocol (as usual), then stratify the held-out predictions by whether the
mutated position falls in a CDR loop (IMGT-numbered CDR1/2/3 boundaries,
already annotated in abdesign_mutations.csv's `is_cdr` column) vs framework.

Only the AB-Bind/SKEMPIv2 ddG subset of the AbDesign eval set has framework
mutations to compare against -- AbDesign's own 658 native mutants are all
CDR-H3 by construction (100% is_cdr==True), so no in-benchmark contrast is
possible there.
"""

from __future__ import annotations

import argparse

import torch

from simba import env

env.setup()

from simba.eval_metrics import full_report  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.data.dataset import load_all_mutations  # noqa: E402
from simba.train import run_training, split_examples  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chain-mode", default="concat", choices=["dual_memory", "concat", "self_attn", "none", "pairformer"])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    print(f"train={len(train)} val={len(val)} eval={len(test)} chain_mode={args.chain_mode}")

    scorer = ESMScorer(model_name=args.backbone, device=device)
    cache = SequenceScoreCache(scorer)
    masked_cache = MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    model, best_val_score, _, _ = run_training(
        train, val, test,
        provider=provider, plm_hidden_size=scorer.model.config.hidden_size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=device, verbose=True,
        model_kwargs={"chain_mode": args.chain_mode},
    )

    model.eval()
    cdr_ids, cdr_true, cdr_pred = [], [], []
    fw_ids, fw_true, fw_pred = [], [], []
    with torch.no_grad():
        for ex in test:
            if ex.ddG is None:
                continue  # restrict this ablation to the ddG-labeled subset (has CDR/framework contrast)
            sites = provider.featurize(ex)
            if any(s["delta_logp"] is None for s in sites):
                continue
            is_cdr = ex.meta.get("is_cdr")
            if is_cdr is None or (isinstance(is_cdr, float) and is_cdr != is_cdr):  # NaN check
                continue
            pred = model(sites).item()
            if bool(is_cdr):
                cdr_ids.append(ex.complex_id); cdr_true.append(ex.ddG); cdr_pred.append(pred)
            else:
                fw_ids.append(ex.complex_id); fw_true.append(ex.ddG); fw_pred.append(pred)

    print(f"\nCDR mutations:       n={len(cdr_true)}")
    print(full_report(cdr_ids, cdr_true, cdr_pred))
    print(f"\nFramework mutations: n={len(fw_true)}")
    print(full_report(fw_ids, fw_true, fw_pred))


if __name__ == "__main__":
    main()
