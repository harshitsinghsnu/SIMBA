"""Full sweep: static core x {concat, none, dual_memory} chain-handling
modes x ESM2-650M + WT/Mut counterfactual representation, evaluated across
every benchmark protocol built so far (RDE-PPI's exact 3-fold split,
ProtAttBA's AB645/AB1101/S1131 exact folds + real predictions, and the
AbDesign zero-shot holdout).

Deliberately scoped down from "every combination": self_attn chain-handling
never won any prior comparison, and the recurrent core already lost decisively
on RDE-PPI, so re-running either across all 5 protocols would spend hours
re-confirming an already-answered question. One shared FeatureProvider (one
loaded backbone, one embedding cache) is reused across every config in the
sweep -- only the small trainable head changes between runs -- so embeddings
already computed for one chain_mode are free for the next.
"""

from __future__ import annotations

import argparse
import json

import torch

from simba import env

env.setup()

from simba.data.dataset import load_all_mutations, load_protattba, load_rde_skempi  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import full_report, per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import run_training, split_examples  # noqa: E402

PROTATTBA_BENCHMARKS = ["ab645", "ab1101", "s1131"]


def eval_rde(provider, hidden, chain_mode, epochs, device):
    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, *_ = run_training(
            train, val, test, provider=provider, plm_hidden_size=hidden,
            epochs=epochs, device=device, seed=fold, verbose=False,
            model_kwargs={"chain_mode": chain_mode},
        )
        model.eval()
        with torch.no_grad():
            for ex in test:
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(model(sites).item())
    return {
        "overall": pooled_metrics(all_true, all_pred),
        "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
    }


def eval_protattba_one(provider, hidden, chain_mode, epochs, name, device):
    examples = load_protattba(name)
    folds = sorted({e.meta["fold"] for e in examples})
    pdb_ids, y_true, y_pred, y_prot = [], [], [], []
    for fold in folds:
        test = [e for e in examples if e.meta["fold"] == fold]
        train_full = [e for e in examples if e.meta["fold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, *_ = run_training(
            train, val, test, provider=provider, plm_hidden_size=hidden,
            epochs=epochs, device=device, seed=fold, verbose=False,
            model_kwargs={"chain_mode": chain_mode},
        )
        model.eval()
        with torch.no_grad():
            for ex in test:
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                pdb_ids.append(ex.meta.get("pdb_id", ex.complex_id))
                y_true.append(ex.ddG)
                y_pred.append(model(sites).item())
                y_prot.append(ex.meta["protattba_prediction"])
    return {
        "overall": pooled_metrics(y_true, y_pred),
        "per_struct": per_complex_metrics(pdb_ids, y_true, y_pred, min_n=3),
        "prot_overall": pooled_metrics(y_true, y_prot),
        "prot_per_struct": per_complex_metrics(pdb_ids, y_true, y_prot, min_n=3),
    }


def eval_abdesign(provider, hidden, chain_mode, epochs, device):
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    model, *_ = run_training(
        train, val, test, provider=provider, plm_hidden_size=hidden,
        epochs=epochs, device=device, verbose=False,
        model_kwargs={"chain_mode": chain_mode},
    )
    model.eval()
    ddg_ids, ddg_true, ddg_pred = [], [], []
    elisa_ids, elisa_true, elisa_pred = [], [], []
    with torch.no_grad():
        for ex in test:
            sites = provider.featurize(ex)
            if any(s["delta_logp"] is None for s in sites):
                continue
            pred = model(sites).item()
            if ex.ddG is not None:
                ddg_ids.append(ex.complex_id); ddg_true.append(ex.ddG); ddg_pred.append(pred)
            elif ex.elisa_ratio is not None:
                elisa_ids.append(ex.complex_id); elisa_true.append(ex.elisa_ratio); elisa_pred.append(pred)
    return {
        "ddg": full_report(ddg_ids, ddg_true, ddg_pred),
        "elisa": full_report(elisa_ids, elisa_true, elisa_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--chain-modes", nargs="+", default=["concat", "none", "dual_memory"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rde-epochs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip", nargs="+", default=[], help="benchmark keys to skip: rde, ab645, ab1101, s1131, abdesign")
    parser.add_argument("--out", default="sweep_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)
    masked_cache = MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    results = {}
    for mode in args.chain_modes:
        print(f"\n{'=' * 70}\nCHAIN_MODE = {mode}\n{'=' * 70}", flush=True)
        results[mode] = {}
        if "rde" not in args.skip:
            print("-- RDE-PPI --", flush=True)
            r = eval_rde(provider, hidden, mode, args.rde_epochs, device)
            results[mode]["rde"] = r
            print(json.dumps(r, default=str), flush=True)
        for name in PROTATTBA_BENCHMARKS:
            if name in args.skip:
                continue
            print(f"-- {name} --", flush=True)
            r = eval_protattba_one(provider, hidden, mode, args.epochs, name, device)
            results[mode][name] = r
            print(json.dumps(r, default=str), flush=True)
        if "abdesign" not in args.skip:
            print("-- AbDesign holdout --", flush=True)
            r = eval_abdesign(provider, hidden, mode, args.epochs, device)
            results[mode]["abdesign"] = r
            print(json.dumps(r, default=str), flush=True)

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n\n" + "=" * 90)
    print("FULL SWEEP SUMMARY")
    print("=" * 90)
    for mode, benches in results.items():
        print(f"\n--- {mode} ---")
        for bench, r in benches.items():
            print(f"  {bench}: {r}")


if __name__ == "__main__":
    main()
