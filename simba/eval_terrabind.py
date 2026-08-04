"""Follow-up ablation for three architecture additions inspired by pairwise
protein-protein interaction modeling:

  1. chain_mode="pairformer" (dense window x memory pairwise map + row/column
     attention) vs the current-best chain_mode="concat", on RDE-PPI's exact
     split and the AbDesign zero-shot holdout -- the same two protocols
     already used to certify "concat" as the winner, so this is a fair
     head-to-head against the incumbent rather than a new benchmark.
  2. ranking_loss_weight in {0.0, 0.3} (within-complex pairwise Huber
     ranking loss) on RDE-PPI, chain_mode="concat" -- targets per-complex
     Spearman directly, so RDE-PPI (which has many complexes with >=10
     mutants each) is the most informative protocol to check it on.
  3. use_epinet=True calibration check on the AbDesign ddG holdout: does
     predict_with_uncertainty's std actually correlate with the model's
     real absolute error? (Pearson between |error| and predicted std,
     reported alongside the point-estimate accuracy so a calibration win
     isn't silently traded for an accuracy loss.)

Deliberately scoped down from a full re-sweep: each of these is tested on
1-2 protocols rather than all 5, since the goal is "does this specific
addition help at all", not re-certifying the whole comparison matrix again.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import pearsonr

from simba import env

env.setup()

from simba.data.dataset import load_all_mutations, load_rde_skempi  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import full_report, per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import run_training, split_examples  # noqa: E402


def eval_rde(provider, hidden, epochs, device, model_kwargs, ranking_loss_weight=0.0):
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
            model_kwargs=model_kwargs, ranking_loss_weight=ranking_loss_weight,
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


def eval_abdesign(provider, hidden, epochs, device, model_kwargs, calibration=False):
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    model, *_ = run_training(
        train, val, test, provider=provider, plm_hidden_size=hidden,
        epochs=epochs, device=device, verbose=False, model_kwargs=model_kwargs,
    )
    model.eval()
    ddg_ids, ddg_true, ddg_pred = [], [], []
    stds = []
    with torch.no_grad():
        for ex in test:
            if ex.ddG is None:
                continue
            sites = provider.featurize(ex)
            if any(s["delta_logp"] is None for s in sites):
                continue
            if calibration:
                mean, std = model.predict_with_uncertainty(sites, n_samples=25)
                pred, std_val = mean.item(), std.item()
                stds.append(std_val)
            else:
                pred = model(sites).item()
            ddg_ids.append(ex.complex_id)
            ddg_true.append(ex.ddG)
            ddg_pred.append(pred)

    result = {"ddg": full_report(ddg_ids, ddg_true, ddg_pred)}
    if calibration:
        abs_err = np.abs(np.asarray(ddg_true) - np.asarray(ddg_pred))
        stds_arr = np.asarray(stds)
        if np.std(stds_arr) > 0 and np.std(abs_err) > 0:
            calib_corr = float(pearsonr(abs_err, stds_arr)[0])
        else:
            calib_corr = float("nan")
        result["calibration"] = {
            "corr_abs_error_vs_predicted_std": calib_corr,
            "mean_predicted_std": float(np.mean(stds_arr)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rde-epochs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="terrabind_ablation_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)
    masked_cache = MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    results = {}

    print("=" * 70, "\n1. chain_mode: concat vs pairformer\n", "=" * 70, flush=True)
    for mode in ("concat", "pairformer"):
        print(f"-- RDE-PPI, chain_mode={mode} --", flush=True)
        r = eval_rde(provider, hidden, args.rde_epochs, device, {"chain_mode": mode})
        results[f"rde_{mode}"] = r
        print(json.dumps(r, default=str), flush=True)
        print(f"-- AbDesign holdout, chain_mode={mode} --", flush=True)
        r = eval_abdesign(provider, hidden, args.epochs, device, {"chain_mode": mode})
        results[f"abdesign_{mode}"] = r
        print(json.dumps(r, default=str), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("=" * 70, "\n2. ranking_loss_weight: 0.0 vs 0.3 (chain_mode=concat) on RDE-PPI\n", "=" * 70, flush=True)
    for w in (0.0, 0.3):
        print(f"-- RDE-PPI, ranking_loss_weight={w} --", flush=True)
        r = eval_rde(provider, hidden, args.rde_epochs, device, {"chain_mode": "concat"},
                     ranking_loss_weight=w)
        results[f"rde_ranking_{w}"] = r
        print(json.dumps(r, default=str), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("=" * 70, "\n3. epinet calibration check on AbDesign ddG holdout\n", "=" * 70, flush=True)
    r = eval_abdesign(provider, hidden, args.epochs, device,
                      {"chain_mode": "concat", "use_epinet": True}, calibration=True)
    results["abdesign_epinet_calibration"] = r
    print(json.dumps(r, default=str), flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n" + "=" * 90)
    print("TERRABIND ABLATION SUMMARY")
    print("=" * 90)
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
