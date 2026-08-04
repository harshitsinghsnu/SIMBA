"""Absolute binding-affinity (dG) benchmark, for comparison against
PPI-Graphomer (Xie et al., BMC Bioinformatics 2025, DOI 10.1186/s12859-025
-06123-2). See simba/data/build_affinity_benchmark.py for why this uses
SKEMPI2's own wildtype Kd values as training data (no PDBbind license) and
the "Corrected Benchmark 1.0" affinity table as a held-out external test set
(same Kastritis-lineage benchmark their Test Set 1 is derived from, though
not the identical 75-sample filtered version -- caveated in the printed
comparison, not glossed over).

This is architecturally a different model (`AbsoluteAffinityModel`) from the
rest of SIMBA -- no mutation, no background ddG term, no counterfactual
site comparison, since there's no mutation here at all, just one wildtype
complex regressed to a scalar dG. Trains fast (few hundred complexes, no
per-mutation-site work), so this whole script runs as a single process,
not split across separate smoke/full runs like the mutation-effect scripts.

Supports three additional ablation arms on top of the base model:
  --n-pairformer-layers N   stack N row/column-attention layers instead of 1.
  --use-epinet              epistemic-index residual head; reports a
                             calibration check (corr(|error|, predicted std))
                             on the test set alongside point accuracy.
  --head-type cosine        cosine-similarity head instead of an MLP
                             regression head (target_min/max computed from
                             the training set's own dG values; trained with
                             a cosine-MSE loss against the target rescaled to
                             [-1, 1], not MSE in dG units directly).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.build_affinity_benchmark import (  # noqa: E402
    build_prott_affinity_test1_reproduction,
    build_skempi2_wt_train_set,
    build_test_set,
)
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

# PPI-Graphomer's own reported numbers (BMC Bioinformatics 2025), for direct
# reference -- their Test Set 1 is 75 samples derived from the same Kastritis
# et al. 2011 benchmark lineage as ours (44 samples, "Corrected Benchmark 1.0").
REFERENCE_BENCHMARK46 = {
    "PPI-Graphomer, 5-fold CV (PDBbind, 2085 samples)": {"pcc": 0.581, "mae": 1.63},
    "PPI-Graphomer, Test Set 1 (Kastritis-derived, 75 samples)": {"pcc": 0.641, "mae": 1.64},
    "PPI-Graphomer, Test Set 2 (PPI-Affinity-derived, 87 samples)": {"pcc": 0.625, "mae": 1.51},
    "PPI-Graphomer, Benchmark v2 (188 samples)": {"pcc": 0.626, "mae": 1.60},
}

# ProtT-Affinity (Lou, arXiv:2511.16113, Nov 2025), Table 3 -- all evaluated
# on the identical "Test set 1" (79 complexes, Kastritis-lineage) under a
# shared curation protocol from Zhou et al. 2024 (ProAffinity-GNN). We recover
# 50/79 of these exact PDB IDs (see build_prott_affinity_test1_reproduction).
REFERENCE_PROTT_TEST1 = {
    "PRODIGY (structure, classic)": {"pcc": 0.735, "mae": 1.43},
    "ProAffinity-GNN (structure, 2024, strongest baseline)": {"pcc": 0.697, "mae": 1.52},
    "PPI-Affinity SVM (2022)": {"pcc": 0.616, "mae": 1.82},
    "DFIRE (structure, classic)": {"pcc": 0.602, "mae": 4.64},
    "ProtT-Affinity (sequence-only, Nov 2025)": {"pcc": 0.628, "mae": 1.65},
    "CP_PIE (structure)": {"pcc": 0.517, "mae": 8.80},
    "ISLAND (sequence)": {"pcc": 0.378, "mae": 2.10},
}


def embed(cache: SequenceScoreCache, sequence: str, device: str, memo: dict) -> torch.Tensor:
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float().to(device)
    return memo[sequence]


def evaluate(model, examples, cache, device, memo, calibration=False, return_raw=False):
    model.eval()
    preds, trues, stds, ids = [], [], [], []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed(cache, ex.self_sequence, device, memo)
            partner_emb = embed(cache, ex.partner_sequence, device, memo)
            if calibration:
                mean, std = model.predict_with_uncertainty(self_emb, partner_emb, n_samples=25)
                preds.append(mean.item())
                stds.append(std.item())
            else:
                preds.append(model(self_emb, partner_emb).item())
            trues.append(ex.dG)
            ids.append(ex.complex_id)
    preds, trues = np.asarray(preds), np.asarray(trues)
    pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
    scc = float(spearmanr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
    mae = float(np.mean(np.abs(trues - preds)))
    rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
    result = {"pcc": pcc, "scc": scc, "mae": mae, "rmse": rmse, "n": len(trues), "ids": ids,
              "pred": preds.tolist(), "true": trues.tolist()}
    if calibration:
        abs_err = np.abs(trues - preds)
        stds_arr = np.asarray(stds)
        calib = (float(pearsonr(abs_err, stds_arr)[0])
                 if np.std(stds_arr) > 0 and np.std(abs_err) > 0 else float("nan"))
        result["calibration"] = {"corr_abs_error_vs_predicted_std": calib,
                                  "mean_predicted_std": float(np.mean(stds_arr))}
        if return_raw:
            result["calibration"]["raw"] = {
                "pred_mean": preds.tolist(), "pred_std": stds_arr.tolist(), "true": trues.tolist(),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pairformer", action="store_true",
                         help="use the simpler single-query GatedMemory readout instead of the "
                              "Pairformer-lite dense pairwise block")
    parser.add_argument("--n-pairformer-layers", type=int, default=1,
                         help="stack this many row/column-attention layers in the Pairformer block")
    parser.add_argument("--use-epinet", action="store_true",
                         help="epistemic-index residual head for calibrated uncertainty")
    parser.add_argument("--dump-calibration", default=None,
                         help="with --use-epinet, dump per-test-example (pred_mean, pred_std, true) to this "
                              "JSON path for a reliability-diagram figure")
    parser.add_argument("--head-type", choices=["mlp", "cosine"], default="mlp",
                         help="cosine = cosine-similarity head + cosine-MSE loss, instead of an MLP head")
    parser.add_argument("--test-set", choices=["benchmark46", "prott_test1"], default="benchmark46",
                         help="benchmark46 = Corrected Benchmark 1.0 (44 resolved complexes); "
                              "prott_test1 = 50/79 recovered complexes from ProtT-Affinity's exact Test Set 1")
    parser.add_argument("--out", default=None,
                         help="optional path to save a summary + per-example test predictions JSON")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test = build_test_set() if args.test_set == "benchmark46" else build_prott_affinity_test1_reproduction()
    reference = REFERENCE_BENCHMARK46 if args.test_set == "benchmark46" else REFERENCE_PROTT_TEST1
    test_ids = {e.complex_id for e in test}
    train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)

    rng = random.Random(args.seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(args.val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]
    print(f"train={len(train)} val={len(val)} test={len(test)} (held-out, zero complex overlap with train/val)")

    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)
    memo: dict = {}

    model_kwargs = dict(
        plm_hidden_size=hidden, use_pairformer=not args.no_pairformer,
        n_pairformer_layers=args.n_pairformer_layers, use_epinet=args.use_epinet,
        head_type=args.head_type,
    )
    if args.head_type == "cosine":
        # target_min/max computed from the training set's own dG values, per
        # the given linear-mapping scheme -- not a fixed universal constant.
        train_targets = [ex.dG for ex in train]
        model_kwargs["target_min"] = min(train_targets)
        model_kwargs["target_max"] = max(train_targets)
        print(f"cosine head target range from training set: "
              f"[{model_kwargs['target_min']:.3f}, {model_kwargs['target_max']:.3f}]")

    model = AbsoluteAffinityModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    best_pcc, best_state, global_step = -math.inf, None, 0

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, args.lr)
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            for ex in batch:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                if args.head_type == "cosine":
                    cos_sim = model.forward_cos_sim(self_emb, partner_emb)
                    target_scaled = 2 * (ex.dG - model.target_min) / (model.target_max - model.target_min) - 1
                    batch_loss = batch_loss + (cos_sim - target_scaled) ** 2
                else:
                    pred = model(self_emb, partner_emb)
                    target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                    batch_loss = batch_loss + (pred - target) ** 2
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item() * len(batch)
            n_seen += len(batch)
            global_step += 1

        val_report = evaluate(model, val, cache, device, memo)
        print(f"epoch {epoch}: train_loss={epoch_loss / max(n_seen, 1):.4f} val_pcc={val_report['pcc']:.4f} "
              f"val_mae={val_report['mae']:.4f}")
        if not math.isnan(val_report["pcc"]) and val_report["pcc"] > best_pcc:
            best_pcc = val_report["pcc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"loaded best checkpoint (val pcc={best_pcc:.4f})")

    test_report = evaluate(model, test, cache, device, memo, calibration=args.use_epinet,
                            return_raw=bool(args.dump_calibration))
    if args.dump_calibration and args.use_epinet:
        with open(args.dump_calibration, "w") as f:
            json.dump(test_report["calibration"]["raw"], f)
        print(f"[calibration] raw (pred_mean, pred_std, true) dumped to {args.dump_calibration}")
    print("\n" + "=" * 70)
    print("ABSOLUTE-AFFINITY RESULTS")
    print("=" * 70)
    variant_bits = []
    variant_bits.append("simple GatedMemory readout" if args.no_pairformer
                         else f"Pairformer x{args.n_pairformer_layers}")
    if args.use_epinet:
        variant_bits.append("epinet")
    if args.head_type == "cosine":
        variant_bits.append("cosine head")
    variant = " + ".join(variant_bits)
    print(f"SIMBA (AbsoluteAffinityModel, {variant}), held-out {test_report['n']}-complex test set "
          f"({args.test_set}, zero overlap with SKEMPI2 wt-affinity training set):")
    print(f"  PCC={test_report['pcc']:.4f}  SCC={test_report['scc']:.4f}  "
          f"MAE={test_report['mae']:.4f}  RMSE={test_report['rmse']:.4f} kcal/mol")
    if args.use_epinet:
        print(f"  calibration: {test_report['calibration']}")
    print(f"\nReference numbers ({args.test_set}) -- different training data/exact test rows, "
          f"same benchmark lineage:")
    for name, r in reference.items():
        print(f"  {name}: PCC={r['pcc']:.3f}  MAE={r['mae']:.3f}")

    if args.out:
        summary = {
            "test_set": args.test_set, "n_pairformer_layers": args.n_pairformer_layers,
            "no_pairformer": args.no_pairformer, "use_epinet": args.use_epinet,
            "head_type": args.head_type, "seed": args.seed,
            "pcc": test_report["pcc"], "scc": test_report["scc"],
            "mae": test_report["mae"], "rmse": test_report["rmse"], "n": test_report["n"],
            "reference": reference,
            "predictions": {"ids": test_report["ids"], "true": test_report["true"], "pred": test_report["pred"]},
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[eval_affinity] saved {args.out}")


if __name__ == "__main__":
    main()
