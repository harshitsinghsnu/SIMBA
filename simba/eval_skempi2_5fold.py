"""Core absolute-affinity (ΔG) ablation benchmark: SKEMPI2's *expanded*
pool — every wildtype complex (340) plus every individually-resolved
mutant row (6,706, RDE-PPI's exact reproduction), each mutant's own real,
experimentally-measured Kd_mut treated as its own absolute-affinity data
point. **7,046 total examples, 340 unique complexes** — general
protein-protein interactions (enzyme-inhibitor, signaling, immune
recognition, antibody-antigen as one subset among many, not the focus;
see `simba/data/build_skempi2_expanded.py`), not PDBbind-gated, no new
fetching needed. 5-fold complex-level cross-validation (never splitting a
complex's wildtype+mutant rows across train/test).

Unlike benchmark46/prott_test1/PPI-Affinity Test Set 1&2 (each a single
small held-out split, 44-90 complexes), this uses every available example
for both training and evaluation across folds, giving a substantially
larger and more statistically robust ablation surface — the intended
"core" dataset for absolute-affinity ablations (architecture variants, PLM
backbone, fine-tuning), per project scope.

Split: `random.Random(SPLIT_SEED).shuffle()` on the sorted complex-ID list,
divided into 5 contiguous folds — same reproducible, complex-level
convention as RDE-PPI's exact-reproduction split elsewhere in this project.
Each fold serves once as held-out test; the model is retrained from scratch
per fold on the other 4 folds (with an inner validation split for checkpoint
selection), and per-fold test predictions are pooled for reporting.

Runs three models for direct comparison:
  --model pairformer      AbsoluteAffinityModel (proposed architecture)
  --model concat_mlp      ConcatMLPModel (no cross-chain interaction at all)
  --model cross_attn_mlp  CrossAttnMLPModel (one cross-attention pass, no
                           pairformer's dense-pairwise-map refinement)

Three fold-construction protocols (`--split`), all complex-level (a
complex's rows never split across train/test) but differing in what else is
kept apart:
  --split complex         (default) fold by raw complex_id only -- the
                           original protocol above. Two different PDB
                           entries can still land in different folds even if
                           they're near-identical complexes (e.g. the same
                           antibody scaffold solved against several
                           antigens), so this is the *easiest* split.
  --split seq_similarity   fold by k-mer-Jaccard sequence cluster
                           (`simba/data/clustering.py`, no external
                           alignment tool needed) over each complex's full
                           self+partner sequence content -- near-duplicate/
                           homologous complexes are forced into the same
                           fold, so no fold can "cheat" via a close relative
                           of a test complex being in train.
  --split cold_target      fold by k-mer-Jaccard cluster over the *partner*
                           sequence alone -- every occurrence of a given
                           binding target (e.g. the same antigen bound by
                           several different antibodies, or the same MHC
                           bound by several different TCRs) lands in exactly
                           one fold, so test performance reflects
                           generalization to binding partners never seen in
                           any complex during training, not just unseen
                           mutations of an already-seen target.
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

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.data.build_skempi2_expanded import build_skempi2_expanded_examples  # noqa: E402
from simba.data.clustering import greedy_cluster, kmers  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.models.concat_mlp import ConcatMLPModel  # noqa: E402
from simba.models.cross_attn_mlp import CrossAttnMLPModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

SPLIT_SEED = 20260712
N_FOLDS = 5


def build_cluster_assignment(examples, split_type: str) -> dict[str, str]:
    """complex_id (upper) -> cluster_key. `split_type="complex"` is the
    identity map (original protocol); `seq_similarity`/`cold_target` group
    several complex_ids under one cluster_key via k-mer Jaccard clustering
    (see module docstring for what each clusters on)."""
    unique: dict[str, object] = {}
    for e in examples:
        cid = e.complex_id.upper()
        if cid not in unique:
            unique[cid] = e

    if split_type == "complex":
        return {cid: cid for cid in unique}

    if split_type == "cold_target":
        rep_kmers = {cid: kmers(ex.partner_sequence) for cid, ex in unique.items()}
    elif split_type == "seq_similarity":
        rep_kmers = {cid: kmers(ex.self_sequence + ex.partner_sequence) for cid, ex in unique.items()}
    else:
        raise ValueError(split_type)

    cluster_idx = greedy_cluster(rep_kmers)  # complex_id -> int cluster index
    return {cid: f"cluster_{idx}" for cid, idx in cluster_idx.items()}


def make_folds(examples, split_type: str = "complex"):
    cluster_by_complex = build_cluster_assignment(examples, split_type)
    cluster_keys = sorted(set(cluster_by_complex.values()))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(cluster_keys)
    fold_size = math.ceil(len(cluster_keys) / N_FOLDS)
    key_folds = [set(cluster_keys[i * fold_size:(i + 1) * fold_size]) for i in range(N_FOLDS)]
    return [
        {cid for cid, key in cluster_by_complex.items() if key in key_fold}
        for key_fold in key_folds
    ]


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def make_model(model_name, hidden):
    if model_name == "pairformer":
        return AbsoluteAffinityModel(plm_hidden_size=hidden)
    elif model_name == "concat_mlp":
        return ConcatMLPModel(plm_hidden_size=hidden)
    elif model_name == "cross_attn_mlp":
        return CrossAttnMLPModel(plm_hidden_size=hidden)
    raise ValueError(model_name)


def evaluate(model, examples, cache, device, memo):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed(cache, ex.self_sequence, device, memo)
            partner_emb = embed(cache, ex.partner_sequence, device, memo)
            preds.append(model(self_emb, partner_emb).item())
            trues.append(ex.dG)
    return np.asarray(preds), np.asarray(trues)


def run_fold(train, val, test, *, model_name, hidden, epochs, batch_size, lr, weight_decay,
             device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    memo: dict = {}
    model = make_model(model_name, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_pcc, best_state, global_step = -math.inf, None, 0

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, lr)
            optimizer.zero_grad()
            # Per-example backward (gradient accumulation), not a single
            # backward() over the whole batch's summed loss: this pool's
            # self-chain length varies enormously across complexes (same
            # issue diagnosed for RDE-PPI, eval_ddg_affinity.py), so an
            # unlucky batch composition can spike peak memory far past any
            # single example's needs. Backward is linear, so accumulating
            # per-example backward() on a loss pre-divided by batch size
            # gives an identical gradient to the batched version, bounding
            # peak memory by the single largest example instead of the batch.
            n_valid = len(batch)
            for ex in batch:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_preds, val_trues = evaluate(model, val, cache, device, memo)
        val_pcc = float(pearsonr(val_trues, val_preds)[0]) if np.std(val_preds) > 0 else float("nan")
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test, cache, device, memo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--model", choices=["pairformer", "concat_mlp", "cross_attn_mlp"], default="pairformer")
    parser.add_argument("--pool", choices=["skempi2", "combined"], default="skempi2",
                         help="skempi2: expanded SKEMPI2 only (7,046 ex). "
                              "combined: + Kastritis-lineage benchmark46 + ATLAS TCR-pMHC.")
    parser.add_argument("--split", choices=["complex", "seq_similarity", "cold_target"], default="complex",
                         help="complex: fold by raw complex_id (original protocol). "
                              "seq_similarity: fold by k-mer-Jaccard cluster over the whole complex's "
                              "sequence content. cold_target: fold by cluster over the partner "
                              "sequence alone (held-out binding targets).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="skempi2_5fold_results.json")
    parser.add_argument("--predictions-out", default=None,
                         help="If set, dump one JSON record per example (complex_id, self/partner "
                              "sequence, experimental dG, predicted dG, fold index, and the example's "
                              "own meta dict) -- raw material for any later per-target/per-complex "
                              "analysis (e.g. Fisher-z intra-target correlation) without retraining. "
                              "Defaults to `<out>` with '.json' replaced by '.predictions.json'.")
    args = parser.parse_args()
    predictions_out = args.predictions_out or args.out.replace(".json", ".predictions.json")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_combined_examples() if args.pool == "combined" else build_skempi2_expanded_examples()
    folds = make_folds(examples, split_type=args.split)
    print(f"[skempi2_5fold] pool={args.pool} split={args.split} {len(examples)} examples, "
          f"{len(set(e.complex_id.upper() for e in examples))} complexes, "
          f"fold sizes={[len(f) for f in folds]}, model={args.model}", flush=True)

    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    all_preds, all_trues = [], []
    fold_pccs, fold_sccs, fold_maes, fold_rmses = [], [], [], []
    all_records = []  # one dict per example: full metadata + experimental value + prediction
    rng = random.Random(args.seed)

    for fold_idx, test_ids in enumerate(folds):
        test = [e for e in examples if e.complex_id in test_ids]
        train_val = [e for e in examples if e.complex_id not in test_ids]
        idx = list(range(len(train_val)))
        rng.shuffle(idx)
        n_val = max(1, int(args.val_frac * len(idx)))
        val = [train_val[i] for i in idx[:n_val]]
        train = [train_val[i] for i in idx[n_val:]]
        print(f"\n=== fold {fold_idx}: train={len(train)} val={len(val)} test={len(test)} ===", flush=True)

        preds, trues = run_fold(train, val, test, model_name=args.model, hidden=hidden,
                                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                 weight_decay=args.weight_decay, device=device,
                                 seed=args.seed * 100 + fold_idx, cache=cache)
        pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        scc = float(spearmanr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        mae = float(np.mean(np.abs(trues - preds)))
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
        fold_pccs.append(pcc); fold_sccs.append(scc); fold_maes.append(mae); fold_rmses.append(rmse)
        print(f"[fold {fold_idx}] PCC={pcc:.4f} SCC={scc:.4f} MAE={mae:.4f} RMSE={rmse:.4f} (n={len(test)})",
              flush=True)
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        for ex, pred, true in zip(test, preds, trues):
            all_records.append({
                "fold": fold_idx, "complex_id": ex.complex_id,
                "self_sequence": ex.self_sequence, "partner_sequence": ex.partner_sequence,
                "dG_true": float(true), "dG_pred": float(pred), "meta": ex.meta,
            })

    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    pooled_pcc = float(pearsonr(all_trues, all_preds)[0])
    pooled_scc = float(spearmanr(all_trues, all_preds)[0])
    pooled_mae = float(np.mean(np.abs(all_trues - all_preds)))
    pooled_rmse = float(np.sqrt(np.mean((all_trues - all_preds) ** 2)))

    result = {
        "model": args.model, "pool": args.pool, "split": args.split,
        "n_complexes": len(examples), "n_folds": N_FOLDS,
        "fold_pcc_mean": float(np.mean(fold_pccs)), "fold_pcc_std": float(np.std(fold_pccs)),
        "fold_scc_mean": float(np.mean(fold_sccs)), "fold_scc_std": float(np.std(fold_sccs)),
        "fold_mae_mean": float(np.mean(fold_maes)), "fold_mae_std": float(np.std(fold_maes)),
        "fold_rmse_mean": float(np.mean(fold_rmses)), "fold_rmse_std": float(np.std(fold_rmses)),
        "fold_pccs": fold_pccs, "fold_sccs": fold_sccs, "fold_maes": fold_maes, "fold_rmses": fold_rmses,
        "pooled_pcc": pooled_pcc, "pooled_scc": pooled_scc,
        "pooled_mae": pooled_mae, "pooled_rmse": pooled_rmse,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(predictions_out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n[skempi2_5fold] model={args.model} "
          f"fold-avg PCC={result['fold_pcc_mean']:.4f}+/-{result['fold_pcc_std']:.4f} "
          f"SCC={result['fold_scc_mean']:.4f}+/-{result['fold_scc_std']:.4f} "
          f"MAE={result['fold_mae_mean']:.4f}+/-{result['fold_mae_std']:.4f} "
          f"RMSE={result['fold_rmse_mean']:.4f}+/-{result['fold_rmse_std']:.4f}  "
          f"pooled PCC={pooled_pcc:.4f} SCC={pooled_scc:.4f} MAE={pooled_mae:.4f} RMSE={pooled_rmse:.4f}",
          flush=True)
    print(f"[skempi2_5fold] saved {len(all_records)} raw per-example predictions to {predictions_out}", flush=True)


if __name__ == "__main__":
    main()
