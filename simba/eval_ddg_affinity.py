"""Mutation-effect (ddG) prediction using the exact same absolute-affinity
architecture as simba/models/affinity.py -- AbsoluteAffinityModel.forward_ddg,
i.e. score the wildtype complex, score the mutant complex, take the
difference. No separate architecture, no background zero-shot term, no
per-site counterfactual/pairwise-epistasis machinery: multi-site mutations
are handled for free since the whole mutant sequence (every substitution
already applied, see simba/data/affinity_ddg.py) is scored in one pass.

Reuses the identical data and splits as simba/train.py's SimbaModel-based
results (RDE-PPI's reproduced 3-fold CV; ProtAttBA's released folds +
real predictions; AbDesign's zero-shot holdout) for a direct, apples-to
-apples comparison between the two architectures.
"""

from __future__ import annotations

import argparse
import copy
import math
import random

import numpy as np
import torch

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_all_mutations, load_mint_skempi, load_protattba, load_rde_skempi  # noqa: E402
from simba.eval_metrics import full_report, per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import REFERENCE as RDE_REFERENCE  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step, split_examples  # noqa: E402


def embed(cache: SequenceScoreCache, sequence: str, device: str, memo: dict) -> torch.Tensor:
    # Cache on CPU, move to device only at point of use. RDE-PPI's fold-level
    # memo can hold thousands of mostly-unique, often-long full-chain
    # embeddings simultaneously; caching them on GPU (as this used to do)
    # reliably exhausts 16GB VRAM by mid-training on the larger folds (same
    # OOM pattern already fixed in eval_ddg_hierarchical.py for AbDesign).
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float()
    return memo[sequence].to(device)


def predict(model, example, cache, device, memo):
    wt_full, mut_full, partner_seq = full_wt_mut_sequences(example)
    wt_emb = embed(cache, wt_full, device, memo)
    mut_emb = embed(cache, mut_full, device, memo)
    partner_emb = (embed(cache, partner_seq, device, memo) if partner_seq
                   else torch.zeros(0, wt_emb.shape[-1], device=device))
    return model.forward_ddg(wt_emb, mut_emb, partner_emb)


def evaluate(model, examples, cache, device, memo, min_n=3):
    model.eval()
    ids, y_true, y_pred = [], [], []
    with torch.no_grad():
        for ex in examples:
            if ex.ddG is None:
                continue
            pred = predict(model, ex, cache, device, memo).item()
            ids.append(ex.complex_id)
            y_true.append(ex.ddG)
            y_pred.append(pred)
    return {
        "overall": pooled_metrics(y_true, y_pred),
        "per_struct": per_complex_metrics(ids, y_true, y_pred, min_n=min_n),
    }


def run_training(train, val, *, hidden, epochs, batch_size, lr, weight_decay, device, seed,
                  cache, memo, min_n=3, verbose=True, fold_label="", use_pairformer=True):
    model = AbsoluteAffinityModel(plm_hidden_size=hidden, use_pairformer=use_pairformer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)

    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_score, best_state, global_step = -math.inf, None, 0

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, lr)
            valid_examples = [ex for ex in batch if ex.ddG is not None]
            if not valid_examples:
                global_step += 1
                continue
            # Per-example forward+backward (gradient accumulation) instead of
            # summing every example's loss into one graph before a single
            # backward() call: RDE-PPI's self-chain length varies enormously
            # example to example, and the pairwise block's [Wq, M, pair_dim]
            # tensor scales with it, so an unlucky batch composition (several
            # long sequences landing together) can spike peak memory far past
            # what any single example needs -- observed to hard-crash the
            # process (no Python traceback, OS/driver-level VRAM overcommit)
            # on RDE-PPI's larger folds. Backward is linear, so accumulating
            # per-example backward() calls on a loss pre-divided by n_valid
            # yields an identical gradient to the batched version; only one
            # example's computation graph is ever live at a time, bounding
            # peak memory by the single largest example instead of the batch.
            optimizer.zero_grad()
            batch_loss_value = 0.0
            n_valid = len(valid_examples)
            for ex in valid_examples:
                pred = predict(model, ex, cache, device, memo)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
                batch_loss_value += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss_value * n_valid
            n_seen += n_valid
            global_step += 1

        val_report = evaluate(model, val, cache, device, memo, min_n=min_n)
        score = val_report["per_struct"]["mean_spearman"]
        score_label = "per_struct_spearman"
        if math.isnan(score):
            # Per-complex Spearman needs >=min_n examples from the *same*
            # complex in the validation split; small per-fold val sets (e.g.
            # ProtAttBA's 10-fold splits) often don't have that for any
            # complex, so every epoch reports NaN and best_state never gets
            # set -- silently falling back to whatever the last epoch's
            # (possibly undertrained/unstable) weights happen to be. Falling
            # back to pooled Spearman keeps checkpoint selection meaningful
            # whenever the per-complex metric can't be computed.
            score = val_report["overall"]["spearman"]
            score_label = "pooled_spearman (per_struct was NaN)"
        is_best = not math.isnan(score) and score > best_score
        if is_best:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"{fold_label}epoch {epoch}: train_mse={epoch_loss / max(n_seen, 1):.4f} "
                  f"val_{score_label}={score:.4f}{'  <- new best' if is_best else ''}",
                  flush=True)
        if device.startswith("cuda"):
            # RDE-PPI's self-chain length varies enormously across examples
            # (the pairwise block materializes a [Wq, M, pair_dim] tensor per
            # example, Wq = full self-chain length), so the per-step tensor
            # shape never repeats -- PyTorch's caching allocator can't reuse
            # freed blocks across shapes and "reserved but unallocated"
            # memory grows across an epoch until CUDA OOMs even though live
            # usage is far below capacity. Clearing the cache once per epoch
            # (cheap relative to a ~270-step epoch) resets fragmentation
            # before it compounds across many epochs/folds/seeds.
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"{fold_label}loaded best checkpoint (val score={best_score:.4f})", flush=True)
    return model, best_score


def eval_rde(cache, hidden, epochs, batch_size, lr, weight_decay, device, use_pairformer=True,
             loader=load_rde_skempi, label="RDE-PPI"):
    examples = loader()
    folds = sorted({e.meta["cvfold"] for e in examples})
    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        # `memo` is scoped per-fold, not per-call: RDE-PPI's CV is complex
        # -level, so folds share zero complexes/mutant sequences, and each
        # fold's full-chain mutant embeddings (large: one unique tensor per
        # mutation row) are never reused once that fold is done. Keeping one
        # `memo` alive across all 3 folds instead accumulates ~the whole
        # dataset's embeddings simultaneously -- observed ballooning to
        # ~18GB RSS by the final fold. Fresh dict per fold bounds memory to
        # roughly one fold's share and lets it be GC'd before the next.
        memo: dict = {}
        print(f"\n=== {label} fold {fold} (held out as test) ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
        model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, min_n=10, fold_label=f"[fold {fold}] ",
                                 use_pairformer=use_pairformer)
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pred = predict(model, ex, cache, device, memo).item()
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(pred)
    return {
        "overall": pooled_metrics(all_true, all_pred),
        "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
        "predictions": {"ids": all_ids, "true": all_true, "pred": all_pred},
    }


def eval_protattba_one(cache, hidden, epochs, batch_size, lr, weight_decay, device, name, use_pairformer=True):
    examples = load_protattba(name)
    folds = sorted({e.meta["fold"] for e in examples})
    pdb_ids, y_true, y_pred, y_prot = [], [], [], []
    # Per-fold pooled metrics (computed within each held-out fold, then
    # mean+/-std across folds) -- this is the convention ProtAttBA's own
    # paper (Table 1, "the average of these...iterations") almost certainly
    # uses for its reported PCC/RMSE, which is neither our all-folds-combined
    # `pooled_metrics` (inflated by between-complex/between-fold variance
    # once everything is mixed together) nor our `per_complex_metrics`
    # (a materially different quantity: mean within-complex rank
    # correlation, deliberately chosen elsewhere in this project specifically
    # to avoid the pooling artifact -- see build plan). Computing this third
    # definition is required for a comparison that actually matches their
    # Table 1 numbers, not just "some correlation metric on the same rows".
    fold_metrics_ours, fold_metrics_prot = [], []
    for fold in folds:
        memo: dict = {}  # per-fold, see eval_rde's comment on why
        print(f"\n=== {name} fold {fold} (held out as test) ===", flush=True)
        test = [e for e in examples if e.meta["fold"] == fold]
        train_full = [e for e in examples if e.meta["fold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, min_n=3, fold_label=f"[{name} fold {fold}] ",
                                 use_pairformer=use_pairformer)
        model.eval()
        fold_true, fold_pred, fold_prot = [], [], []
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pdb_ids.append(ex.meta.get("pdb_id", ex.complex_id))
                pred = predict(model, ex, cache, device, memo).item()
                y_true.append(ex.ddG)
                y_pred.append(pred)
                y_prot.append(ex.meta["protattba_prediction"])
                fold_true.append(ex.ddG)
                fold_pred.append(pred)
                fold_prot.append(ex.meta["protattba_prediction"])
        fold_metrics_ours.append(pooled_metrics(fold_true, fold_pred))
        fold_metrics_prot.append(pooled_metrics(fold_true, fold_prot))

    def _fold_avg(fold_metrics):
        pearsons = [m["pearson"] for m in fold_metrics if not math.isnan(m["pearson"])]
        spearmans = [m["spearman"] for m in fold_metrics if not math.isnan(m["spearman"])]
        rmses = [m["rmse"] for m in fold_metrics if not math.isnan(m["rmse"])]
        return {
            "pcc_mean": float(np.mean(pearsons)) if pearsons else float("nan"),
            "pcc_std": float(np.std(pearsons)) if pearsons else float("nan"),
            "scc_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
            "scc_std": float(np.std(spearmans)) if spearmans else float("nan"),
            "rmse_mean": float(np.mean(rmses)) if rmses else float("nan"),
            "rmse_std": float(np.std(rmses)) if rmses else float("nan"),
            "n_folds": len(fold_metrics),
        }

    return {
        "overall": pooled_metrics(y_true, y_pred),
        "per_struct": per_complex_metrics(pdb_ids, y_true, y_pred, min_n=3),
        "prot_overall": pooled_metrics(y_true, y_prot),
        "prot_per_struct": per_complex_metrics(pdb_ids, y_true, y_prot, min_n=3),
        "fold_avg_ours": _fold_avg(fold_metrics_ours),
        "fold_avg_prot": _fold_avg(fold_metrics_prot),
        "predictions": {"ids": pdb_ids, "true": y_true, "pred": y_pred, "prot_pred": y_prot},
    }


def eval_abdesign(cache, hidden, epochs, batch_size, lr, weight_decay, device, use_pairformer=True):
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    memo: dict = {}
    model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                             lr=lr, weight_decay=weight_decay, device=device, seed=0,
                             cache=cache, memo=memo, min_n=3, use_pairformer=use_pairformer)
    model.eval()
    ddg_ids, ddg_true, ddg_pred = [], [], []
    elisa_ids, elisa_true, elisa_pred = [], [], []
    with torch.no_grad():
        for ex in test:
            pred = predict(model, ex, cache, device, memo).item()
            if ex.ddG is not None:
                ddg_ids.append(ex.complex_id); ddg_true.append(ex.ddG); ddg_pred.append(pred)
            elif ex.elisa_ratio is not None:
                elisa_ids.append(ex.complex_id); elisa_true.append(ex.elisa_ratio); elisa_pred.append(pred)
    return {
        "ddg": full_report(ddg_ids, ddg_true, ddg_pred),
        "elisa": full_report(elisa_ids, elisa_true, elisa_pred),
        "predictions": {
            "ddg": {"ids": ddg_ids, "true": ddg_true, "pred": ddg_pred},
            "elisa": {"ids": elisa_ids, "true": elisa_true, "pred": elisa_pred},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--benchmarks", nargs="+", default=["rde"],
                         choices=["rde", "ab645", "ab1101", "s1131", "abdesign", "mint_split"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pairformer", action="store_true",
                         help="use the simpler single-query GatedMemory readout instead of the "
                              "Pairformer-lite dense pairwise block, for the ddG-via-affinity ablation")
    parser.add_argument("--out", default=None, help="optional path to dump all results as JSON")
    args = parser.parse_args()
    use_pairformer = not args.no_pairformer
    all_results = {}

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    if "rde" in args.benchmarks:
        print(f"\n--- RDE-PPI (unified absolute-affinity-diff model, pairformer={use_pairformer}) ---", flush=True)
        r = eval_rde(cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
                     use_pairformer=use_pairformer)
        print(r, flush=True)
        print("\nReference (RDE-PPI Table 1):", flush=True)
        for name, ref in RDE_REFERENCE.items():
            print(f"  {name}: per_struct P/S={ref['per_struct_pearson']:.3f}/{ref['per_struct_spearman']:.3f} "
                  f"overall P/S={ref['overall_pearson']:.3f}/{ref['overall_spearman']:.3f}", flush=True)
        all_results["rde"] = r

    if "mint_split" in args.benchmarks:
        print(f"\n--- MINT-protocol SKEMPI split (same rows as RDE-PPI, seed=2023, "
              f"unified absolute-affinity-diff model, pairformer={use_pairformer}) ---", flush=True)
        r = eval_rde(cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
                     use_pairformer=use_pairformer, loader=load_mint_skempi, label="MINT-split")
        print(r, flush=True)
        all_results["mint_split"] = r

    for name in ("ab645", "ab1101", "s1131"):
        if name in args.benchmarks:
            print(f"\n--- {name} (unified absolute-affinity-diff model, pairformer={use_pairformer}) ---", flush=True)
            r = eval_protattba_one(cache, hidden, args.epochs, args.batch_size, args.lr,
                                    args.weight_decay, device, name, use_pairformer=use_pairformer)
            print(r, flush=True)
            all_results[name] = r

    if "abdesign" in args.benchmarks:
        print(f"\n--- AbDesign holdout (unified absolute-affinity-diff model, pairformer={use_pairformer}) ---",
              flush=True)
        r = eval_abdesign(cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
                           use_pairformer=use_pairformer)
        print(r, flush=True)
        all_results["abdesign"] = r

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved results to {args.out}]", flush=True)


if __name__ == "__main__":
    main()
