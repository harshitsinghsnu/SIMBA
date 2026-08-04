"""Evaluates SimbaOperatorModel (simba/models/operator.py) -- the Response
Propagation Operator ablation -- on the same exact splits/data as every
other ddG comparison in this project, for a direct, apples-to-apples number
against the unified pairformer model's results.

Uses the same (wildtype self-sequence, mutant self-sequence, partner
sequence) data pipeline as simba/eval_ddg_affinity.py
(`full_wt_mut_sequences` -- self+teammate chains concatenated, the standard
approach used everywhere else in this project), so this ablation isolates
exactly one variable: the interaction-processing primitive (Response
Propagation Operator vs. pairformer), not the data representation.
"""

from __future__ import annotations

import argparse
import copy
import math
import random

import torch

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_all_mutations, load_protattba, load_rde_skempi  # noqa: E402
from simba.eval_metrics import full_report, per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import REFERENCE as RDE_REFERENCE  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.operator import SimbaOperatorModel  # noqa: E402
from simba.train import lr_at_step, split_examples  # noqa: E402


def embed(cache: SequenceScoreCache, sequence: str, device: str, memo: dict, hidden: int) -> torch.Tensor:
    if not sequence:
        return torch.zeros(0, hidden, device=device)
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float()
    return memo[sequence].to(device)


def predict(model, example, cache, device, memo, hidden):
    wt_full, mut_full, partner_seq = full_wt_mut_sequences(example)
    wt_emb = embed(cache, wt_full, device, memo, hidden)
    mut_emb = embed(cache, mut_full, device, memo, hidden)
    partner_emb = embed(cache, partner_seq, device, memo, hidden)
    return model(wt_emb, mut_emb, partner_emb)


def evaluate(model, examples, cache, device, memo, hidden, min_n=3):
    model.eval()
    ids, y_true, y_pred = [], [], []
    with torch.no_grad():
        for ex in examples:
            if ex.ddG is None:
                continue
            pred = predict(model, ex, cache, device, memo, hidden).item()
            ids.append(ex.complex_id)
            y_true.append(ex.ddG)
            y_pred.append(pred)
    return {
        "overall": pooled_metrics(y_true, y_pred),
        "per_struct": per_complex_metrics(ids, y_true, y_pred, min_n=min_n),
    }


def run_training(train, val, *, hidden, epochs, batch_size, lr, weight_decay, device, seed,
                  cache, memo, n_rounds, coupling_kernel_size, min_n=3, fold_label=""):
    model = SimbaOperatorModel(plm_hidden_size=hidden, n_rounds=n_rounds,
                                coupling_kernel_size=coupling_kernel_size).to(device)
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
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            n_valid = 0
            for ex in batch:
                if ex.ddG is None:
                    continue
                pred = predict(model, ex, cache, device, memo, hidden)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                batch_loss = batch_loss + (pred - target) ** 2
                n_valid += 1
            if n_valid == 0:
                global_step += 1
                continue
            batch_loss = batch_loss / n_valid
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item() * n_valid
            n_seen += n_valid
            global_step += 1

        val_report = evaluate(model, val, cache, device, memo, hidden, min_n=min_n)
        score = val_report["per_struct"]["mean_spearman"]
        score_label = "per_struct_spearman"
        if math.isnan(score):
            score = val_report["overall"]["spearman"]
            score_label = "pooled_spearman (per_struct was NaN)"
        is_best = not math.isnan(score) and score > best_score
        if is_best:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(f"{fold_label}epoch {epoch}: train_mse={epoch_loss / max(n_seen, 1):.4f} "
              f"val_{score_label}={score:.4f}{'  <- new best' if is_best else ''}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"{fold_label}loaded best checkpoint (val score={best_score:.4f})", flush=True)
    return model, best_score


def eval_rde(cache, hidden, epochs, batch_size, lr, weight_decay, device, n_rounds, coupling_kernel_size):
    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        memo: dict = {}
        print(f"\n=== RDE-PPI fold {fold} (held out as test) ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
        model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, n_rounds=n_rounds,
                                 coupling_kernel_size=coupling_kernel_size, min_n=10,
                                 fold_label=f"[fold {fold}] ")
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pred = predict(model, ex, cache, device, memo, hidden).item()
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(pred)
    return {
        "overall": pooled_metrics(all_true, all_pred),
        "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
    }


def eval_protattba_one(cache, hidden, epochs, batch_size, lr, weight_decay, device, name,
                        n_rounds, coupling_kernel_size):
    examples = load_protattba(name)
    folds = sorted({e.meta["fold"] for e in examples})
    pdb_ids, y_true, y_pred, y_prot = [], [], [], []
    for fold in folds:
        memo: dict = {}
        print(f"\n=== {name} fold {fold} (held out as test) ===", flush=True)
        test = [e for e in examples if e.meta["fold"] == fold]
        train_full = [e for e in examples if e.meta["fold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, n_rounds=n_rounds,
                                 coupling_kernel_size=coupling_kernel_size, min_n=3,
                                 fold_label=f"[{name} fold {fold}] ")
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pdb_ids.append(ex.meta.get("pdb_id", ex.complex_id))
                y_true.append(ex.ddG)
                y_pred.append(predict(model, ex, cache, device, memo, hidden).item())
                y_prot.append(ex.meta["protattba_prediction"])
    return {
        "overall": pooled_metrics(y_true, y_pred),
        "per_struct": per_complex_metrics(pdb_ids, y_true, y_pred, min_n=3),
        "prot_overall": pooled_metrics(y_true, y_prot),
        "prot_per_struct": per_complex_metrics(pdb_ids, y_true, y_prot, min_n=3),
    }


def eval_abdesign(cache, hidden, epochs, batch_size, lr, weight_decay, device, n_rounds, coupling_kernel_size):
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    memo: dict = {}
    model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                             lr=lr, weight_decay=weight_decay, device=device, seed=0,
                             cache=cache, memo=memo, n_rounds=n_rounds,
                             coupling_kernel_size=coupling_kernel_size, min_n=3, fold_label="[abdesign] ")
    model.eval()
    ddg_ids, ddg_true, ddg_pred = [], [], []
    elisa_ids, elisa_true, elisa_pred = [], [], []
    with torch.no_grad():
        for ex in test:
            pred = predict(model, ex, cache, device, memo, hidden).item()
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
    parser.add_argument("--benchmarks", nargs="+", default=["rde"],
                         choices=["rde", "ab645", "ab1101", "s1131", "abdesign"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-rounds", type=int, default=4)
    parser.add_argument("--coupling-kernel-size", type=int, default=7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    if "rde" in args.benchmarks:
        print("\n--- RDE-PPI (SimbaOperatorModel: Response Propagation Operator) ---", flush=True)
        r = eval_rde(cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
                     args.n_rounds, args.coupling_kernel_size)
        print(r, flush=True)
        print("\nReference (RDE-PPI Table 1):", flush=True)
        for name, ref in RDE_REFERENCE.items():
            print(f"  {name}: per_struct P/S={ref['per_struct_pearson']:.3f}/{ref['per_struct_spearman']:.3f} "
                  f"overall P/S={ref['overall_pearson']:.3f}/{ref['overall_spearman']:.3f}", flush=True)

    for name in ("ab645", "ab1101", "s1131"):
        if name in args.benchmarks:
            print(f"\n--- {name} (SimbaOperatorModel) ---", flush=True)
            r = eval_protattba_one(cache, hidden, args.epochs, args.batch_size, args.lr,
                                    args.weight_decay, device, name, args.n_rounds, args.coupling_kernel_size)
            print(r, flush=True)

    if "abdesign" in args.benchmarks:
        print("\n--- AbDesign holdout (SimbaOperatorModel) ---", flush=True)
        r = eval_abdesign(cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device,
                           args.n_rounds, args.coupling_kernel_size)
        print(r, flush=True)


if __name__ == "__main__":
    main()
