"""Concat+MLP baseline (simba/models/concat_mlp.py) vs. the proposed
pairformer architecture, on the two confirmed-winning benchmarks
(benchmark46 absolute affinity, RDE-PPI ddG) -- same data, same splits,
same training loop shape as the pairformer runs, only the model swapped.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random

import numpy as np
import torch
from scipy.stats import pearsonr

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.build_affinity_benchmark import build_skempi2_wt_train_set, build_test_set  # noqa: E402
from simba.data.dataset import load_rde_skempi  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.concat_mlp import ConcatMLPModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float()
    return memo[sequence].to(device)


def run_affinity(cache, hidden, epochs, batch_size, lr, weight_decay, val_frac, device, seed):
    test = build_test_set()
    test_ids = {e.complex_id for e in test}
    train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]

    memo: dict = {}
    model = ConcatMLPModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_pcc, best_state, global_step = -math.inf, None, 0

    def evaluate(examples):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ex in examples:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                preds.append(model(self_emb, partner_emb).item())
                trues.append(ex.dG)
        preds, trues = np.asarray(preds), np.asarray(trues)
        pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        mae = float(np.mean(np.abs(trues - preds)))
        return pcc, mae

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, lr)
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            for ex in batch:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                batch_loss = batch_loss + (pred - target) ** 2
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        val_pcc, _ = evaluate(val)
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pcc, test_mae = evaluate(test)
    return {"pcc": test_pcc, "mae": test_mae, "n": len(test)}


def run_rde(cache, hidden, epochs, batch_size, lr, weight_decay, device, seed):
    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    all_ids, all_true, all_pred = [], [], []

    for fold in folds:
        memo: dict = {}
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)

        torch.manual_seed(seed * 100 + fold)
        rng = random.Random(seed * 100 + fold)
        model = ConcatMLPModel(plm_hidden_size=hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
        total_steps = steps_per_epoch * epochs
        best_score, best_state, global_step = -math.inf, None, 0

        def predict(ex):
            wt_full, mut_full, partner_seq = full_wt_mut_sequences(ex)
            wt_emb = embed(cache, wt_full, device, memo)
            mut_emb = embed(cache, mut_full, device, memo)
            partner_emb = (embed(cache, partner_seq, device, memo) if partner_seq
                           else torch.zeros(0, hidden, device=device))
            return model.forward_ddg(wt_emb, mut_emb, partner_emb)

        def evaluate(examples, min_n=10):
            model.eval()
            ids, y_true, y_pred = [], [], []
            with torch.no_grad():
                for ex in examples:
                    if ex.ddG is None:
                        continue
                    ids.append(ex.complex_id)
                    y_true.append(ex.ddG)
                    y_pred.append(predict(ex).item())
            return {"overall": pooled_metrics(y_true, y_pred),
                    "per_struct": per_complex_metrics(ids, y_true, y_pred, min_n=min_n)}

        for epoch in range(epochs):
            model.train()
            rng.shuffle(train)
            for start in range(0, len(train), batch_size):
                batch = train[start:start + batch_size]
                valid = [ex for ex in batch if ex.ddG is not None]
                if not valid:
                    global_step += 1
                    continue
                for g in optimizer.param_groups:
                    g["lr"] = lr_at_step(global_step, total_steps, lr)
                optimizer.zero_grad()
                n_valid = len(valid)
                for ex in valid:
                    pred = predict(ex)
                    target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                    loss = (pred - target) ** 2 / n_valid
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            val_report = evaluate(val)
            score = val_report["per_struct"]["mean_spearman"]
            if math.isnan(score):
                score = val_report["overall"]["spearman"]
            if not math.isnan(score) and score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(predict(ex).item())

    return {"overall": pooled_metrics(all_true, all_pred),
            "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--benchmarks", nargs="+", default=["benchmark46", "rde"])
    parser.add_argument("--epochs-affinity", type=int, default=60)
    parser.add_argument("--epochs-rde", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="concat_mlp_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = {}
    if "benchmark46" in args.benchmarks:
        r = run_affinity(cache, hidden, args.epochs_affinity, args.batch_size, args.lr,
                          args.weight_decay, args.val_frac, device, args.seed)
        results["benchmark46"] = r
        print(f"[concat_mlp] benchmark46: PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)

    if "rde" in args.benchmarks:
        r = run_rde(cache, hidden, args.epochs_rde, args.batch_size, args.lr, args.weight_decay,
                     device, args.seed)
        results["rde"] = r
        print(f"[concat_mlp] RDE-PPI: overall={r['overall']} per_struct={r['per_struct']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved to {args.out}]")


if __name__ == "__main__":
    main()
