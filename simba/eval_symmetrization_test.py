"""Chain-order symmetrization test (reviewer question): AbsoluteAffinityModel
is architecturally asymmetric in self vs. partner roles (partner writes into
the interface memory, self reads from it via the Pairformer -- see
simba/models/affinity.py's module docstring and paper/simba.tex's Limitations
section). This script trains the model exactly as in
eval_affinity_extended.py's benchmark46 protocol, then at *inference time
only* checks:

  1. f(A, B)  -- the model's ordinary prediction (self=A, partner=B).
  2. f(B, A)  -- the same trained model with the two chains swapped.
  3. (f(A,B) + f(B,A)) / 2 -- a symmetrized prediction (no retraining, just
     averaging two forward passes at eval time).

reported against the true dG for the Kastritis-lineage (benchmark46) test
set, across the same 3 seeds used for the paper's headline result, so the
answer is directly comparable to the reported 0.788+/-0.008 Pearson.

No architecture change and no new training objective -- this only tests
whether the *existing* trained model happens to be close to swap-invariant,
and whether cheap post-hoc averaging helps.
"""

from __future__ import annotations

import os

# expandable_segments is a Linux-only CUDA allocator feature; leaving it set
# (e.g. inherited from a shell that previously ran eval_plm_ablation_cold.py)
# only produces a noisy UserWarning on Windows PyTorch builds. Must be set
# before `import torch` to take effect.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

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

from simba.data.build_affinity_benchmark import build_skempi2_wt_train_set, build_test_set  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def evaluate_all_orders(model, examples, cache, device, memo):
    """Returns three prediction arrays (forward, swapped, averaged) plus the
    true dG array, for every test example."""
    model.eval()
    fwd, swp, true = [], [], []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed(cache, ex.self_sequence, device, memo)
            partner_emb = embed(cache, ex.partner_sequence, device, memo)
            fwd.append(model(self_emb, partner_emb).item())
            swp.append(model(partner_emb, self_emb).item())
            true.append(ex.dG)
    fwd, swp, true = np.asarray(fwd), np.asarray(swp), np.asarray(true)
    avg = (fwd + swp) / 2.0
    return fwd, swp, avg, true


def run_one_seed(train_full, test, *, hidden, epochs, batch_size, lr, weight_decay,
                  val_frac, device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]
    memo: dict = {}

    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
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
            batch_loss = torch.zeros((), device=device)
            for ex in batch:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                batch_loss = batch_loss + (pred - target) ** 2
            (batch_loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        model.eval()
        val_fwd, _, _, val_true = evaluate_all_orders(model, val, cache, device, memo)
        val_pcc = float(pearsonr(val_true, val_fwd)[0]) if np.std(val_fwd) > 0 else float("nan")
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate_all_orders(model, test, cache, device, memo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="symmetrization_test_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    test = build_test_set()
    test_ids_set = {e.complex_id for e in test}
    train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids_set)
    test_complex_ids = [e.complex_id for e in test]
    print(f"[symmetrization] test={len(test)} complexes, train_pool={len(train_full)} complexes", flush=True)

    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    per_seed = []
    for seed in args.seeds:
        fwd, swp, avg, true = run_one_seed(
            train_full, test, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, val_frac=args.val_frac, device=device,
            seed=seed, cache=cache)
        pcc_fwd = float(pearsonr(true, fwd)[0])
        pcc_swp = float(pearsonr(true, swp)[0])
        pcc_avg = float(pearsonr(true, avg)[0])
        pcc_fwd_vs_swp = float(pearsonr(fwd, swp)[0])  # how (dis)similar the two orders are
        print(f"[seed {seed}] f(A,B) PCC={pcc_fwd:.4f}  f(B,A) PCC={pcc_swp:.4f}  "
              f"avg PCC={pcc_avg:.4f}  corr(f(A,B),f(B,A))={pcc_fwd_vs_swp:.4f}", flush=True)
        # Raw per-example predictions are saved (not just summary PCCs) so any
        # later figure/bootstrap-CI recomputation never needs to retrain --
        # same convention as every other *_results.json in this project.
        per_seed.append({
            "seed": seed, "pcc_forward": pcc_fwd, "pcc_swapped": pcc_swp, "pcc_averaged": pcc_avg,
            "corr_forward_vs_swapped": pcc_fwd_vs_swp,
            "ids": test_complex_ids, "true": true.tolist(),
            "pred_forward": fwd.tolist(), "pred_swapped": swp.tolist(), "pred_averaged": avg.tolist(),
        })

    def mean_std(key):
        vals = [r[key] for r in per_seed]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {k: dict(zip(("mean", "std"), mean_std(k)))
               for k in ("pcc_forward", "pcc_swapped", "pcc_averaged", "corr_forward_vs_swapped")}
    print("\n=== summary (mean +/- std over seeds) ===")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"per_seed": per_seed, "summary": summary}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
