"""Interface-memory-size sensitivity sweep (reviewer question): re-runs the
existing cold-target-split ablation protocol (simba/eval_skempi2_5fold.py,
--split cold_target -- the same protocol behind Figure 3C / panel2C) while
varying only `AbsoluteAffinityModel`'s `n_memory` argument (default 24 in
every other result in the paper), holding everything else fixed. Reuses
`make_folds`/`embed` from eval_skempi2_5fold.py directly so the fold
assignment is byte-identical to every other cold_target result already
reported.
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
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.eval_skempi2_5fold import embed, make_folds  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402
from simba.models.simba import PairwiseInterfaceBlock, make_projection  # noqa: E402
import torch.nn as nn  # noqa: E402


class NoMemoryModel(nn.Module):
    """Memory ablation: same projection, Pairformer-lite and head as SIMBA,
    but the 24-slot memory is removed and the Pairformer runs on the full
    L_self x L_partner pair map built from the raw projected partner tokens
    (cost grows with partner length; partners are capped at `max_partner`/`max_self`
    tokens, by uniform subsampling, to fit in memory)."""

    def __init__(self, plm_hidden_size, d_model=256, proj_mid_dim=512, n_heads=4, pair_hidden=64,
                 mlp_hidden=128, dropout=0.1, n_layers=1, max_partner=128, max_self=256):
        super().__init__()
        self.max_partner, self.max_self = max_partner, max_self
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.pairwise_block = PairwiseInterfaceBlock(d_model, n_heads, pair_hidden, dropout, n_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1))

    def forward(self, self_embedding, partner_embedding):
        partner = self.proj(partner_embedding)
        if partner.shape[0] > self.max_partner:
            idx = torch.linspace(0, partner.shape[0] - 1, self.max_partner, device=partner.device).long()
            partner = partner[idx]
        own = self.proj(self_embedding)
        if own.shape[0] > self.max_self:
            idx = torch.linspace(0, own.shape[0] - 1, self.max_self, device=own.device).long()
            own = own[idx]
        return self.head(self.pairwise_block(own, partner)).squeeze(-1)


def make_model(variant, hidden, n_memory, n_layers):
    if variant == "memory":          # proposed SIMBA (n_memory slots, Pairformer-lite)
        return AbsoluteAffinityModel(plm_hidden_size=hidden, n_memory=n_memory, n_pairformer_layers=n_layers)
    if variant == "memory_only":     # memory + single-query readout, no Pairformer
        return AbsoluteAffinityModel(plm_hidden_size=hidden, n_memory=n_memory, use_pairformer=False)
    if variant == "no_memory":       # Pairformer on raw partner tokens, no memory
        return NoMemoryModel(hidden, n_layers=n_layers)
    raise ValueError(variant)


def evaluate(model, examples, cache, device, memo):
    model.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed(cache, ex.self_sequence, device, memo)
            partner_emb = embed(cache, ex.partner_sequence, device, memo)
            preds.append(model(self_emb, partner_emb).item())
            trues.append(ex.dG)
            ids.append(ex.complex_id)
    return np.asarray(preds), np.asarray(trues), ids


def run_fold(train, val, test, *, variant, n_layers, hidden, n_memory, epochs, batch_size, lr, weight_decay,
             device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    memo: dict = {}
    model = make_model(variant, hidden, n_memory, n_layers).to(device)
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
            n_valid = len(batch)
            for ex in batch:
                self_emb = embed(cache, ex.self_sequence, device, memo)
                partner_emb = embed(cache, ex.partner_sequence, device, memo)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                (( pred - target) ** 2 / n_valid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_preds, val_trues, _ = evaluate(model, val, cache, device, memo)
        val_pcc = float(pearsonr(val_trues, val_preds)[0]) if np.std(val_preds) > 0 else float("nan")
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, test, cache, device, memo)


def run_sweep_value(examples, folds, *, variant, n_layers, n_memory, hidden, epochs, batch_size, lr, weight_decay,
                     val_frac, device, seed, cache):
    all_preds, all_trues, all_ids = [], [], []
    rng = random.Random(seed)
    for fold_idx, test_ids in enumerate(folds):
        test = [e for e in examples if e.complex_id in test_ids]
        train_val = [e for e in examples if e.complex_id not in test_ids]
        idx = list(range(len(train_val)))
        rng.shuffle(idx)
        n_val = max(1, int(val_frac * len(idx)))
        val = [train_val[i] for i in idx[:n_val]]
        train = [train_val[i] for i in idx[n_val:]]
        preds, trues, ids = run_fold(train, val, test, variant=variant, n_layers=n_layers, hidden=hidden, n_memory=n_memory, epochs=epochs,
                                      batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                      device=device, seed=seed * 100 + fold_idx, cache=cache)
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        all_ids.extend(ids)
        print(f"  [n_memory={n_memory}] fold {fold_idx}: n={len(test)}", flush=True)

    all_preds_arr, all_trues_arr = np.asarray(all_preds), np.asarray(all_trues)
    pcc = float(pearsonr(all_trues_arr, all_preds_arr)[0])
    scc = float(spearmanr(all_trues_arr, all_preds_arr)[0])
    rmse = float(np.sqrt(np.mean((all_trues_arr - all_preds_arr) ** 2)))
    # Raw per-example predictions are saved alongside the summary metrics so
    # bootstrap CIs / figures can be recomputed later without retraining.
    return {"variant": variant, "n_layers": n_layers, "n_memory": n_memory, "pearson": pcc, "spearman": scc, "rmse": rmse, "n": len(all_trues),
            "predictions": {"ids": all_ids, "true": all_trues, "pred": all_preds}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-sizes", type=int, nargs="+", default=[8, 16, 24, 32, 48])
    parser.add_argument("--variant", choices=["memory", "memory_only", "no_memory"], default="memory")
    parser.add_argument("--layers", type=int, default=1, help="Pairformer-lite depth (memory / no_memory)")
    parser.add_argument("--split", choices=["complex", "seq_similarity", "cold_target"], default="cold_target")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="memory_size_sweep_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_combined_examples()
    folds = make_folds(examples, split_type=args.split)
    print(f"[memory_size_sweep] split={args.split} {len(examples)} examples, "
          f"{len(set(e.complex_id.upper() for e in examples))} complexes, "
          f"memory_sizes={args.memory_sizes}", flush=True)

    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = []
    for n_memory in args.memory_sizes:
        r = run_sweep_value(examples, folds, variant=args.variant, n_layers=args.layers, n_memory=n_memory, hidden=hidden, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                             val_frac=args.val_frac, device=device, seed=args.seed, cache=cache)
        print(f"[n_memory={n_memory}] Pearson={r['pearson']:.4f} Spearman={r['spearman']:.4f} "
              f"RMSE={r['rmse']:.4f} (n={r['n']})", flush=True)
        results.append(r)
        # Save after every sweep value, not just at the end -- a crash/OOM on
        # a later value (as happened at n_memory=48) used to lose every
        # already-completed result too, since json.dump only ran once after
        # the whole loop finished.
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "results": results}, f, indent=2)
        print(f"  (incremental save: {args.out} now has {len(results)}/{len(args.memory_sizes)} values)",
              flush=True)

    with open(args.out, "w") as f:
        json.dump({"split": args.split, "results": results}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
