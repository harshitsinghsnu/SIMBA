"""Flamingo-style gating ablation (reviewer question): the proposed
architecture's *only* gating is inside the Pairformer-lite block's row- and
column-attention residual updates (simba/models/simba.py's PairformerLayer,
`gate_row`/`gate_col`, each a learnable scalar passed through tanh,
initialized at 0). The interface memory's write step (GatedMemory.write) has
no gating mechanism in the proposed model at all -- GatedMemory separately
defines a gated *read* operation, but that is only used by the separate
no-pairformer single-query-readout ablation, never by the proposed
architecture evaluated here.

This script compares the proposed model (`gated=True`, imports the real,
unmodified `AbsoluteAffinityModel`) against an ablated variant with the
Pairformer's row/col attention gates removed (`gated=False`, the same
attention outputs added with a fixed coefficient of 1.0 instead of
tanh(learnable gate)) -- isolating specifically whether that internal
gating helps, on the same cold-target-split protocol used throughout the
paper's ablation (Figure 3C / panel2C's split; simba/eval_skempi2_5fold.py).

The ungated classes are defined locally in this file (not edited into
simba/models/simba.py) since gating is used by every other model/ablation in
the project and this is a one-off comparison, not a permanent architecture
option.
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
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.eval_skempi2_5fold import embed, make_folds  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.simba import make_projection  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


class UngatedMemory(nn.Module):
    """Same as simba.models.simba.GatedMemory, but the read step's
    contribution uses a fixed coefficient of 1.0 instead of tanh(learnable
    gate)."""

    def __init__(self, d_model: int, n_heads: int, n_memory: int):
        super().__init__()
        self.memory_init = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.write_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.read_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def write(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[0] == 0:
            return self.memory_init
        kv = context.unsqueeze(0)
        mem_out, _ = self.write_attn(self.memory_init.unsqueeze(0), kv, kv)
        return mem_out.squeeze(0)

    def read(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem_kv = memory.unsqueeze(0)
        read_out, _ = self.read_attn(query.unsqueeze(0), mem_kv, mem_kv)
        return read_out.mean(dim=1).squeeze(0)


class UngatedPairformerLayer(nn.Module):
    """Same as simba.models.simba.PairformerLayer, no gates."""

    def __init__(self, pair_dim: int, n_heads: int):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        row_out, _ = self.row_attn(pair, pair, pair)
        pair = pair + row_out
        pair_t = pair.transpose(0, 1)
        col_out, _ = self.col_attn(pair_t, pair_t, pair_t)
        pair_t = pair_t + col_out
        return pair_t.transpose(0, 1)


class UngatedPairwiseInterfaceBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pair_hidden: int, dropout: float, n_layers: int = 1):
        super().__init__()
        self.pair_dim = pair_hidden
        self.to_pair = nn.Linear(2 * d_model, pair_hidden)
        self.layers = nn.ModuleList([UngatedPairformerLayer(pair_hidden, n_heads) for _ in range(n_layers)])
        self.from_pair = nn.Linear(pair_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, self_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        Ls, M = self_tokens.shape[0], memory.shape[0]
        self_exp = self_tokens.unsqueeze(1).expand(Ls, M, -1)
        mem_exp = memory.unsqueeze(0).expand(Ls, M, -1)
        pair = self.to_pair(torch.cat([self_exp, mem_exp], dim=-1))
        for layer in self.layers:
            pair = layer(pair)
        pooled = pair.mean(dim=(0, 1))
        return self.drop(self.from_pair(pooled))


class UngatedAffinityModel(nn.Module):
    """Minimal reimplementation of AbsoluteAffinityModel's default
    (pairformer, mlp head, no epinet, no hl-cross-attn) configuration with
    every gate replaced by a fixed 1.0 coefficient -- otherwise identical
    shapes/hyperparameters, so any performance difference isolates the
    effect of gating alone."""

    def __init__(self, plm_hidden_size=1280, d_model=256, proj_mid_dim=512, n_heads=4,
                 n_memory=24, pair_hidden=64, mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.partner_memory = UngatedMemory(d_model, n_heads, n_memory)
        self.pairwise_block = UngatedPairwiseInterfaceBlock(d_model, n_heads, pair_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_tokens = self.proj(self_embedding)
        partner_tokens = self.proj(partner_embedding)
        memory = self.partner_memory.write(partner_tokens)
        pooled = self.pairwise_block(self_tokens, memory)
        return self.head(pooled).squeeze(-1)


def make_model(gated: bool, hidden: int):
    if gated:
        from simba.models.affinity import AbsoluteAffinityModel
        return AbsoluteAffinityModel(plm_hidden_size=hidden)
    return UngatedAffinityModel(plm_hidden_size=hidden)


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


def run_fold(train, val, test, *, gated, hidden, epochs, batch_size, lr, weight_decay, device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    memo: dict = {}
    model = make_model(gated, hidden).to(device)
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
                ((pred - target) ** 2 / n_valid).backward()
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


def run_config(gated, examples, folds, *, hidden, epochs, batch_size, lr, weight_decay,
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
        preds, trues, ids = run_fold(train, val, test, gated=gated, hidden=hidden, epochs=epochs,
                                      batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                      device=device, seed=seed * 100 + fold_idx, cache=cache)
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        all_ids.extend(ids)
        print(f"  [gated={gated}] fold {fold_idx}: n={len(test)}", flush=True)

    all_preds_arr, all_trues_arr = np.asarray(all_preds), np.asarray(all_trues)
    # Raw per-example predictions are saved alongside the summary metrics so
    # bootstrap CIs / figures can be recomputed later without retraining.
    return {
        "gated": gated,
        "pearson": float(pearsonr(all_trues_arr, all_preds_arr)[0]),
        "spearman": float(spearmanr(all_trues_arr, all_preds_arr)[0]),
        "rmse": float(np.sqrt(np.mean((all_trues_arr - all_preds_arr) ** 2))),
        "n": len(all_trues_arr),
        "predictions": {"ids": all_ids, "true": all_trues, "pred": all_preds},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["complex", "seq_similarity", "cold_target"], default="cold_target")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="gate_ablation_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_combined_examples()
    folds = make_folds(examples, split_type=args.split)
    print(f"[gate_ablation] split={args.split} {len(examples)} examples, "
          f"{len(set(e.complex_id.upper() for e in examples))} complexes", flush=True)

    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = []
    for gated in (True, False):
        r = run_config(gated, examples, folds, hidden=hidden, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                        val_frac=args.val_frac, device=device, seed=args.seed, cache=cache)
        print(f"[gated={gated}] Pearson={r['pearson']:.4f} Spearman={r['spearman']:.4f} "
              f"RMSE={r['rmse']:.4f} (n={r['n']})", flush=True)
        results.append(r)
        # Save after every config, not just at the end -- a crash on the
        # second config used to lose the first one's results too.
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "results": results}, f, indent=2)

    with open(args.out, "w") as f:
        json.dump({"split": args.split, "results": results}, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
