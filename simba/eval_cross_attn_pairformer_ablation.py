"""New ablation: does adding genuine self<->partner(memory) cross-attention
on top of the Pairformer-lite refinement help? In the proposed architecture,
self tokens and the interface memory never actually cross-attend to each
other -- they only "meet" via broadcast + concatenation + a linear
projection (simba/models/simba.py's PairwiseInterfaceBlock.forward), and
everything attention-based after that (the Pairformer's row/column
attention) is self-attention *within* the resulting pairwise grid, not
between self and partner. The only real cross-attention anywhere in the
proposed model is the earlier memory *write* step (partner tokens -> 24
latents).

This script adds one more step before the existing pairwise-grid
construction: self tokens cross-attend directly to the 24-slot memory
(self tokens as query, memory as key/value, gated Flamingo-style, matching
this project's convention), producing partner-aware self tokens that are
then broadcast/concatenated with the memory and refined by the unchanged
Pairformer-lite block exactly as before. This isolates whether explicit
self<->partner cross-attention adds anything on top of what the existing
design already gets from pure self-attention refinement of the
broadcast-built grid.

Compared against the proposed model, on the same cold-target-split
protocol used throughout the paper's ablation (Figure 3C / panel2C;
simba/eval_skempi2_5fold.py).
"""

from __future__ import annotations

import os

# expandable_segments is a Linux-only CUDA allocator feature; leaving it set
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
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.models.simba import GatedMemory, PairwiseInterfaceBlock, make_projection  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


class CrossAttnPairformerModel(nn.Module):
    """AbsoluteAffinityModel's default (pairformer, mlp head) configuration
    with one new step inserted: self tokens cross-attend to the interface
    memory (self as query, memory as key/value, Flamingo-gated) before the
    unchanged PairwiseInterfaceBlock/Pairformer-lite refinement runs on the
    cross-attended self tokens + memory. Everything else -- projection,
    GatedMemory.write, PairformerLayer, MLP head -- is the identical,
    unmodified code SIMBA already uses, so any performance difference
    isolates the effect of adding this one cross-attention step."""

    def __init__(self, plm_hidden_size=1280, d_model=256, proj_mid_dim=512, n_heads=4,
                 n_memory=24, pair_hidden=64, mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.partner_memory = GatedMemory(d_model, n_heads, n_memory)
        self.self_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_gate = nn.Parameter(torch.tensor(0.0))
        self.pairwise_block = PairwiseInterfaceBlock(d_model, n_heads, pair_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_tokens = self.proj(self_embedding)          # [Ls, d_model]
        partner_tokens = self.proj(partner_embedding)    # [Lp, d_model]
        memory = self.partner_memory.write(partner_tokens)  # [M, d_model], ungated (unchanged)

        # New: genuine self<->partner(memory) cross-attention, gated so
        # training starts equivalent to the unmodified proposed model
        # (tanh(0)=0 => self_tokens passed through unchanged initially).
        q = self_tokens.unsqueeze(0)
        kv = memory.unsqueeze(0)
        attn_out, _ = self.self_cross_attn(q, kv, kv)
        self_tokens = self_tokens + torch.tanh(self.cross_attn_gate) * attn_out.squeeze(0)

        pooled = self.pairwise_block(self_tokens, memory)  # unchanged Pairformer-lite
        return self.head(pooled).squeeze(-1)


def make_model(variant: str, hidden: int):
    if variant == "proposed":
        return AbsoluteAffinityModel(plm_hidden_size=hidden)
    return CrossAttnPairformerModel(plm_hidden_size=hidden)


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


def run_fold(train, val, test, *, variant, hidden, epochs, batch_size, lr, weight_decay, device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    memo: dict = {}
    model = make_model(variant, hidden).to(device)
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


def run_variant(variant, examples, folds, *, hidden, epochs, batch_size, lr, weight_decay,
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
        preds, trues, ids = run_fold(train, val, test, variant=variant, hidden=hidden, epochs=epochs,
                                      batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                                      device=device, seed=seed * 100 + fold_idx, cache=cache)
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        all_ids.extend(ids)
        print(f"  [{variant}] fold {fold_idx}: n={len(test)}", flush=True)

    all_preds_arr, all_trues_arr = np.asarray(all_preds), np.asarray(all_trues)
    return {
        "variant": variant,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                         help="Multiple seeds so the single-seed finding (cross_attn_pairformer "
                              "beating proposed on all 3 metrics) can be confirmed rather than "
                              "reported on noise -- matches the 3-seed convention used for every "
                              "other headline result in this project.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="cross_attn_pairformer_ablation_multiseed_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_combined_examples()
    folds = make_folds(examples, split_type=args.split)
    print(f"[cross_attn_pairformer_ablation] split={args.split} {len(examples)} examples, "
          f"{len(set(e.complex_id.upper() for e in examples))} complexes, seeds={args.seeds}", flush=True)

    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = {}
    for variant in ("proposed", "cross_attn_pairformer"):
        per_seed = []
        for seed in args.seeds:
            r = run_variant(variant, examples, folds, hidden=hidden, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                             val_frac=args.val_frac, device=device, seed=seed, cache=cache)
            r["seed"] = seed
            print(f"[{variant}] seed={seed}: Pearson={r['pearson']:.4f} Spearman={r['spearman']:.4f} "
                  f"RMSE={r['rmse']:.4f} (n={r['n']})", flush=True)
            per_seed.append(r)
            # Save after every single seed, not just at the end or per variant --
            # a crash partway through (as happened with the memory-size sweep)
            # must not lose already-completed seeds.
            results[variant] = {
                "per_seed": per_seed,
                "pearson_mean": float(np.mean([x["pearson"] for x in per_seed])),
                "pearson_std": float(np.std([x["pearson"] for x in per_seed])),
                "spearman_mean": float(np.mean([x["spearman"] for x in per_seed])),
                "spearman_std": float(np.std([x["spearman"] for x in per_seed])),
                "rmse_mean": float(np.mean([x["rmse"] for x in per_seed])),
                "rmse_std": float(np.std([x["rmse"] for x in per_seed])),
            }
            with open(args.out, "w") as f:
                json.dump({"split": args.split, "results": results}, f, indent=2)
        r_summary = results[variant]
        print(f"[{variant}] SUMMARY over {len(args.seeds)} seeds: "
              f"Pearson={r_summary['pearson_mean']:.4f}+/-{r_summary['pearson_std']:.4f} "
              f"Spearman={r_summary['spearman_mean']:.4f}+/-{r_summary['spearman_std']:.4f} "
              f"RMSE={r_summary['rmse_mean']:.4f}+/-{r_summary['rmse_std']:.4f}", flush=True)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
