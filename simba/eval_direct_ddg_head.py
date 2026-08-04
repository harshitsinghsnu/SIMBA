"""Direct-ddG-head comparison (reviewer question): the paper's headline
formulation is two-pass subtraction, ddG_pred = f(mutant) - f(wildtype), no
task-specific ddG architecture (simba/models/affinity.py's forward_ddg).
This script compares that against a joint readout that sees *both* pooled
representations at once and predicts ddG directly through a learned MLP,
rather than being constrained to exact subtraction:

  pooled_wt  = shared backbone -> projection -> interface memory -> pairformer
  pooled_mut = same shared pathway, mutant sequence
  ddG_pred   = MLP([pooled_wt, pooled_mut, pooled_wt - pooled_mut])

The shared pathway (projection/memory/pairformer weights) is identical
between the two models -- only the final head differs -- so any gap
isolates the readout choice, not a confound from a different backbone
representation. Evaluated on RDE-PPI's exact reproduced 3-fold split (same
benchmark/split used for the paper's per-benchmark training-target
clarification, Section 3.3), same hyperparameters as
simba/eval_ddg_affinity.py's `--benchmarks rde` run.
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

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_rde_skempi  # noqa: E402
from simba.eval_metrics import per_complex_metrics, pooled_metrics  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


class DirectDDGHeadModel(nn.Module):
    """Wraps an AbsoluteAffinityModel's shared pathway (projection, memory,
    pairformer -- everything up to the pooled representation, reusing
    `_pooled` directly) and replaces the scalar dG head with a joint ddG
    head that sees both the wildtype- and mutant-pooled representations at
    once, instead of subtracting two independent scalar dG predictions."""

    def __init__(self, plm_hidden_size=1280, d_model=256, mlp_hidden=128, dropout=0.1):
        super().__init__()
        self.backbone = AbsoluteAffinityModel(plm_hidden_size=plm_hidden_size, d_model=d_model)
        # The backbone's own scalar `head` is never called here -- only
        # `_pooled` (shared proj/memory/pairformer pathway) is reused.
        self.ddg_head = nn.Sequential(
            nn.Linear(3 * d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward_ddg(self, wt_embedding, mut_embedding, partner_embedding):
        pooled_wt = self.backbone._pooled(wt_embedding, partner_embedding)
        pooled_mut = self.backbone._pooled(mut_embedding, partner_embedding)
        joint = torch.cat([pooled_wt, pooled_mut, pooled_wt - pooled_mut], dim=-1)
        return self.ddg_head(joint).squeeze(-1)


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def predict(model, ex, cache, device, memo, mode):
    wt_full, mut_full, partner_seq = full_wt_mut_sequences(ex)
    wt_emb = embed(cache, wt_full, device, memo)
    mut_emb = embed(cache, mut_full, device, memo)
    partner_emb = (embed(cache, partner_seq, device, memo) if partner_seq
                   else torch.zeros(0, wt_emb.shape[-1], device=device))
    if mode == "subtraction":
        return model.forward_ddg(wt_emb, mut_emb, partner_emb)
    return model.forward_ddg(wt_emb, mut_emb, partner_emb)  # DirectDDGHeadModel has the same call signature


def evaluate(model, examples, cache, device, memo, mode, min_n=10):
    model.eval()
    ids, y_true, y_pred = [], [], []
    with torch.no_grad():
        for ex in examples:
            if ex.ddG is None:
                continue
            pred = predict(model, ex, cache, device, memo, mode).item()
            ids.append(ex.complex_id)
            y_true.append(ex.ddG)
            y_pred.append(pred)
    return {"overall": pooled_metrics(y_true, y_pred), "per_struct": per_complex_metrics(ids, y_true, y_pred, min_n=min_n),
            "predictions": {"ids": ids, "true": y_true, "pred": y_pred}}


def make_model(mode, hidden):
    if mode == "subtraction":
        return AbsoluteAffinityModel(plm_hidden_size=hidden)
    return DirectDDGHeadModel(plm_hidden_size=hidden)


def run_training(train, val, *, mode, hidden, epochs, batch_size, lr, weight_decay, device, seed, cache):
    torch.manual_seed(seed)
    memo: dict = {}
    model = make_model(mode, hidden).to(device)
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
            valid = [ex for ex in batch if ex.ddG is not None]
            if not valid:
                global_step += 1
                continue
            optimizer.zero_grad()
            n_valid = len(valid)
            batch_loss_value = 0.0
            for ex in valid:
                pred = predict(model, ex, cache, device, memo, mode)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
                batch_loss_value += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += batch_loss_value * n_valid
            n_seen += n_valid
            global_step += 1

        val_report = evaluate(model, val, cache, device, memo, mode, min_n=10)
        score = val_report["per_struct"]["mean_spearman"]
        if math.isnan(score):
            score = val_report["overall"]["spearman"]
        is_best = not math.isnan(score) and score > best_score
        if is_best:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{mode}] epoch {epoch}: train_mse={epoch_loss / max(n_seen, 1):.4f} val_score={score:.4f}"
              f"{'  <- new best' if is_best else ''}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_rde(mode, cache, hidden, epochs, batch_size, lr, weight_decay, device):
    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    all_ids, all_true, all_pred = [], [], []
    for fold in folds:
        memo: dict = {}
        print(f"\n=== [{mode}] RDE-PPI fold {fold} ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model = run_training(train, val, mode=mode, hidden=hidden, epochs=epochs, batch_size=batch_size,
                              lr=lr, weight_decay=weight_decay, device=device, seed=fold, cache=cache)
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pred = predict(model, ex, cache, device, memo, mode).item()
                all_ids.append(ex.complex_id)
                all_true.append(ex.ddG)
                all_pred.append(pred)
    return {"overall": pooled_metrics(all_true, all_pred),
            "per_struct": per_complex_metrics(all_ids, all_true, all_pred, min_n=10),
            "predictions": {"ids": all_ids, "true": all_true, "pred": all_pred}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="direct_ddg_head_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = {}
    for mode in ("subtraction", "direct_head"):
        r = eval_rde(mode, cache, hidden, args.epochs, args.batch_size, args.lr, args.weight_decay, device)
        print(f"\n[{mode}] Overall Pearson/Spearman={r['overall']['pearson']:.4f}/{r['overall']['spearman']:.4f}  "
              f"Per-Struct Pearson/Spearman={r['per_struct']['mean_pearson']:.4f}/{r['per_struct']['mean_spearman']:.4f}",
              flush=True)
        results[mode] = r

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
