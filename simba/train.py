"""Minimal training loop for SIMBA. No batched tensor ops across examples --
given the frozen PLM backbone is precomputed/cached, per-example forward
passes over small cached slices are already fast at this dataset scale
(thousands, not millions, of rows), so the loop favors simplicity/correctness
over batching sophistication.

`run_training` is the reusable core (used directly by this script's CLI, and
by simba/eval_rde_folds.py and simba/eval_protattba.py for cross-validation
against exact SOTA splits) -- it owns the optimizer, LR schedule, gradient
clipping, and validation-based best-checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import math
import random

import torch
import torch.nn.functional as F

from simba import env

env.setup()

from simba.data.dataset import load_all_mutations  # noqa: E402
from simba.data.featurize import FeatureProvider  # noqa: E402
from simba.eval_metrics import full_report  # noqa: E402
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.simba import SimbaModel  # noqa: E402

SPLIT_COLUMNS = {
    "random": "split_random",
    "cluster": "split_cluster",
    "abdesign_holdout": "split_abdesign_holdout",
}


def split_examples(examples, protocol: str, seed: int = 0):
    col = SPLIT_COLUMNS[protocol]
    if protocol == "abdesign_holdout":
        train_pool = [e for e in examples if e.meta.get(col) == "train_pool"]
        eval_set = [e for e in examples if e.meta.get(col) == "eval_abdesign"]
        rng = random.Random(seed)
        idx = list(range(len(train_pool)))
        rng.shuffle(idx)
        n_val = max(1, int(0.1 * len(idx)))
        val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
        train = [train_pool[i] for i in train_idx]
        val = [train_pool[i] for i in val_idx]
        return train, val, eval_set
    train = [e for e in examples if e.meta.get(col) == "train"]
    val = [e for e in examples if e.meta.get(col) == "val"]
    test = [e for e in examples if e.meta.get(col) == "test"]
    return train, val, test


def evaluate(model, provider, examples, label: str, min_complex_n: int = 3, verbose: bool = True):
    model.eval()
    complex_ids, y_true, y_pred = [], [], []
    elisa_true, elisa_pred, elisa_ids = [], [], []
    with torch.no_grad():
        for ex in examples:
            sites = provider.featurize(ex)
            if any(s["delta_logp"] is None for s in sites):
                continue
            pred = model(sites).item()
            if ex.ddG is not None:
                complex_ids.append(ex.complex_id)
                y_true.append(ex.ddG)
                y_pred.append(pred)
            elif ex.elisa_ratio is not None:
                elisa_ids.append(ex.complex_id)
                elisa_true.append(ex.elisa_ratio)
                elisa_pred.append(pred)

    report = full_report(complex_ids, y_true, y_pred, min_complex_n=min_complex_n) if y_true else None
    if verbose:
        print(f"[{label}] ddG-labeled n={len(y_true)}: {report}")
    if elisa_true:
        elisa_report = full_report(elisa_ids, elisa_true, elisa_pred, min_complex_n=min_complex_n)
        if verbose:
            print(f"[{label}] elisa-ratio-labeled n={len(elisa_true)} "
                  f"(negative correlation expected -- higher predicted ddG should mean lower ratio): {elisa_report}")
        return report, elisa_report
    return report, None


def lr_at_step(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.1) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_training(
    train, val, test,
    *,
    provider: FeatureProvider,
    plm_hidden_size: int,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    warmup_frac: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
    model_kwargs: dict | None = None,
    ranking_loss_weight: float = 0.0,
):
    """Trains a fresh SimbaModel, returns (model, best_val_report, test_report).

    `ranking_loss_weight` (TerraBind-inspired, default 0.0 = off): adds a
    Huber loss on (pred_i - pred_j) vs (true_i - true_j) for every pair of
    examples in the same training batch that share a complex_id, on top of
    the ordinary per-example MSE. Targets per-complex Spearman directly
    (the primary metric used throughout this project) rather than only
    absolute-value accuracy; zero when a batch has no same-complex pairs, so
    it is strictly additive and never changes the loss when weight=0."""
    model = SimbaModel(plm_hidden_size=plm_hidden_size, **(model_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs

    best_score = -math.inf
    best_state = None
    global_step = 0

    rng = random.Random(seed)
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, lr, warmup_frac)
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=device)
            n_valid = 0
            preds_by_complex: dict = {}
            for ex in batch:
                if ex.ddG is None:
                    continue
                sites = provider.featurize(ex)
                if any(s["delta_logp"] is None for s in sites):
                    continue
                pred = model(sites)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                batch_loss = batch_loss + (pred - target) ** 2
                n_valid += 1
                if ranking_loss_weight > 0:
                    preds_by_complex.setdefault(ex.complex_id, []).append((pred, target))
            if n_valid == 0:
                global_step += 1
                continue
            batch_loss = batch_loss / n_valid

            if ranking_loss_weight > 0:
                rank_loss = torch.zeros((), device=device)
                n_pairs = 0
                for group in preds_by_complex.values():
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            pred_i, true_i = group[i]
                            pred_j, true_j = group[j]
                            rank_loss = rank_loss + F.smooth_l1_loss(pred_i - pred_j, true_i - true_j)
                            n_pairs += 1
                if n_pairs > 0:
                    batch_loss = batch_loss + ranking_loss_weight * (rank_loss / n_pairs)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += batch_loss.item() * n_valid
            n_seen += n_valid
            global_step += 1

        if verbose:
            print(f"epoch {epoch}: train_mse={epoch_loss / max(n_seen, 1):.4f} (n={n_seen}) "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        report, _ = evaluate(model, provider, val, label=f"val@epoch{epoch}", verbose=verbose)
        score = report["per_complex"]["mean_spearman"] if report else float("-inf")
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"  -> new best (val per-complex mean spearman={score:.4f}), checkpointed")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"loaded best checkpoint (val per-complex mean spearman={best_score:.4f}) for final evaluation")

    test_report, test_elisa_report = evaluate(model, provider, test, label="final_test", verbose=verbose)
    return model, best_score, test_report, test_elisa_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-protocol", choices=list(SPLIT_COLUMNS), default="cluster")
    parser.add_argument("--backbone", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="debug: cap training set size for a quick smoke test")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-masked-marginal", action="store_true",
                         help="use single-pass wildtype-marginal scoring instead of masked-marginal")
    parser.add_argument("--core", choices=["static", "recurrent"], default="static",
                         help="static = dual gated memory + explicit pairwise MLP; "
                              "recurrent = shared interaction field with iterative site<->field message passing")
    parser.add_argument("--n-rounds", type=int, default=3, help="recurrent core: refinement rounds")
    parser.add_argument("--chain-mode", default="concat",
                         choices=["dual_memory", "concat", "self_attn", "none", "pairformer"])
    parser.add_argument("--use-epinet", action="store_true",
                         help="TerraBind-inspired epistemic-index residual head for calibrated uncertainty")
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0,
                         help="TerraBind-inspired within-complex pairwise ranking loss weight (0 = off)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_all_mutations()
    train, val, test = split_examples(examples, args.split_protocol)
    if args.max_train_examples:
        train = train[: args.max_train_examples]
    print(f"train={len(train)} val={len(val)} test/eval={len(test)}")

    scorer = ESMScorer(model_name=args.backbone, device=device)
    cache = SequenceScoreCache(scorer)
    masked_cache = None if args.no_masked_marginal else MaskedMarginalCache(scorer)
    provider = FeatureProvider(cache, device, masked_cache=masked_cache)

    run_training(
        train, val, test,
        provider=provider, plm_hidden_size=scorer.model.config.hidden_size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        warmup_frac=args.warmup_frac, device=device,
        model_kwargs={"core": args.core, "n_rounds": args.n_rounds,
                      "chain_mode": args.chain_mode, "use_epinet": args.use_epinet},
        ranking_loss_weight=args.ranking_loss_weight,
    )


if __name__ == "__main__":
    main()
