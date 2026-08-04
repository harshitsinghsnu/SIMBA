"""PLM-backbone ablation on the cold_target split (MODEL_REPORT.md S3.20):
same combined general-PPI pool (7,247 examples, 541 complexes), same
cold_target fold construction (partner-sequence k-mer-Jaccard clusters,
5-fold complex-level CV), same `AbsoluteAffinityModel` (pairformer,
architecture unchanged) and training protocol as
`eval_skempi2_5fold.py --pool combined --split cold_target --model
pairformer` -- swapping only the frozen PLM backbone that produces the
per-residue embeddings fed into it:

  esm2     facebook/esm2_t33_650M_UR50D (default everywhere else in this
           project) -- independent per-sequence encoding, no cross-chain
           context at the PLM level.
  ablang2  Antibody-specific PLM (Olsen et al.). Independent per-sequence
           encoding, same as ESM2. This pool is general PPI, not just
           antibody-antigen, so most sequences here are genuinely
           out-of-domain for AbLang2 -- an intentional part of the
           ablation (does an antibody-specific representation transfer),
           not an oversight.
  mint     PPI-pretrained joint pair encoder (Ullanat et al., 96M STRING
           pairs). Architecturally different from the other two: encodes
           self+partner *jointly* in one forward pass with a shared
           chain-id tensor, so its per-residue embeddings are already
           cross-chain-aware before SIMBA's own pairformer ever sees them.
           This is MINT's actual intended usage, not a workaround -- see
           `simba/features/mint_plm.py` for exactly how the per-chain
           residue rows are recovered from the joint pass.

ProtBert was requested too but could not be included: it only publishes a
legacy pickle (.bin) checkpoint, and `transformers` (v5.4.0) refuses to
`torch.load` non-safetensors weights on torch<2.6 for a disclosed security
reason (CVE-2025-32434); this environment runs torch 2.5.0+cu118. Retried
directly for this ablation (not just assumed from the earlier ProtT5/Ankh
failure) and hit the identical error -- see MODEL_REPORT.md S3.9/S3.20.

Training config matches `eval_skempi2_5fold.py` exactly (epochs=40 ceiling,
batch_size=16, lr=1e-3, val-based checkpoint selection) plus real early
stopping on top (`--patience`, default 8 epochs with no new best val PCC) --
the existing checkpoint-selection logic alone doesn't save any wall-clock
time, since it still ran the full fixed epoch budget even after validation
plateaued. Early stopping cuts that dead time without changing the epoch
ceiling or the selected checkpoint for any backbone, so all three backbones
are compared under an identical training budget -- not a shortened one.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

# Must be set before `import torch` -- the CUDA caching allocator reads this
# once at init time, so setting it via the shell (`PYTORCH_CUDA_ALLOC_CONF=...
# python ...`) only works if the env var actually reaches this process
# before torch's first import; setting it here in-process is the only way
# to be sure it took effect, after a shell-level attempt produced an
# identical OOM signature (same 3.48 GiB allocation, same ~11GB baseline)
# and was never actually confirmed to have taken effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.eval_skempi2_5fold import make_folds  # noqa: E402
from simba.features.ablang2_plm import HIDDEN as ABLANG2_HIDDEN, AbLang2Scorer  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.mint_plm import HIDDEN as MINT_HIDDEN, MintPairCache, MINTPairScorer  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

N_FOLDS = 5
BACKBONE_HIDDEN = {"esm2": 1280, "ablang2": ABLANG2_HIDDEN, "mint": MINT_HIDDEN}


def build_backend(backbone: str, device: str):
    """Returns (hidden_size, get_embeddings(ex, device, memo) -> (self, partner),
    free_backbone_gpu_memory()). The third return value matters once
    pre-warming is done: after that point the backbone model itself is
    never touched again (every embedding comes from the disk cache), but
    it's still sitting on the GPU taking up memory it no longer needs --
    for MINT (a 650M-param model) that's enough to OOM training on a
    16GB GPU once the pairformer's own activations for a long sequence
    pair are added on top. Moving the backbone off the GPU after
    pre-warming frees that memory for training."""
    if backbone == "esm2":
        scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
        cache = SequenceScoreCache(scorer)

        def get(ex, device, memo):
            for seq in (ex.self_sequence, ex.partner_sequence):
                if seq not in memo:
                    memo[seq] = torch.from_numpy(cache.get(seq).embedding).float()
            return memo[ex.self_sequence].to(device), memo[ex.partner_sequence].to(device)

        def free():
            scorer.model.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return BACKBONE_HIDDEN[backbone], get, free

    if backbone == "ablang2":
        scorer = AbLang2Scorer()
        cache = SequenceScoreCache.__new__(SequenceScoreCache)  # reuse disk-cache path logic only
        from pathlib import Path
        from simba.features.cache import CACHE_DIR, _sanitize
        cache.scorer = scorer
        cache.dir = CACHE_DIR / _sanitize(scorer.model_name)
        cache.dir.mkdir(parents=True, exist_ok=True)

        def get(ex, device, memo):
            for seq in (ex.self_sequence, ex.partner_sequence):
                if seq not in memo:
                    memo[seq] = torch.from_numpy(cache.get(seq).embedding).float()
            return memo[ex.self_sequence].to(device), memo[ex.partner_sequence].to(device)

        def free():
            pass  # AbLang2 already lives on CPU, never touched the GPU

        return BACKBONE_HIDDEN[backbone], get, free

    if backbone == "mint":
        scorer = MINTPairScorer(device=device)
        cache = MintPairCache(scorer)

        def get(ex, device, memo):
            key = (ex.self_sequence, ex.partner_sequence)
            if key not in memo:
                se, pe = cache.get(ex.self_sequence, ex.partner_sequence)
                memo[key] = (torch.from_numpy(se).float(), torch.from_numpy(pe).float())
            se, pe = memo[key]
            return se.to(device), pe.to(device)

        def free():
            scorer.wrapper.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        return BACKBONE_HIDDEN[backbone], get, free

    raise ValueError(backbone)


def prewarm_cache(examples, get_embeddings, device):
    """Populate the on-disk embedding cache for every example in the pool
    *once*, up front, before any fold's training loop starts. Folds
    overlap heavily (each example sits in 4 folds' train/val sets and 1
    fold's test set), so without this the first fold to touch a given
    example still pays the real model-inference cost and every later fold
    just gets a cheap disk-cache hit -- correct, but the "embedding" and
    "training" phases end up interleaved and hard to separate when timing
    or debugging a run. Pre-warming here makes the split explicit: this
    step is the only one that ever calls the actual PLM/MINT model."""
    from tqdm import tqdm
    memo: dict = {}
    for ex in tqdm(examples, desc="pre-warming embedding cache"):
        get_embeddings(ex, device, memo)


def evaluate(model, examples, get_embeddings, device, memo):
    model.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for ex in examples:
            self_emb, partner_emb = get_embeddings(ex, device, memo)
            preds.append(model(self_emb, partner_emb).item())
            trues.append(ex.dG)
            ids.append(ex.complex_id)
    return np.asarray(preds), np.asarray(trues), ids


def run_fold(train, val, test, *, hidden, get_embeddings, epochs, batch_size, lr, weight_decay,
             device, seed, patience=8):
    """Same protocol as `eval_skempi2_5fold.py --pool combined --split
    cold_target` (epochs default matches that script's 40, not shortened
    here), plus real early stopping on top of the existing best-val
    -checkpoint selection: that selection alone doesn't actually save any
    time (still runs the full fixed epoch budget even once validation has
    plateaued) -- stopping once `patience` epochs pass with no new best val
    PCC cuts wall-clock time for backbones that converge sooner, without
    changing the epoch *ceiling* or the selected checkpoint for any
    backbone (identical result to the no-early-stopping version whenever
    training doesn't actually plateau early)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    memo: dict = {}
    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_pcc, best_state, global_step, epochs_since_best = -math.inf, None, 0, 0

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
                self_emb, partner_emb = get_embeddings(ex, device, memo)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        val_preds, val_trues, _ = evaluate(model, val, get_embeddings, device, memo)
        val_pcc = float(pearsonr(val_trues, val_preds)[0]) if np.std(val_preds) > 0 else float("nan")
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        print(f"    epoch {epoch}: val_pcc={val_pcc:.4f} best={best_pcc:.4f} "
              f"(no improvement for {epochs_since_best} epochs)", flush=True)
        if epochs_since_best >= patience:
            print(f"    early stop at epoch {epoch} (patience={patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    preds, trues, ids = evaluate(model, test, get_embeddings, device, memo)
    return preds, trues, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["esm2", "ablang2", "mint"], required=True)
    parser.add_argument("--epochs", type=int, default=40,
                         help="Ceiling, matching eval_skempi2_5fold.py's default -- early stopping "
                              "(--patience) usually cuts this short in practice.")
    parser.add_argument("--patience", type=int, default=8,
                         help="Stop a fold once this many epochs pass with no new best val PCC.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-folds", type=int, default=None, help="For quick timing tests only.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = args.out or f"plm_ablation_cold_{args.backbone}_results.json"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = build_combined_examples()
    folds = make_folds(examples, split_type="cold_target")
    print(f"[plm_ablation_cold] backbone={args.backbone} {len(examples)} examples, "
          f"{len(set(e.complex_id.upper() for e in examples))} complexes, "
          f"fold sizes={[len(f) for f in folds]}", flush=True)

    hidden, get_embeddings, free_backbone_gpu_memory = build_backend(args.backbone, device)

    print(f"[plm_ablation_cold] pre-warming {args.backbone} embedding cache for all "
          f"{len(examples)} examples once, up front...", flush=True)
    prewarm_cache(examples, get_embeddings, device)
    free_backbone_gpu_memory()
    print("[plm_ablation_cold] cache pre-warm done, backbone model freed from GPU -- every fold "
          "below is a disk-cache hit, no further model inference.", flush=True)

    all_preds, all_trues, all_ids, all_folds = [], [], [], []
    fold_pccs, fold_sccs, fold_maes, fold_rmses = [], [], [], []
    rng = random.Random(args.seed)
    n_folds_to_run = args.max_folds or N_FOLDS

    for fold_idx, test_ids in enumerate(folds[:n_folds_to_run]):
        t0 = time.time()
        test = [e for e in examples if e.complex_id in test_ids]
        train_val = [e for e in examples if e.complex_id not in test_ids]
        idx = list(range(len(train_val)))
        rng.shuffle(idx)
        n_val = max(1, int(args.val_frac * len(idx)))
        val = [train_val[i] for i in idx[:n_val]]
        train = [train_val[i] for i in idx[n_val:]]
        print(f"\n=== fold {fold_idx}: train={len(train)} val={len(val)} test={len(test)} ===", flush=True)

        preds, trues, ids = run_fold(train, val, test, hidden=hidden, get_embeddings=get_embeddings,
                                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                      weight_decay=args.weight_decay, device=device,
                                      seed=args.seed * 100 + fold_idx, patience=args.patience)
        # Explicit cleanup between folds: run_fold's model/optimizer/best_state
        # (a GPU-resident deepcopy of the state dict, replaced repeatedly
        # over up to `epochs` checkpoints) should be garbage-collected once
        # the function returns, but relying on refcounting alone left ~11GB
        # "still allocated by PyTorch" at the start of the next fold in
        # practice (confirmed: fold 0 succeeded, fold 1 immediately OOM'd on
        # the same GPU that showed 0MiB used before this run started) --
        # force it explicitly rather than debug the exact reference cycle.
        import gc
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        scc = float(spearmanr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        mae = float(np.mean(np.abs(trues - preds)))
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
        fold_pccs.append(pcc); fold_sccs.append(scc); fold_maes.append(mae); fold_rmses.append(rmse)
        print(f"[fold {fold_idx}] PCC={pcc:.4f} SCC={scc:.4f} MAE={mae:.4f} RMSE={rmse:.4f} "
              f"(n={len(test)}) [{time.time() - t0:.1f}s]", flush=True)
        all_preds.extend(preds.tolist())
        all_trues.extend(trues.tolist())
        all_ids.extend(ids)
        all_folds.extend([fold_idx] * len(ids))

    all_preds, all_trues = np.array(all_preds), np.array(all_trues)
    pooled_pcc = float(pearsonr(all_trues, all_preds)[0])
    pooled_scc = float(spearmanr(all_trues, all_preds)[0])
    pooled_mae = float(np.mean(np.abs(all_trues - all_preds)))
    pooled_rmse = float(np.sqrt(np.mean((all_trues - all_preds) ** 2)))

    result = {
        "backbone": args.backbone, "pool": "combined", "split": "cold_target",
        "epochs": args.epochs, "n_folds_run": n_folds_to_run,
        "fold_pcc_mean": float(np.mean(fold_pccs)), "fold_pcc_std": float(np.std(fold_pccs)),
        "fold_scc_mean": float(np.mean(fold_sccs)), "fold_scc_std": float(np.std(fold_sccs)),
        "fold_mae_mean": float(np.mean(fold_maes)), "fold_mae_std": float(np.std(fold_maes)),
        "fold_rmse_mean": float(np.mean(fold_rmses)), "fold_rmse_std": float(np.std(fold_rmses)),
        "fold_pccs": fold_pccs, "fold_sccs": fold_sccs, "fold_maes": fold_maes, "fold_rmses": fold_rmses,
        "pooled_pcc": pooled_pcc, "pooled_scc": pooled_scc,
        "pooled_mae": pooled_mae, "pooled_rmse": pooled_rmse,
        "predictions": {"ids": all_ids, "true": all_trues.tolist(), "pred": all_preds.tolist(),
                         "fold": all_folds},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[plm_ablation_cold] backbone={args.backbone} "
          f"fold-avg PCC={result['fold_pcc_mean']:.4f}+/-{result['fold_pcc_std']:.4f} "
          f"SCC={result['fold_scc_mean']:.4f}+/-{result['fold_scc_std']:.4f} "
          f"MAE={result['fold_mae_mean']:.4f}+/-{result['fold_mae_std']:.4f} "
          f"RMSE={result['fold_rmse_mean']:.4f}+/-{result['fold_rmse_std']:.4f}  "
          f"pooled PCC={pooled_pcc:.4f} SCC={pooled_scc:.4f} MAE={pooled_mae:.4f} RMSE={pooled_rmse:.4f}",
          flush=True)
    print(f"[plm_ablation_cold] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
