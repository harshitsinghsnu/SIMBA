"""Extended absolute-affinity evaluation: (1) multi-seed reruns of the
proposed architecture (mean +/- std, not single-run numbers), and (2) a
backbone x projection-size sweep for the proposed architecture. Reuses
`embed`/`evaluate` from simba/eval_affinity.py so results are directly
comparable to every number already reported there.

Deliberately excludes the cosine-similarity head and any self-attention
-based chain-fusion variant -- both already established as non-winning
ablations (simba/eval_affinity.py, MODEL_REPORT.md S3.5/S3.7); this script
only ever runs the proposed configuration (pairformer, MLP head, no epinet)
with an explicit `torch.manual_seed` per run (added here; the original
single-seed scripts never controlled torch's global RNG for model
init/dropout, only the data-shuffling order) so multi-seed numbers are
genuinely reproducible per seed, not just re-shuffled data on the same
weight initialization.
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

from simba.data.build_affinity_benchmark import (  # noqa: E402
    build_prott_affinity_test1_reproduction,
    build_skempi2_wt_train_set,
    build_test_set,
)
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


def embed(cache: SequenceScoreCache, sequence: str, device: str, memo: dict) -> torch.Tensor:
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float().to(device)
    return memo[sequence]


def evaluate(model, examples, cache, device, memo, return_raw=False):
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
    result = {"pcc": pcc, "mae": mae, "n": len(trues)}
    if return_raw:
        result["raw"] = {"pred": preds.tolist(), "true": trues.tolist()}
    return result


def run_one(train_full, test, *, hidden, d_model, proj_mid_dim, epochs, batch_size, lr,
            weight_decay, val_frac, device, seed, cache, return_raw=False):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]

    memo: dict = {}
    model = AbsoluteAffinityModel(plm_hidden_size=hidden, d_model=d_model, proj_mid_dim=proj_mid_dim).to(device)
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
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        val_report = evaluate(model, val, cache, device, memo)
        if not math.isnan(val_report["pcc"]) and val_report["pcc"] > best_pcc:
            best_pcc = val_report["pcc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return evaluate(model, test, cache, device, memo, return_raw=return_raw)


def multi_seed(train_full, test, *, hidden, d_model, proj_mid_dim, epochs, batch_size, lr,
               weight_decay, val_frac, device, cache, seeds, label):
    pccs, maes = [], []
    scatter_seed0 = None
    test_ids = [e.complex_id for e in test]
    predictions_by_seed = []
    for seed in seeds:
        r = run_one(train_full, test, hidden=hidden, d_model=d_model, proj_mid_dim=proj_mid_dim,
                    epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                    val_frac=val_frac, device=device, seed=seed, cache=cache, return_raw=True)
        pccs.append(r["pcc"])
        maes.append(r["mae"])
        if seed == seeds[0]:
            scatter_seed0 = r["raw"]
        predictions_by_seed.append({"seed": seed, "ids": test_ids, "true": r["raw"]["true"],
                                     "pred": r["raw"]["pred"]})
        print(f"  [{label}] seed={seed}: PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)
    result = {
        "pcc_mean": float(np.mean(pccs)), "pcc_std": float(np.std(pccs)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "pcc_all": pccs, "mae_all": maes, "n_seeds": len(seeds),
        "scatter_seed0": scatter_seed0,
        "predictions_by_seed": predictions_by_seed,
    }
    print(f"[{label}] PCC = {result['pcc_mean']:.4f} +/- {result['pcc_std']:.4f}  "
          f"MAE = {result['mae_mean']:.4f} +/- {result['mae_std']:.4f}  (n_seeds={len(seeds)})", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["multiseed", "plm_sweep"], required=True)
    parser.add_argument("--test-sets", nargs="+", default=["prott_test1", "benchmark46"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="affinity_extended_results.json")
    # plm_sweep-specific
    parser.add_argument("--backbones", nargs="+",
                         default=["facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"])
    parser.add_argument("--d-models", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    def get_test(name):
        return build_test_set() if name == "benchmark46" else build_prott_affinity_test1_reproduction()

    if args.mode == "multiseed":
        scorer = ESMScorer(model_name="facebook/esm2_t33_650M_UR50D", device=device)
        hidden = scorer.model.config.hidden_size
        cache = SequenceScoreCache(scorer)
        for test_name in args.test_sets:
            test = get_test(test_name)
            test_ids = {e.complex_id for e in test}
            train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
            print(f"\n=== multiseed: {test_name} (train={len(train_full)}, test={len(test)}) ===", flush=True)
            r = multi_seed(train_full, test, hidden=hidden, d_model=256, proj_mid_dim=512,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, val_frac=args.val_frac, device=device,
                           cache=cache, seeds=args.seeds, label=test_name)
            results[test_name] = r
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    else:  # plm_sweep
        for backbone in args.backbones:
            scorer = ESMScorer(model_name=backbone, device=device)
            hidden = scorer.model.config.hidden_size
            cache = SequenceScoreCache(scorer)
            for test_name in args.test_sets:
                test = get_test(test_name)
                test_ids = {e.complex_id for e in test}
                train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
                for d_model in args.d_models:
                    proj_mid = d_model * 2 if d_model * 2 < hidden else None
                    label = f"{backbone.split('/')[-1]}_d{d_model}_{test_name}"
                    print(f"\n=== plm_sweep: {label} (train={len(train_full)}, test={len(test)}) ===", flush=True)
                    r = run_one(train_full, test, hidden=hidden, d_model=d_model, proj_mid_dim=proj_mid,
                                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                weight_decay=args.weight_decay, val_frac=args.val_frac, device=device,
                                seed=0, cache=cache)
                    print(f"[{label}] PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)
                    results[label] = r
                    with open(args.out, "w") as f:
                        json.dump(results, f, indent=2)

    print("\n\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
