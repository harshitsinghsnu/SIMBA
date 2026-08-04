"""Evaluates SIMBA on PPI-Affinity's "Test Set 2" (Romero-Molina et al.
2022) — the same 90-complex set PPI-Graphomer's Table 1 calls "Test set 2"
(87 rows after their own filtering; this project resolves 89/90 from
public RCSB structures — see simba/data/build_ppi_affinity_test2.py for
the full provenance chain). Unlike the Test Set 1 reconstruction
(eval_ppi_graphomer_test1.py), this table's `Binding_affinity` uses the
standard sign convention (negative=favorable) matching SIMBA's own dG
directly — no sign-flip needed to compare magnitudes here.

Real per-complex predictions from PRODIGY, DFIRE, CP_PIE, ISLAND, and
PPI-Affinity's own SVM are carried through from the source table for a
genuinely row-matched comparison against the same methods PPI-Graphomer's
Table 1 compares against.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import pearsonr

from simba import env

env.setup()

from simba.data.build_affinity_benchmark import build_skempi2_wt_train_set  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


@dataclass
class Ex:
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def load_test2() -> list[Ex]:
    with open("data/processed/ppi_affinity_test2.json") as f:
        raw = json.load(f)
    return [Ex(**r) for r in raw]


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def run_one(train, test, *, hidden, epochs, batch_size, lr, weight_decay, val_frac, device, seed, cache):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val = [train[i] for i in idx[:n_val]]
    train_split = [train[i] for i in idx[n_val:]]

    memo: dict = {}
    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_split) / batch_size))
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
        return {"pcc": pcc, "mae": mae, "n": len(trues), "preds": preds.tolist(), "trues": trues.tolist()}

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_split)
        for start in range(0, len(train_split), batch_size):
            batch = train_split[start:start + batch_size]
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

        val_report = evaluate(val)
        if not math.isnan(val_report["pcc"]) and val_report["pcc"] > best_pcc:
            best_pcc = val_report["pcc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="ppi_affinity_test2_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    test = load_test2()
    test_ids = {e.complex_id for e in test}
    train = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
    print(f"[ppi_affinity_test2] train={len(train)} test={len(test)}", flush=True)

    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    true = np.array([e.dG for e in test])
    baseline_names = sorted({k for e in test for k in e.meta["baselines"]})
    baseline_pcc = {}
    for name in baseline_names:
        vals, matched_true = [], []
        for e in test:
            if name in e.meta["baselines"]:
                vals.append(e.meta["baselines"][name])
                matched_true.append(e.dG)
        vals, matched_true = np.array(vals), np.array(matched_true)
        if len(vals) >= 5 and np.std(vals) > 0:
            baseline_pcc[name] = {"pcc": float(pearsonr(matched_true, vals)[0]), "n": len(vals)}
    print("\n[real per-complex baseline PCCs, same rows]", flush=True)
    for name, r in sorted(baseline_pcc.items(), key=lambda kv: -abs(kv[1]["pcc"])):
        print(f"  {name}: PCC={r['pcc']:.4f} (n={r['n']})", flush=True)

    test_ids_ordered = [e.complex_id for e in test]
    pccs, maes, predictions_by_seed = [], [], []
    for seed in args.seeds:
        r = run_one(train, test, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, weight_decay=args.weight_decay, val_frac=args.val_frac,
                    device=device, seed=seed, cache=cache)
        pccs.append(r["pcc"]); maes.append(r["mae"])
        predictions_by_seed.append({"seed": seed, "ids": test_ids_ordered, "true": r["trues"], "pred": r["preds"]})
        print(f"[ppi_affinity_test2] seed={seed}: PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)

    result = {
        "n_test": len(test),
        "pcc_mean": float(np.mean(pccs)), "pcc_std": float(np.std(pccs)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "pcc_all": pccs, "mae_all": maes, "n_seeds": len(args.seeds),
        "real_baseline_pccs_same_rows": baseline_pcc,
        "predictions_by_seed": predictions_by_seed,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[ppi_affinity_test2] PCC={result['pcc_mean']:.4f}+/-{result['pcc_std']:.4f} "
          f"MAE={result['mae_mean']:.4f}+/-{result['mae_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
