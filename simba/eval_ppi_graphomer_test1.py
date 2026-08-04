"""Fuller, fairer reproduction of PPI-Graphomer's (Xie et al., BMC
Bioinformatics 2025) "Test Set 1" than the older prott_test1 comparison in
eval_affinity.py: 66/79 of their exact PDB-ID test set (vs. 50/79 before),
using real per-complex labels *and* 15 other methods' real per-complex
predictions from the eLife supplementary table PPI-Graphomer's own paper
references (see build_ppi_graphomer_test1_reproduction's docstring for the
full provenance chain). This lets us report a genuinely row-matched
comparison against DFIRE, PMF, ZRANK, ZRANK2, ROSETTADOCK, PYDOCK_TOT,
SIPPER, AP_PISA, FIREDOCK (x3), CP_PIE, Kastritis2014, Horton-Lewis, and
Moal2011 -- not just PPI-Graphomer's own reported aggregate PCC on a
non-identical row set.

Sign convention note: the eLife table's `Binding_affinity` column is a
positive "binding energy" scale (higher = tighter binding per PPI-Graphomer's
own paper text, range ~4.3-18.6), the opposite convention from this project's
own dG (kcal/mol, more negative = tighter binding, from RT*ln(Kd)). Pearson
correlation is reported as computed for every method including ours -- no
sign-flipping applied -- so a strongly negative PCC for our model is exactly
the expected, correct sign given the two conventions are inverted; the
magnitude |PCC| is what's comparable to the other methods' reported figures
(some of which -- ZRANK, ROSETTADOCK, PYDOCK_TOT, FIREDOCK -- are themselves
"lower=better" docking energy scores and so *also* correlate negatively with
Binding_affinity; this is normal and already how such tables are read).
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
    build_ppi_graphomer_test1_reproduction,
    build_skempi2_wt_train_set,
)
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        score = cache.get(sequence)
        memo[sequence] = torch.from_numpy(score.embedding).float()
    return memo[sequence].to(device)


def evaluate(model, examples, cache, device, memo):
    model.eval()
    preds = []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed(cache, ex.self_sequence, device, memo)
            partner_emb = embed(cache, ex.partner_sequence, device, memo)
            preds.append(model(self_emb, partner_emb).item())
    return np.asarray(preds)


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
    val_true = np.array([e.dG for e in val])

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

        val_preds = evaluate(model, val, cache, device, memo)
        val_pcc = float(pearsonr(val_true, val_preds)[0]) if np.std(val_preds) > 0 else float("nan")
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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
    parser.add_argument("--out", default="ppi_graphomer_test1_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test = build_ppi_graphomer_test1_reproduction()
    test_ids = {e.complex_id for e in test}
    train = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
    print(f"[ppi_graphomer_test1] train={len(train)} test={len(test)} (66/79 recovered)", flush=True)

    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    true = np.array([e.dG for e in test])

    # Real per-complex baseline predictions, same 66 rows, straight from the
    # eLife table -- computed once, doesn't depend on our model/seed.
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
    print("\n[real per-complex baseline PCCs, same 66 rows]", flush=True)
    for name, r in sorted(baseline_pcc.items(), key=lambda kv: -abs(kv[1]["pcc"])):
        print(f"  {name}: PCC={r['pcc']:.4f} (n={r['n']})", flush=True)

    pccs, maes = [], []
    per_row_dump = []
    for seed in args.seeds:
        model = run_one(train, test, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, weight_decay=args.weight_decay, val_frac=args.val_frac,
                         device=device, seed=seed, cache=cache)
        memo: dict = {}
        preds = evaluate(model, test, cache, device, memo)
        for ex, pred in zip(test, preds):
            per_row_dump.append({"complex_id": ex.complex_id, "source": ex.meta["source"],
                                  "seed": seed, "pred": float(pred), "true": float(ex.dG)})

        # per-source breakdown (diagnostic: is the newly-added BM5-clean data
        # a weaker source than the Corrected Benchmark 1.0 rows?)
        for source in ("affinity_benchmark_1.0", "bm5_clean"):
            idx = [i for i, e in enumerate(test) if e.meta["source"] == source]
            if len(idx) >= 5:
                sub_true = true[idx]
                sub_pred = preds[idx]
                if np.std(sub_pred) > 0:
                    r = pearsonr(sub_true, sub_pred)[0]
                    print(f"  [seed={seed}] source={source}: PCC={r:.4f} (n={len(idx)})", flush=True)
        pcc = float(pearsonr(true, preds)[0]) if np.std(preds) > 0 else float("nan")
        mae = float(np.mean(np.abs(true - preds)))
        pccs.append(pcc)
        maes.append(mae)
        print(f"[ppi_graphomer_test1] seed={seed}: PCC={pcc:.4f} MAE={mae:.4f}", flush=True)

    result = {
        "n_test": len(test), "n_recovered_of_79": len(test),
        "pcc_mean": float(np.mean(pccs)), "pcc_std": float(np.std(pccs)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "pcc_all": pccs, "maes_all": maes, "n_seeds": len(args.seeds),
        "note": "SIMBA's dG (kcal/mol, more negative=tighter) and the eLife Binding_affinity "
                "column (positive scale, higher=tighter) are inverted conventions -- PCC sign "
                "reflects that, magnitude is what's comparable to other methods below.",
        "real_baseline_pccs_same_rows": baseline_pcc,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    with open("ppi_graphomer_test1_per_row.json", "w") as f:
        json.dump(per_row_dump, f, indent=2)
    print(f"\n[ppi_graphomer_test1] PCC={result['pcc_mean']:.4f}+/-{result['pcc_std']:.4f} "
          f"MAE={result['mae_mean']:.4f}+/-{result['mae_std']:.4f} (n_seeds={len(args.seeds)})", flush=True)


if __name__ == "__main__":
    main()
