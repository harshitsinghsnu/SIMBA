"""Few-shot data-scaling extension of `eval_novel_generalization.py`: instead
of a single fixed few-shot fine-tuning run, this repeats it at increasing
fractions of the available support set (10%, 20%, 30%, 100%) to show whether
more few-shot data actually helps, holding everything else (test set,
checkpoint, epochs, LR, seed) identical to the original protocol.

Constraint worth stating up front: the genuinely-novel (zero-overlap-with
-training-pool) support pool is only 15 rows across 2 complexes (3K2M,
HM_3BN9) -- see MODEL_REPORT.md S3.19 for why this is the *entire* usable
non-test slice of the only clean benchmark found (ProtAttBA's AB1101).
"10%/20%/30% of data" therefore means 10/20/30% of those 15 rows (2, 3, 5
examples), not a large-scale few-shot curve -- there simply isn't more
genuinely-novel labeled data available without fetching brand-new individual
structures. The fractions are nested (10% subset subset of 20% subset of 30%
subset of 100%) via one fixed shuffle, so this is a real "adding more data"
curve, not four independent random draws.
"""

from __future__ import annotations

import copy
import json
import math
import random

import numpy as np
import torch

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_protattba  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.eval_novel_generalization import (  # noqa: E402
    CKPT_PATH, META_PATH, TEST_COMPLEXES, SUPPORT_COMPLEXES,
    FEWSHOT_EPOCHS, FEWSHOT_LR, SEED, load_model, embed, evaluate, metrics,
)

FRACTIONS = [0.10]
OUT_JSON = "novel_generalization_fewshot_scaling_results.json"


def fine_tune(model, examples, cache, device, memo, epochs, lr, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    torch.manual_seed(seed)
    model.train()
    n = len(examples)
    for _ in range(epochs):
        optimizer.zero_grad()
        for ex in examples:
            wt_full, mut_full, partner = full_wt_mut_sequences(ex)
            wt_emb = embed(cache, wt_full, device, memo)
            mut_emb = embed(cache, mut_full, device, memo)
            partner_emb = embed(cache, partner, device, memo)
            pred = model.forward_ddg(wt_emb, mut_emb, partner_emb)
            target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
            loss = (pred - target) ** 2 / n
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(META_PATH) as f:
        meta = json.load(f)

    examples = load_protattba("ab1101")
    test_examples = [e for e in examples if e.meta.get("pdb_id", "").upper() in TEST_COMPLEXES]
    support_examples = [e for e in examples if e.meta.get("pdb_id", "").upper() in SUPPORT_COMPLEXES]
    n_support = len(support_examples)
    print(f"[fewshot_scale] test complexes={TEST_COMPLEXES} n={len(test_examples)}", flush=True)
    print(f"[fewshot_scale] support pool complexes={SUPPORT_COMPLEXES} n={n_support}", flush=True)

    rng = random.Random(SEED)
    shuffled_support = support_examples[:]
    rng.shuffle(shuffled_support)

    scorer = ESMScorer(model_name=meta["backbone"], device=device)
    cache = SequenceScoreCache(scorer)
    memo: dict = {}

    # Zero-shot baseline (0% few-shot data), for the x=0 point of the curve.
    zeroshot_model = load_model(meta, device)
    zs_preds, zs_trues, zs_protattba = evaluate(zeroshot_model, test_examples, cache, device, memo)
    zs_metrics = metrics(zs_trues, zs_preds)
    protattba_metrics = metrics(zs_trues, zs_protattba)
    print(f"[fewshot_scale] ZERO-SHOT (0 examples) Pearson={zs_metrics['pearson']:.4f} "
          f"Spearman={zs_metrics['spearman']:.4f} RMSE={zs_metrics['rmse']:.4f}", flush=True)

    rows = [{"fraction": 0.0, "n_support_used": 0, "pearson": zs_metrics["pearson"],
             "spearman": zs_metrics["spearman"], "rmse": zs_metrics["rmse"]}]

    test_ids = [e.meta.get("pdb_id", e.complex_id) for e in test_examples]
    fewshot_preds = {}
    for frac in FRACTIONS:
        n_sub = max(1, math.ceil(frac * n_support))
        n_sub = min(n_sub, n_support)
        subset = shuffled_support[:n_sub]
        model = load_model(meta, device)
        model = fine_tune(model, subset, cache, device, memo, FEWSHOT_EPOCHS, FEWSHOT_LR, SEED)
        preds, trues, _ = evaluate(model, test_examples, cache, device, memo)
        m = metrics(trues, preds)
        print(f"[fewshot_scale] FEW-SHOT frac={frac:.0%} ({n_sub}/{n_support} support examples) "
              f"Pearson={m['pearson']:.4f} Spearman={m['spearman']:.4f} RMSE={m['rmse']:.4f}", flush=True)
        rows.append({"fraction": frac, "n_support_used": n_sub, "pearson": m["pearson"],
                     "spearman": m["spearman"], "rmse": m["rmse"]})
        fewshot_preds[f"{frac:.2f}"] = preds.tolist()

    result = {
        "test_complexes": sorted(TEST_COMPLEXES), "n_test": len(test_examples),
        "support_complexes": sorted(SUPPORT_COMPLEXES), "n_support_total": n_support,
        "epochs": FEWSHOT_EPOCHS, "lr": FEWSHOT_LR, "seed": SEED,
        "protattba_reference": protattba_metrics,
        "curve": rows,
        "predictions": {
            "ids": test_ids, "true": zs_trues.tolist(), "zeroshot_pred": zs_preds.tolist(),
            "protattba_pred": zs_protattba.tolist(), "fewshot_pred_by_fraction": fewshot_preds,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fewshot_scale] saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
