"""Zero-shot and few-shot generalization test on genuinely novel complexes
-- PDB structures that never appeared anywhere in this project's absolute
-affinity training pool (SKEMPI2 expanded + benchmark46 + ATLAS, 541
complexes). Almost every standard curated PPI-affinity benchmark (AB-Bind,
GeoPPI's M1707, ProtAttBA's own S1131/AB645/AB1101) turns out to share
110+ of its ~112-120 complexes with that pool -- these datasets all draw
from the same finite universe of experimentally-characterized, publicly
deposited PPI complexes, so "zero complex overlap" data is genuinely scarce
without fetching brand-new individual structures one at a time. Checked
directly (not assumed) for four candidate benchmarks; ProtAttBA's AB1101
(the only one with real multi-site mutations) has the most usable novel
slice: 6 complexes / 94 resolved rows never in the training pool (3K2M,
plus 5 homology-model-based AB-Bind entries HM_1KTZ/1YY9/2NYY/2NZ9/3BN9 --
sequence-only means a homology-model reference structure is exactly as
valid as a crystal structure, since no 3D coordinates are ever model input;
the ddG values are real experimental mutagenesis measurements either way).

Protocol:
  Zero-shot:  the existing checkpoint (`simba_pairformer_combined.pt`,
              trained only on absolute dG, never on this dataset or task),
              `forward_ddg` called directly with NO fine-tuning at all, on
              a 79-row test set (4 novel complexes: HM_2NYY, HM_1KTZ,
              HM_1YY9, HM_2NZ9).
  Few-shot:   the same checkpoint, fine-tuned for a handful of epochs on a
              disjoint 15-row support set (2 different novel complexes:
              3K2M, HM_3BN9 -- smallest available, never seen by either the
              zero-shot or few-shot model during pretraining), then
              evaluated on the identical 79-row test set for a direct,
              apples-to-apples zero-shot vs. few-shot comparison.

ProtAttBA's own real per-row predictions (already resolved in this
project's data) are reported on the same 79-row test set as a reference
point, though ProtAttBA WAS trained with this benchmark's own released
folds (unlike SIMBA here), so it is not a fair peer comparison -- included
for context only.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_protattba  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402

CKPT_PATH = "checkpoints/simba_pairformer_combined.pt"
META_PATH = "checkpoints/simba_pairformer_combined.json"
TEST_COMPLEXES = {"HM_2NYY", "HM_1KTZ", "HM_1YY9", "HM_2NZ9"}
SUPPORT_COMPLEXES = {"3K2M", "HM_3BN9"}
FEWSHOT_EPOCHS = 15
FEWSHOT_LR = 5e-4
SEED = 0
OUT_JSON = "novel_generalization_results.json"


def load_model(meta, device):
    model = AbsoluteAffinityModel(
        plm_hidden_size=meta["plm_hidden_size"], d_model=meta["d_model"],
        n_heads=meta["n_heads"], n_memory=meta["n_memory"], pair_hidden=meta["pair_hidden"],
        mlp_hidden=meta["mlp_hidden"], use_pairformer=meta["use_pairformer"],
        n_pairformer_layers=meta["n_pairformer_layers"],
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    return model


def embed(cache, sequence, device, memo):
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def evaluate(model, examples, cache, device, memo):
    model.eval()
    preds, trues, protattba_preds = [], [], []
    with torch.no_grad():
        for ex in examples:
            wt_full, mut_full, partner = full_wt_mut_sequences(ex)
            wt_emb = embed(cache, wt_full, device, memo)
            mut_emb = embed(cache, mut_full, device, memo)
            partner_emb = embed(cache, partner, device, memo)
            pred = model.forward_ddg(wt_emb, mut_emb, partner_emb).item()
            preds.append(pred)
            trues.append(ex.ddG)
            protattba_preds.append(ex.meta["protattba_prediction"])
    return np.array(preds), np.array(trues), np.array(protattba_preds)


def metrics(trues, preds):
    rmse = float(np.sqrt(np.mean((np.asarray(trues) - np.asarray(preds)) ** 2)))
    if np.std(preds) == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": rmse}
    return {"pearson": float(pearsonr(trues, preds)[0]), "spearman": float(spearmanr(trues, preds)[0]),
            "rmse": rmse}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(META_PATH) as f:
        meta = json.load(f)

    examples = load_protattba("ab1101")
    test_examples = [e for e in examples if e.meta.get("pdb_id", "").upper() in TEST_COMPLEXES]
    support_examples = [e for e in examples if e.meta.get("pdb_id", "").upper() in SUPPORT_COMPLEXES]
    print(f"[novel_gen] test complexes={TEST_COMPLEXES} n={len(test_examples)}", flush=True)
    print(f"[novel_gen] support (few-shot) complexes={SUPPORT_COMPLEXES} n={len(support_examples)}", flush=True)

    scorer = ESMScorer(model_name=meta["backbone"], device=device)
    cache = SequenceScoreCache(scorer)
    memo: dict = {}

    # --- Zero-shot ---
    zeroshot_model = load_model(meta, device)
    zs_preds, zs_trues, zs_protattba = evaluate(zeroshot_model, test_examples, cache, device, memo)
    zs_metrics = metrics(zs_trues, zs_preds)
    protattba_metrics = metrics(zs_trues, zs_protattba)
    print(f"[novel_gen] ZERO-SHOT  Pearson={zs_metrics['pearson']:.4f} Spearman={zs_metrics['spearman']:.4f}", flush=True)
    print(f"[novel_gen] ProtAttBA (own released weights, trained WITH this benchmark's folds; context only) "
          f"Pearson={protattba_metrics['pearson']:.4f} Spearman={protattba_metrics['spearman']:.4f}", flush=True)

    # --- Few-shot: fine-tune a fresh copy of the checkpoint on the support set ---
    fewshot_model = load_model(meta, device)
    optimizer = torch.optim.AdamW(fewshot_model.parameters(), lr=FEWSHOT_LR, weight_decay=1e-4)
    torch.manual_seed(SEED)
    fewshot_model.train()
    for epoch in range(FEWSHOT_EPOCHS):
        optimizer.zero_grad()
        n = len(support_examples)
        for ex in support_examples:
            wt_full, mut_full, partner = full_wt_mut_sequences(ex)
            wt_emb = embed(cache, wt_full, device, memo)
            mut_emb = embed(cache, mut_full, device, memo)
            partner_emb = embed(cache, partner, device, memo)
            pred = fewshot_model.forward_ddg(wt_emb, mut_emb, partner_emb)
            target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
            loss = (pred - target) ** 2 / n
            loss.backward()
        torch.nn.utils.clip_grad_norm_(fewshot_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    fs_preds, fs_trues, _ = evaluate(fewshot_model, test_examples, cache, device, memo)
    fs_metrics = metrics(fs_trues, fs_preds)
    print(f"[novel_gen] FEW-SHOT ({len(support_examples)} support examples, {FEWSHOT_EPOCHS} epochs) "
          f"Pearson={fs_metrics['pearson']:.4f} Spearman={fs_metrics['spearman']:.4f}", flush=True)

    result = {
        "test_complexes": sorted(TEST_COMPLEXES), "n_test": len(test_examples),
        "support_complexes": sorted(SUPPORT_COMPLEXES), "n_support": len(support_examples),
        "zero_shot": zs_metrics, "few_shot": fs_metrics, "protattba_reference": protattba_metrics,
        "zero_shot_preds": zs_preds.tolist(), "few_shot_preds": fs_preds.tolist(),
        "true": zs_trues.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[novel_gen] saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
