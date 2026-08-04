"""Trains and saves one pairformer checkpoint on the combined absolute
-affinity pool (§3.15/build_combined_affinity.py), for the interpretability
case study (Integrated Gradients + PyMOL structure mapping): the 5-fold CV
scripts elsewhere in this project never persist weights to disk (each
fold's model is discarded after scoring), so a dedicated single training
run is needed to get a checkpoint at all.

1BRS (barnase-barstar, a classic, extensively alanine-scanned hotspot
system -- Schreiber & Fersht) is held out of training entirely, not just
excluded from the loss, so the interpretability case study reflects
genuine generalization to an unseen complex rather than memorization of
its exact training signal.
"""

from __future__ import annotations

import copy
import json
import math
import random

import numpy as np
import torch
from scipy.stats import pearsonr

from simba import env

env.setup()

from simba.data.build_combined_affinity import build_combined_examples  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

HOLDOUT_COMPLEX = "1BRS"
SEED = 0
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.10
CKPT_PATH = "checkpoints/simba_pairformer_combined.pt"
META_PATH = "checkpoints/simba_pairformer_combined.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = [e for e in build_combined_examples() if e.complex_id.upper() != HOLDOUT_COMPLEX]
    print(f"[train_final_checkpoint] {len(examples)} examples "
          f"(holding out {HOLDOUT_COMPLEX} entirely)", flush=True)

    backbone = "facebook/esm2_t33_650M_UR50D"
    scorer = ESMScorer(model_name=backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)
    memo: dict = {}

    def embed(sequence):
        if sequence not in memo:
            memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
        return memo[sequence].to(device)

    rng = random.Random(SEED)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(VAL_FRAC * len(idx)))
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]
    print(f"[train_final_checkpoint] train={len(train)} val={len(val)}", flush=True)

    torch.manual_seed(SEED)
    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(train) / BATCH_SIZE))
    total_steps = steps_per_epoch * EPOCHS
    best_pcc, best_state, global_step = -math.inf, None, 0

    for epoch in range(EPOCHS):
        model.train()
        rng.shuffle(train)
        for start in range(0, len(train), BATCH_SIZE):
            batch = train[start:start + BATCH_SIZE]
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, LR)
            optimizer.zero_grad()
            n_valid = len(batch)
            for ex in batch:
                self_emb = embed(ex.self_sequence)
                partner_emb = embed(ex.partner_sequence)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
        if device == "cuda":
            torch.cuda.empty_cache()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for ex in val:
                self_emb = embed(ex.self_sequence)
                partner_emb = embed(ex.partner_sequence)
                val_preds.append(model(self_emb, partner_emb).item())
                val_trues.append(ex.dG)
        val_preds, val_trues = np.asarray(val_preds), np.asarray(val_trues)
        val_pcc = float(pearsonr(val_trues, val_preds)[0]) if np.std(val_preds) > 0 else float("nan")
        print(f"epoch {epoch}: val_pcc={val_pcc:.4f}", flush=True)
        if not math.isnan(val_pcc) and val_pcc > best_pcc:
            best_pcc = val_pcc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), CKPT_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "backbone": backbone, "plm_hidden_size": hidden, "d_model": 256,
            "n_heads": 4, "n_memory": 24, "pair_hidden": 64, "mlp_hidden": 128,
            "use_pairformer": True, "n_pairformer_layers": 1,
            "holdout_complex": HOLDOUT_COMPLEX, "best_val_pcc": best_pcc,
            "n_train": len(train), "n_val": len(val),
        }, f, indent=2)
    print(f"[train_final_checkpoint] saved to {CKPT_PATH}, best_val_pcc={best_pcc:.4f}", flush=True)


if __name__ == "__main__":
    main()
