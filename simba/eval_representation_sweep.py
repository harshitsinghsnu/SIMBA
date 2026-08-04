"""Representation sweep: ESM2-150M, ESM2-650M, ProtT5, Ankh -- same
downstream head (AbsoluteAffinityModel, pairformer, unchanged) on
benchmark46. Directly tests "is representation the bottleneck" without
inventing another interaction-processing module (per the four-primitives
-converge finding, main report §3.6): swap only the frozen backbone.

ProtT5 and Ankh are both T5-encoder architectures (different tokenizer/
input convention than ESM's BERT-style tokenizer -- space-separated amino
acids, rare residues mapped to 'X'), so they're embedded via a separate
T5EncoderModel path rather than the existing ESM-specific SequenceScoreCache.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re

import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer, T5EncoderModel

from simba import env

env.setup()

from simba.data.build_affinity_benchmark import build_skempi2_wt_train_set, build_test_set  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402

T5_MODELS = {"Rostlab/prot_t5_xl_uniref50", "ElnaggarLab/ankh-base"}


class T5EmbeddingScorer:
    """Minimal T5-encoder embedding extractor, API-compatible enough with
    this project's `embed(cache, sequence, device, memo)` pattern (used via
    a closure below) -- ProtT5/Ankh convention: uppercase, rare residues
    (U/Z/O/B) -> 'X', space-separated single-letter codes as input."""

    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, sequence: str) -> np.ndarray:
        cleaned = re.sub(r"[UZOB]", "X", sequence.upper())
        spaced = " ".join(list(cleaned))
        toks = self.tokenizer(spaced, return_tensors="pt").to(self.device)
        out = self.model(**toks).last_hidden_state[0]
        return out[:-1].cpu().numpy()  # strip T5's trailing </s> token


def make_embed_fn(backbone_name: str, device: str):
    """Returns (embed_fn(sequence, memo) -> Tensor[L,hidden], hidden_size)."""
    if backbone_name in T5_MODELS:
        scorer = T5EmbeddingScorer(backbone_name, device)
        hidden = scorer.model.config.d_model

        def embed(sequence, memo):
            if sequence not in memo:
                memo[sequence] = torch.from_numpy(scorer.embed(sequence)).float()
            return memo[sequence].to(device)
        return embed, hidden
    else:
        esm_scorer = ESMScorer(model_name=backbone_name, device=device)
        cache = SequenceScoreCache(esm_scorer)
        hidden = esm_scorer.model.config.hidden_size

        def embed(sequence, memo):
            if sequence not in memo:
                memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
            return memo[sequence].to(device)
        return embed, hidden


def run_one(train_full, test, *, embed_fn, hidden, epochs, batch_size, lr, weight_decay,
            val_frac, device, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]

    memo: dict = {}
    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_pcc, best_state, global_step = -math.inf, None, 0

    def evaluate(examples):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ex in examples:
                self_emb = embed_fn(ex.self_sequence, memo)
                partner_emb = embed_fn(ex.partner_sequence, memo)
                preds.append(model(self_emb, partner_emb).item())
                trues.append(ex.dG)
        preds, trues = np.asarray(preds), np.asarray(trues)
        pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
        mae = float(np.mean(np.abs(trues - preds)))
        return {"pcc": pcc, "mae": mae, "n": len(trues)}

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
                self_emb = embed_fn(ex.self_sequence, memo)
                partner_emb = embed_fn(ex.partner_sequence, memo)
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
    parser.add_argument("--backbones", nargs="+", default=[
        "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D",
        "Rostlab/prot_t5_xl_uniref50", "ElnaggarLab/ankh-base",
    ])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="representation_sweep_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    test = build_test_set()
    test_ids = {e.complex_id for e in test}
    train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
    print(f"train={len(train_full)} test={len(test)}", flush=True)

    results = {}
    for backbone in args.backbones:
        print(f"\n=== {backbone} ===", flush=True)
        embed_fn, hidden = make_embed_fn(backbone, device)
        r = run_one(train_full, test, embed_fn=embed_fn, hidden=hidden, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                    val_frac=args.val_frac, device=device, seed=args.seed)
        results[backbone] = r
        print(f"[{backbone}] PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
