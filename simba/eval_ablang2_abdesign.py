"""AbLang2 (antibody-specific PLM) for the antibody (heavy+light) side,
frozen ESM2-650M unchanged for the antigen (partner) side, on AbDesign's
ΔΔG subset (§3.6 of the main report: 0.371±0.027 per-complex Spearman
multi-seed with ESM2 on both sides -- our one confirmed antibody-benchmark
win). Tests whether an antibody-specific representation helps on the one
benchmark where that specificity should matter most.

Uses the official `ablang2` PyPI package (Olsen et al.'s own release, not
a third-party port) -- avoids the safetensors/torch<2.6 restriction that
blocked ProtT5/Ankh/ProtBert entirely, since it has its own `torch.load`
-based loading, independent of `transformers`.

Mixed-backbone architecture note: AbLang2 (480-dim) and ESM2-650M (1280-dim)
have different embedding dimensions, so `AbsoluteAffinityModel`'s single
shared `proj` (built for one fixed input dim) can't be reused as-is for
both sides. Instead, this script builds its own small trainable
`nn.Linear(480, d_model)` for the AbLang2 side and calls the model's
lower-level `_pooled_from_tokens(self_tokens, partner_embedding)` /
`_scalar_from_pooled` directly (already-projected self_tokens in, partner
still projected internally via the model's own `proj`, unchanged, still
1280-dim ESM2) -- reusing the exact same GatedMemory/pairformer/head
machinery as everywhere else in this project, just swapping what builds
`self_tokens`.

AbLang2 chain extraction: calling with the *other* chain passed as an empty
string reliably isolates one chain's own per-residue embeddings (verified
directly: heavy+empty-light output length is always heavy_len+3, i.e.
BOS+heavy+EOS with no separator/light contribution) -- `out[0][1:1+len(seq)]`
extracts exactly that chain's own per-residue rows, with no ambiguity about
where a paired heavy+light output would need to be split.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random

import ablang2
import numpy as np
import torch
import torch.nn as nn

from simba import env

env.setup()

from simba.data.dataset import load_all_mutations  # noqa: E402
from simba.eval_metrics import full_report  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step, split_examples  # noqa: E402

ABLANG2_HIDDEN = 480


def ablang2_embed(model, sequence: str, other_empty_placeholder: str, device: str, memo: dict, is_heavy: bool) -> torch.Tensor:
    if not sequence:
        return torch.zeros(0, ABLANG2_HIDDEN, device=device)
    key = ("H" if is_heavy else "L", sequence)
    if key not in memo:
        pair = [sequence, ""] if is_heavy else ["", sequence]
        with torch.no_grad():
            out = model([pair], mode="rescoding")[0]  # numpy [len(sequence)+3, 480]
        chain_emb = out[1:1 + len(sequence)]
        memo[key] = torch.from_numpy(chain_emb).float()
    return memo[key].to(device)


def esm_embed(cache, sequence, device, memo):
    if not sequence:
        return torch.zeros(0, cache.scorer.model.config.hidden_size, device=device)
    if sequence not in memo:
        memo[sequence] = torch.from_numpy(cache.get(sequence).embedding).float()
    return memo[sequence].to(device)


def predict_ddg(model, self_proj, ablang2_model, ex, esm_cache, device, ablang2_memo, esm_memo):
    site = ex.sites[0]
    # AbDesign rows: teammate_sequence carries the *other* antibody chain
    # unchanged; self_sequence/mut carry the mutated chain. `mutation_chain_id`
    # (which role, H or L, was mutated) isn't preserved through the SKEMPI2/
    # AB-Bind/AbDesign merge into all_mutations.csv, so heavy/light role is
    # inferred by length (heavy V-domain or V+CH1 is consistently longer than
    # light for the same construct level across every real antibody) -- a
    # documented approximation, not exact metadata.
    self_seq_wt = site.self_sequence
    self_seq_mut = self_seq_wt[:site.seq_index] + site.mut_aa + self_seq_wt[site.seq_index + 1:]
    teammate_seq = site.teammate_sequence
    partner_seq = site.partner_sequence

    # Heuristic consistent with build_abdesign.py: heavy chains are longer
    # (~120 aa) than light chains (~107 aa) for the antibody Fv; use the
    # actual chain identity carried in meta if available, else infer by
    # relative length (heavy > light in every real Fv).
    if len(self_seq_wt) >= len(teammate_seq) if teammate_seq else True:
        heavy_wt, light_wt = self_seq_wt, teammate_seq
        heavy_mut, light_mut = self_seq_mut, teammate_seq
    else:
        heavy_wt, light_wt = teammate_seq, self_seq_wt
        heavy_mut, light_mut = teammate_seq, self_seq_mut

    def score(heavy, light):
        h_emb = ablang2_embed(ablang2_model, heavy, "", device, ablang2_memo, is_heavy=True)
        l_emb = ablang2_embed(ablang2_model, light, "", device, ablang2_memo, is_heavy=False)
        self_tokens_raw = torch.cat([h_emb, l_emb], dim=0) if l_emb.shape[0] else h_emb
        self_tokens = self_proj(self_tokens_raw)
        partner_emb = esm_embed(esm_cache, partner_seq, device, esm_memo)
        pooled = model._pooled_from_tokens(self_tokens, partner_emb)
        return model._scalar_from_pooled(pooled, None)

    return score(heavy_mut, light_mut) - score(heavy_wt, light_wt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm-backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="ablang2_abdesign_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    esm_scorer = ESMScorer(model_name=args.esm_backbone, device=device)
    esm_hidden = esm_scorer.model.config.hidden_size
    esm_cache = SequenceScoreCache(esm_scorer)
    ablang2_model = ablang2.pretrained(model_to_use="ablang2-paired", device="cpu")  # its own internal device mgmt

    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    train = [e for e in train if len(e.sites) == 1]
    val = [e for e in val if len(e.sites) == 1]
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = AbsoluteAffinityModel(plm_hidden_size=esm_hidden).to(device)  # d_model default 256
    self_proj = nn.Linear(ABLANG2_HIDDEN, 256).to(device)
    ablang2_memo: dict = {}
    esm_memo: dict = {}

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(self_proj.parameters()),
                                   lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    best_score, best_state, best_proj_state, global_step = -math.inf, None, None, 0

    def evaluate(examples_):
        model.eval(); self_proj.eval()
        ids, y_true, y_pred = [], [], []
        with torch.no_grad():
            for ex in examples_:
                if ex.ddG is None:
                    continue
                pred = predict_ddg(model, self_proj, ablang2_model, ex, esm_cache, device, ablang2_memo, esm_memo).item()
                ids.append(ex.complex_id); y_true.append(ex.ddG); y_pred.append(pred)
        return full_report(ids, y_true, y_pred, min_complex_n=3)

    for epoch in range(args.epochs):
        model.train(); self_proj.train()
        rng.shuffle(train)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            valid = [ex for ex in batch if ex.ddG is not None]
            if not valid:
                global_step += 1
                continue
            for g in optimizer.param_groups:
                g["lr"] = lr_at_step(global_step, total_steps, args.lr)
            optimizer.zero_grad()
            n_valid = len(valid)
            for ex in valid:
                pred = predict_ddg(model, self_proj, ablang2_model, ex, esm_cache, device, ablang2_memo, esm_memo)
                target = torch.tensor(ex.ddG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
                epoch_loss += loss.item() * n_valid
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(self_proj.parameters()), 1.0)
            optimizer.step()
            n_seen += n_valid
            global_step += 1
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        val_report = evaluate(val)
        score = val_report["per_complex"]["mean_spearman"]
        if math.isnan(score):
            score = val_report["pooled"]["spearman"]
        print(f"epoch {epoch}: train_mse={epoch_loss / max(n_seen, 1):.4f} val_score={score:.4f}", flush=True)
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_proj_state = copy.deepcopy(self_proj.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        self_proj.load_state_dict(best_proj_state)

    test_report = evaluate(test)
    print("\nTEST:", json.dumps(test_report, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(test_report, f, indent=2)


if __name__ == "__main__":
    main()
