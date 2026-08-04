"""Error analysis for the proposed architecture: when does the model fail?

Reuses `run_training`/`predict` from eval_ddg_affinity.py completely
unchanged (same data, same splits, same training loop as the headline
RDE-PPI and AbDesign-ddG results already reported) -- the only addition is
capturing per-example (prediction, target, metadata) triples instead of
just aggregate metrics, then slicing residuals by failure-mode category:

  - Multi-point vs single-point mutations (`len(ex.sites)` / `n_mutations`),
    available on every benchmark.
  - Interface location -- COR/RIM/SUP (at or adjacent to the binding
    interface) vs INT (buried, away from the interface) vs SUR (surface,
    non-interface) -- from SKEMPI2's own `iMutation_Location(s)` structural
    annotation, carried through into rde_skempi_mutations.csv as
    `interface_location` (see build_rde_skempi.py). Only available for
    RDE-PPI (SKEMPI2-derived); multi-mutation rows get one location string
    per site (e.g. "COR,RIM"), bucketed by majority vote across sites.
  - CDR vs framework mutations -- from AbDesign's own IMGT-derived `is_cdr`
    flag (build_abdesign.py), available for the AbDesign ddG subset
    (SKEMPIv2 + AB-Bind antibody-antigen rows the AbDesign authors curated).

Trains once per benchmark (not per-seed -- this is a diagnostic breakdown of
a single representative run, not a headline number), on the same splits as
the corresponding multi-seed/single-seed results already in MODEL_REPORT.md.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402
from simba.data.dataset import load_all_mutations, load_rde_skempi  # noqa: E402
from simba.eval_ddg_affinity import embed, predict, run_training  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.train import split_examples  # noqa: E402

INTERFACE_BUCKETS = {"COR": "interface", "RIM": "interface", "SUP": "interface",
                      "SUR": "surface", "INT": "buried"}


def bucket_interface_location(raw: str | float | None) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    buckets = [INTERFACE_BUCKETS.get(t) for t in tokens]
    buckets = [b for b in buckets if b is not None]
    if not buckets:
        return None
    # majority vote across sites of a multi-point mutation; ties -> "mixed"
    counts = {b: buckets.count(b) for b in set(buckets)}
    top = max(counts.values())
    winners = [b for b, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else "mixed"


def extract_attention_for_example(model, example, cache, device, memo, max_self_len=80):
    """Replicate AbsoluteAffinityModel's pairformer forward pass by hand
    (model._pooled -> proj -> pairwise_block) but with need_weights=True on
    the single PairformerLayer's row/col MultiheadAttention calls, so the
    real attention matrices used for this example's prediction can be
    visualized. Only meaningful for `use_pairformer=True, n_pairformer_layers
    =1` (the proposed architecture -- the only config this is called with),
    since with >1 layers "the" attention map is ambiguous (which layer?).
    Returns None if the example's self-chain is too long for a legible
    heatmap or the model wasn't built with exactly one pairformer layer.
    """
    if not model.use_pairformer or len(model.pairwise_block.layers) != 1:
        return None
    wt_full, mut_full, partner_seq = full_wt_mut_sequences(example)
    if len(mut_full) > max_self_len or not partner_seq:
        return None

    layer = model.pairwise_block.layers[0]
    with torch.no_grad():
        self_emb = embed(cache, mut_full, device, memo)
        partner_emb = embed(cache, partner_seq, device, memo)
        self_tokens = model.proj(self_emb)
        partner_tokens = model.proj(partner_emb)
        memory = model.partner_memory.write(partner_tokens)

        d_model = self_tokens.shape[-1]
        wq, m = self_tokens.shape[0], memory.shape[0]
        q_exp = self_tokens.unsqueeze(1).expand(wq, m, d_model)
        m_exp = memory.unsqueeze(0).expand(wq, m, d_model)
        pair = model.pairwise_block.pair_proj(torch.cat([q_exp, m_exp], dim=-1))

        _, row_weights = layer.row_attn(pair, pair, pair, need_weights=True, average_attn_weights=True)
        pair_t = pair.transpose(0, 1)
        _, col_weights = layer.col_attn(pair_t, pair_t, pair_t, need_weights=True, average_attn_weights=True)

    return {
        "complex_id": example.complex_id,
        "self_length": wq,
        "n_memory": m,
        "n_mutations": len(example.sites),
        # row_weights: [Wq, M, M] (per self-residue, attention over memory slots x memory slots) --
        # average over the query (first) axis to get a single [M, M] memory-memory attention summary.
        "row_attention_memory_x_memory": row_weights.mean(dim=0).cpu().tolist(),
        # col_weights: [M, Wq, Wq] (per memory slot, attention over self-residues x self-residues) --
        # average over memory slots to get a single [Wq, Wq] self-residue attention summary.
        "col_attention_self_x_self": col_weights.mean(dim=0).cpu().tolist(),
    }


def category_metrics(records: list[dict], group_key: str) -> dict:
    groups: dict = {}
    for r in records:
        groups.setdefault(r[group_key], []).append(r)
    out = {}
    for cat, rows in groups.items():
        y_true = np.array([r["y_true"] for r in rows])
        y_pred = np.array([r["y_pred"] for r in rows])
        resid = y_pred - y_true
        entry = {
            "n": len(rows),
            "mae": float(np.mean(np.abs(resid))),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "bias": float(np.mean(resid)),
        }
        if len(rows) >= 5 and np.std(y_pred) > 0 and np.std(y_true) > 0:
            entry["pearson"] = float(pearsonr(y_true, y_pred)[0])
            entry["spearman"] = float(spearmanr(y_true, y_pred)[0])
        else:
            entry["pearson"] = None
            entry["spearman"] = None
        out[str(cat)] = entry
    return out


def run_rde_error_analysis(cache, hidden, epochs, batch_size, lr, weight_decay, device, n_attention_examples=4):
    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    records = []
    attention_examples = []
    for fold in folds:
        memo: dict = {}
        print(f"\n=== [error-analysis] RDE-PPI fold {fold} ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, min_n=10, fold_label=f"[fold {fold}] ")
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                pred = predict(model, ex, cache, device, memo).item()
                n_mut = len(ex.sites)
                records.append({
                    "complex_id": ex.complex_id,
                    "y_true": ex.ddG,
                    "y_pred": pred,
                    "n_mutations": n_mut,
                    "mutation_arity": "single-point" if n_mut == 1 else "multi-point",
                    "interface_bucket": bucket_interface_location(ex.meta.get("interface_location")),
                })
        if fold == folds[0]:
            print(f"\n=== [error-analysis] extracting attention weights (fold {fold} model) ===", flush=True)
            for ex in test:
                if len(attention_examples) >= n_attention_examples:
                    break
                att = extract_attention_for_example(model, ex, cache, device, memo)
                if att is not None:
                    attention_examples.append(att)

    report = {
        "n_examples": len(records),
        "by_mutation_arity": category_metrics(records, "mutation_arity"),
        "by_interface_location": category_metrics(
            [r for r in records if r["interface_bucket"] is not None], "interface_bucket"),
    }
    return report, records, attention_examples


def run_abdesign_error_analysis(cache, hidden, epochs, batch_size, lr, weight_decay, device):
    examples = load_all_mutations()
    train, val, test = split_examples(examples, "abdesign_holdout")
    memo: dict = {}
    model, _ = run_training(train, val, hidden=hidden, epochs=epochs, batch_size=batch_size,
                             lr=lr, weight_decay=weight_decay, device=device, seed=0,
                             cache=cache, memo=memo, min_n=3)
    model.eval()
    records = []
    with torch.no_grad():
        for ex in test:
            if ex.ddG is None:
                continue  # error analysis restricted to the ddG-labeled subset (elisa collapsed already)
            pred = predict(model, ex, cache, device, memo).item()
            n_mut = len(ex.sites)
            is_cdr = ex.meta.get("is_cdr")
            records.append({
                "complex_id": ex.complex_id,
                "y_true": ex.ddG,
                "y_pred": pred,
                "n_mutations": n_mut,
                "mutation_arity": "single-point" if n_mut == 1 else "multi-point",
                "cdr_region": (None if is_cdr is None or (isinstance(is_cdr, float) and np.isnan(is_cdr))
                               else ("CDR" if bool(is_cdr) else "framework")),
            })

    report = {
        "n_examples": len(records),
        "by_mutation_arity": category_metrics(records, "mutation_arity"),
        "by_cdr_region": category_metrics(
            [r for r in records if r["cdr_region"] is not None], "cdr_region"),
    }
    return report, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--benchmarks", nargs="+", default=["rde", "abdesign"], choices=["rde", "abdesign"])
    parser.add_argument("--rde-epochs", type=int, default=15)
    parser.add_argument("--abdesign-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="error_analysis_results.json")
    parser.add_argument("--records-out", default="error_analysis_records.json")
    parser.add_argument("--attention-out", default="attention_weights.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results, all_records = {}, {}

    if "rde" in args.benchmarks:
        report, records, attention_examples = run_rde_error_analysis(
            cache, hidden, args.rde_epochs, args.batch_size, args.lr, args.weight_decay, device)
        results["rde"] = report
        all_records["rde"] = records
        print("\n[RDE-PPI error analysis]", json.dumps(report, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        with open(args.records_out, "w") as f:
            json.dump(all_records, f, indent=2)
        with open(args.attention_out, "w") as f:
            json.dump({"rde": attention_examples}, f)
        print(f"[attention] saved {len(attention_examples)} example(s) to {args.attention_out}", flush=True)

    if "abdesign" in args.benchmarks:
        report, records = run_abdesign_error_analysis(cache, hidden, args.abdesign_epochs, args.batch_size,
                                                        args.lr, args.weight_decay, device)
        results["abdesign_ddg"] = report
        all_records["abdesign_ddg"] = records
        print("\n[AbDesign ddG error analysis]", json.dumps(report, indent=2), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        with open(args.records_out, "w") as f:
            json.dump(all_records, f, indent=2)

    print("\n\nFINAL ERROR ANALYSIS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
