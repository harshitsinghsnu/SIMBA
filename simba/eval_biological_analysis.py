"""Biological error analysis for RDE-PPI: does prediction error correlate
with structural/evolutionary properties of the mutated residue, not just
statistical categories (multi-point vs. single, §3.12's coarse
interface/buried/surface buckets)? Four biological dimensions, all derived
from data already on disk (PDB structures already downloaded for RDE-PPI,
frozen-PLM log-probabilities already computed by the embedding cache) --
no new external data source, consistent with this project's sequence-only,
reuse-what's-already-there approach:

  1. Full Core/Rim/Support/Surface/Interior breakdown (SKEMPI2's own
     `iMutation_Location(s)` structural classification, all 5 categories
     kept separate -- §3.12 collapsed COR/RIM/SUP into one "interface"
     bucket; this script reports them individually).
  2. Geometric distance from the mutated residue's CA to the nearest CA in
     the partner chain(s) -- a real continuous distance-to-interface
     measurement (from the PDB structures already downloaded for RDE-PPI),
     not the categorical proxy in (1). Post-hoc diagnostic only, never fed
     into the model.
  3. Conservation: the frozen ESM2 backbone's own wildtype-marginal
     log-probability at the mutated position (already computed by
     `SequenceScoreCache` for every training run in this project -- no new
     inference needed beyond what training already does). Standard,
     well-established PLM-likelihood-as-conservation-proxy technique.
  4. Flexibility: the crystal structure's own CA B-factor at the mutated
     residue (already in the PDB files on disk) -- a cheap, no-new-pipeline
     proxy for local backbone mobility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.dataset import load_rde_skempi  # noqa: E402
from simba.data.pdb_utils import min_distance_to_chains, parse_pdb_bfactors, parse_pdb_ca_coords  # noqa: E402
from simba.eval_ddg_affinity import predict, run_training  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer, STANDARD_AAS  # noqa: E402

PDB_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "skempi2" / "PDBs"
_AA_TO_COL = {aa: i for i, aa in enumerate(STANDARD_AAS)}


def site_biology(ex, cache, wt_full_seq, pdb_cache):
    """Returns (interface_location, distance_to_partner, conservation,
    bfactor) for the FIRST mutation site of a (typically single-point)
    example -- multi-point examples are skipped for this analysis, since
    each of these four properties is inherently per-site."""
    if len(ex.sites) != 1:
        return None
    site = ex.sites[0]
    raw = ex.meta.get("sites_raw")
    if not raw or len(raw) != 1:
        return None
    raw_site = raw[0]
    # load_rde_skempi() uses `pdbcode` as the complex_id column (so it's
    # excluded from `meta`, which only keeps non-identifier columns) --
    # `ex.complex_id` *is* the bare PDB code here, e.g. "1CSE".
    pdbcode = ex.complex_id
    group_a = set(ex.meta.get("group_a_chains") or [])
    group_b = set(ex.meta.get("group_b_chains") or [])
    chain = raw_site["chain"]
    partner_chains = group_b if chain in group_a else group_a

    if pdbcode not in pdb_cache:
        pdb_path = PDB_DIR / f"{pdbcode}.pdb"
        if pdb_path.exists():
            pdb_cache[pdbcode] = {
                "bfactors": parse_pdb_bfactors(pdb_path),
                "coords": parse_pdb_ca_coords(pdb_path),
            }
        else:
            pdb_cache[pdbcode] = None
    entry = pdb_cache[pdbcode]

    distance, bfactor = None, None
    if entry is not None:
        icode = (raw_site.get("icode") or " ").upper() or " "
        key = (raw_site["resnum"], icode)
        point = entry["coords"].get(chain, {}).get(key)
        if point is not None:
            distance = min_distance_to_chains(point, entry["coords"], partner_chains)
        bfactor = entry["bfactors"].get((chain, raw_site["resnum"], icode))

    conservation = None
    wt_aa = site.wt_aa
    if wt_aa in _AA_TO_COL and 0 <= site.seq_index < len(wt_full_seq):
        score = cache.get(wt_full_seq)
        if site.seq_index < score.logprobs.shape[0]:
            conservation = float(score.logprobs[site.seq_index, _AA_TO_COL[wt_aa]])

    return ex.meta.get("interface_location"), distance, conservation, bfactor


def bucket_and_correlate(records, key, n_bins=4):
    vals = np.array([r[key] for r in records if r[key] is not None])
    errs = np.array([r["abs_error"] for r in records if r[key] is not None])
    if len(vals) < 10:
        return {"n": len(vals), "pearson": None, "spearman": None, "bins": []}
    pearson = float(pearsonr(vals, errs)[0]) if np.std(vals) > 0 else float("nan")
    spearman = float(spearmanr(vals, errs)[0]) if np.std(vals) > 0 else float("nan")
    order = np.argsort(vals)
    bins = np.array_split(order, min(n_bins, len(vals)))
    bin_stats = [{"mean_" + key: float(vals[b].mean()), "mean_abs_error": float(errs[b].mean()), "n": len(b)}
                 for b in bins]
    return {"n": len(vals), "pearson": pearson, "spearman": spearman, "bins": bin_stats}


def category_mae(records, key):
    groups: dict = {}
    for r in records:
        v = r[key]
        if v is None:
            continue
        groups.setdefault(v, []).append(r["abs_error"])
    return {str(k): {"mae": float(np.mean(v)), "n": len(v)} for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="biological_analysis_results.json")
    parser.add_argument("--records-out", default="biological_analysis_records.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    examples = load_rde_skempi()
    folds = sorted({e.meta["cvfold"] for e in examples})
    records = []
    pdb_cache: dict = {}

    for fold in folds:
        memo: dict = {}
        print(f"\n=== [biological-analysis] RDE-PPI fold {fold} ===", flush=True)
        test = [e for e in examples if e.meta["cvfold"] == fold]
        train_full = [e for e in examples if e.meta["cvfold"] != fold]
        train, val = make_val_split(train_full, seed=fold)
        model, _ = run_training(train, val, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
                                 lr=args.lr, weight_decay=args.weight_decay, device=device, seed=fold,
                                 cache=cache, memo=memo, min_n=10, fold_label=f"[fold {fold}] ")
        model.eval()
        with torch.no_grad():
            for ex in test:
                if ex.ddG is None:
                    continue
                from simba.data.affinity_ddg import full_wt_mut_sequences
                wt_full, _, _ = full_wt_mut_sequences(ex)
                bio = site_biology(ex, cache, wt_full, pdb_cache)
                if bio is None:
                    continue
                interface_loc, distance, conservation, bfactor = bio
                pred = predict(model, ex, cache, device, memo).item()
                records.append({
                    "complex_id": ex.complex_id, "y_true": ex.ddG, "y_pred": pred,
                    "abs_error": abs(pred - ex.ddG),
                    "interface_location": interface_loc,
                    "distance_to_partner": distance,
                    "conservation": conservation,
                    "bfactor": bfactor,
                })

    print(f"\n[biological-analysis] {len(records)} single-point examples with full biological annotation "
          f"(of {sum(1 for e in examples if e.ddG is not None)} total RDE-PPI examples)", flush=True)

    report = {
        "n_examples": len(records),
        "by_interface_location_full": category_mae(records, "interface_location"),
        "distance_to_partner_vs_error": bucket_and_correlate(records, "distance_to_partner"),
        "conservation_vs_error": bucket_and_correlate(records, "conservation"),
        "bfactor_vs_error": bucket_and_correlate(records, "bfactor"),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    with open(args.records_out, "w") as f:
        json.dump(records, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
