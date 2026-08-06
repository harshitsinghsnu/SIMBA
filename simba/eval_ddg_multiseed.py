"""Multi-seed reruns of the two headline ddG results (RDE-PPI, AbDesign ddG
subset) for the proposed architecture, reporting mean +/- std instead of a
single run. Reuses `eval_rde`/`eval_abdesign` from eval_ddg_affinity.py
unchanged; the only addition is `torch.manual_seed(seed)` before each full
run (the original single-seed scripts never controlled torch's global RNG
for model init/dropout, only the data-shuffling order via Python's own
`random.Random(seed)`).

Deliberately scoped to RDE-PPI and AbDesign's ddG subset -- the two
headline wins (vs. FoldX and vs. the reported SOTA-collapse
baselines respectively) -- not the full 5-benchmark matrix; ProtAttBA's
S1131 (also a clear win) and AB645/AB1101 (already near-zero/noise, not a
"winning" result) are left at single-seed given the added multi-hour cost
of repeating full 10-fold CV three times each. Excludes the
cosine-similarity head and hierarchical H<->L cross-attention entirely, per
instruction -- both already established as non-winning ablations.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from simba import env

env.setup()

from simba.eval_ddg_affinity import eval_abdesign, eval_rde  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--benchmarks", nargs="+", default=["rde", "abdesign"], choices=["rde", "abdesign"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rde-epochs", type=int, default=15)
    parser.add_argument("--abdesign-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="ddg_multiseed_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    results = {}

    if "rde" in args.benchmarks:
        overall_p, overall_s, struct_p, struct_s, predictions_by_seed = [], [], [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed)
            print(f"\n=== RDE-PPI multiseed run, seed={seed} ===", flush=True)
            r = eval_rde(cache, hidden, args.rde_epochs, args.batch_size, args.lr, args.weight_decay, device)
            overall_p.append(r["overall"]["pearson"]); overall_s.append(r["overall"]["spearman"])
            struct_p.append(r["per_struct"]["mean_pearson"]); struct_s.append(r["per_struct"]["mean_spearman"])
            predictions_by_seed.append({"seed": seed, **r["predictions"]})
            print(f"[RDE-PPI seed={seed}] overall P/S={r['overall']['pearson']:.4f}/{r['overall']['spearman']:.4f} "
                  f"per_struct P/S={r['per_struct']['mean_pearson']:.4f}/{r['per_struct']['mean_spearman']:.4f}",
                  flush=True)
        results["rde"] = {
            "overall_pearson_mean": float(np.mean(overall_p)), "overall_pearson_std": float(np.std(overall_p)),
            "overall_spearman_mean": float(np.mean(overall_s)), "overall_spearman_std": float(np.std(overall_s)),
            "per_struct_pearson_mean": float(np.mean(struct_p)), "per_struct_pearson_std": float(np.std(struct_p)),
            "per_struct_spearman_mean": float(np.mean(struct_s)), "per_struct_spearman_std": float(np.std(struct_s)),
            "n_seeds": len(args.seeds),
            "predictions_by_seed": predictions_by_seed,
        }
        print(f"\n[RDE-PPI] overall P={results['rde']['overall_pearson_mean']:.4f}"
              f"+/-{results['rde']['overall_pearson_std']:.4f} "
              f"S={results['rde']['overall_spearman_mean']:.4f}+/-{results['rde']['overall_spearman_std']:.4f}  "
              f"per_struct P={results['rde']['per_struct_pearson_mean']:.4f}"
              f"+/-{results['rde']['per_struct_pearson_std']:.4f} "
              f"S={results['rde']['per_struct_spearman_mean']:.4f}"
              f"+/-{results['rde']['per_struct_spearman_std']:.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if "abdesign" in args.benchmarks:
        pooled_p, pooled_s, struct_p, struct_s, predictions_by_seed = [], [], [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed)
            print(f"\n=== AbDesign multiseed run, seed={seed} ===", flush=True)
            r = eval_abdesign(cache, hidden, args.abdesign_epochs, args.batch_size, args.lr,
                              args.weight_decay, device)
            ddg = r["ddg"]
            pooled_p.append(ddg["pooled"]["pearson"]); pooled_s.append(ddg["pooled"]["spearman"])
            struct_p.append(ddg["per_complex"]["mean_pearson"]); struct_s.append(ddg["per_complex"]["mean_spearman"])
            predictions_by_seed.append({"seed": seed, **r["predictions"]["ddg"]})
            print(f"[AbDesign seed={seed}] pooled P/S={ddg['pooled']['pearson']:.4f}/{ddg['pooled']['spearman']:.4f} "
                  f"per_complex P/S={ddg['per_complex']['mean_pearson']:.4f}/{ddg['per_complex']['mean_spearman']:.4f}",
                  flush=True)
        results["abdesign_ddg"] = {
            "pooled_pearson_mean": float(np.mean(pooled_p)), "pooled_pearson_std": float(np.std(pooled_p)),
            "pooled_spearman_mean": float(np.mean(pooled_s)), "pooled_spearman_std": float(np.std(pooled_s)),
            "per_complex_pearson_mean": float(np.mean(struct_p)), "per_complex_pearson_std": float(np.std(struct_p)),
            "per_complex_spearman_mean": float(np.mean(struct_s)), "per_complex_spearman_std": float(np.std(struct_s)),
            "n_seeds": len(args.seeds),
            "predictions_by_seed": predictions_by_seed,
        }
        print(f"\n[AbDesign ddG] pooled P={results['abdesign_ddg']['pooled_pearson_mean']:.4f}"
              f"+/-{results['abdesign_ddg']['pooled_pearson_std']:.4f}  "
              f"per_complex S={results['abdesign_ddg']['per_complex_spearman_mean']:.4f}"
              f"+/-{results['abdesign_ddg']['per_complex_spearman_std']:.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
