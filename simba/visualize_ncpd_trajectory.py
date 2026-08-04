"""NCPD trajectory visualization (§3.14): trains NCPDModel on RDE-PPI fold
0 (reusing eval_ddg_ncpd.py's training loop unchanged), then picks a few
single-point test examples spanning stabilizing/destabilizing/neutral true
ddG and calls `forward_trajectory` to capture the full WT->mutant latent
relaxation path (h(t) for t=0..n_steps, plus the decoded ddG at each step),
not just the endpoint prediction. Saves the raw trajectories (JSON) and a
2D PCA projection + per-step decoded-ddG plot (PNG).
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from simba import env

env.setup()

from simba.data.dataset import load_rde_skempi  # noqa: E402
from simba.eval_ddg_ncpd import embed, run_training  # noqa: E402
from simba.eval_rde_folds import make_val_split  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.data.affinity_ddg import full_wt_mut_sequences  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="ncpd_trajectory_data.json")
    parser.add_argument("--figure-out", default="figures/ncpd_trajectory.png")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ESMScorer(model_name=args.backbone, device=device)
    hidden = scorer.model.config.hidden_size
    cache = SequenceScoreCache(scorer)

    examples = load_rde_skempi()
    fold = sorted({e.meta["cvfold"] for e in examples})[0]
    test = [e for e in examples if e.meta["cvfold"] == fold and len(e.sites) == 1 and e.ddG is not None]
    train_full = [e for e in examples if e.meta["cvfold"] != fold]
    train, val = make_val_split(train_full, seed=fold)

    memo: dict = {}
    print(f"train={len(train)} val={len(val)} single-point test={len(test)}", flush=True)
    model, _ = run_training(train, val, hidden=hidden, epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, weight_decay=args.weight_decay, device=device, seed=fold,
                             cache=cache, memo=memo, n_steps=args.n_steps, min_n=10,
                             fold_label="[ncpd-trajectory] ")

    # pick a spread: most stabilizing, most destabilizing, and a few near-neutral
    test_sorted = sorted(test, key=lambda e: e.ddG)
    picks = []
    if len(test_sorted) >= args.n_examples:
        picks = [test_sorted[0], test_sorted[-1]]
        mid = test_sorted[len(test_sorted) // 2 - 1:len(test_sorted) // 2 + args.n_examples - 2]
        picks += mid
    else:
        picks = test_sorted

    model.eval()
    trajectories = []
    with torch.no_grad():
        for ex in picks[:args.n_examples]:
            wt_full, mut_full, partner_seq = full_wt_mut_sequences(ex)
            wt_emb = embed(cache, wt_full, device, memo, hidden)
            mut_emb = embed(cache, mut_full, device, memo, hidden)
            partner_emb = embed(cache, partner_seq, device, memo, hidden)
            traj = model.forward_trajectory(wt_emb, mut_emb, partner_emb)
            trajectories.append({
                "complex_id": ex.complex_id, "true_ddg": ex.ddG,
                "ddg_per_step": traj["ddg_per_step"],
                "trajectory": [h.cpu().numpy().tolist() for h in traj["trajectory"]],
            })
            print(f"{ex.complex_id}: true_ddG={ex.ddG:.2f} ddg_per_step={[f'{d:.2f}' for d in traj['ddg_per_step']]}",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(trajectories, f, indent=2)

    # Figure: (a) 2D PCA of every trajectory's latent path, (b) decoded ddG per step
    all_points = np.array([h for t in trajectories for h in t["trajectory"]])
    pca = PCA(n_components=2)
    pca.fit(all_points)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.coolwarm
    true_vals = [t["true_ddg"] for t in trajectories]
    vmin, vmax = min(true_vals), max(true_vals)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for t in trajectories:
        pts2d = pca.transform(np.array(t["trajectory"]))
        color = cmap(norm(t["true_ddg"]))
        axes[0].plot(pts2d[:, 0], pts2d[:, 1], "-o", color=color, markersize=4, linewidth=1.5, alpha=0.85)
        axes[0].annotate("WT", pts2d[0], fontsize=6, color=color)
        axes[0].annotate(f"{t['complex_id']}\nddG={t['true_ddg']:.1f}", pts2d[-1], fontsize=6, color=color)
        axes[1].plot(range(len(t["ddg_per_step"])), t["ddg_per_step"], "-o", color=color, markersize=4,
                     label=f"{t['complex_id']} (true={t['true_ddg']:.1f})")

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    axes[0].set_title("(a) NCPD latent trajectory, WT -> t=0..n_steps -> mutant-relaxed state\n"
                       "(color = true ΔΔG, blue=stabilizing, red=destabilizing)")
    axes[1].set_xlabel("integration step t")
    axes[1].set_ylabel("decoded ΔΔG at step t")
    axes[1].set_title("(b) Decoded ΔΔG along the relaxation path")
    axes[1].legend(fontsize=6, loc="best")
    axes[1].axhline(0, color="gray", linewidth=0.7, linestyle="--")

    fig.tight_layout()
    fig.savefig(args.figure_out, dpi=200)
    print(f"saved {args.figure_out}")


if __name__ == "__main__":
    main()
