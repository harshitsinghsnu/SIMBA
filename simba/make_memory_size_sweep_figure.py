"""Interface-memory-size sensitivity sweep figure (reviewer question),
single-seed, cold-target split, combined pool. n_memory=48's fold 4 hit a
CUDA out-of-memory error after an extended run (logs/overnight_*/
05_memory_size_sweep.log) and is excluded -- only M in {8, 16, 24, 32}
completed. Numbers are hardcoded from that log (the crash happened before
this run's incremental-save fix was added to eval_memory_size_sweep.py, so
no raw predictions survived for M=48 -- and none of the completed values
were saved to JSON either, since the old script only wrote its output file
once at the very end of the whole loop). No bootstrap CI is shown: with no
raw per-example predictions surviving, only the point Pearson/Spearman/RMSE
from the log are available, single-seed.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.figure_style import PANEL_LABEL_KW, RCPARAMS, sequential_purples  # noqa: E402

plt.rcParams.update(RCPARAMS)

# From logs/overnight_20260727_234422/05_memory_size_sweep.log (single seed,
# cold-target split, combined pool, 7,247 examples). M=48 excluded (CUDA OOM
# at fold 4, after M=8/16/24/32 had already completed).
RESULTS = [
    {"n_memory": 8, "pearson": 0.6616, "spearman": 0.6800, "rmse": 2.4943},
    {"n_memory": 16, "pearson": 0.6655, "spearman": 0.6790, "rmse": 2.4782},
    {"n_memory": 24, "pearson": 0.6513, "spearman": 0.6614, "rmse": 2.5442},
    {"n_memory": 32, "pearson": 0.6626, "spearman": 0.6761, "rmse": 2.5182},
]


def main():
    ns = [r["n_memory"] for r in RESULTS]
    color = sequential_purples(3)[2]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    metrics = [("pearson", "Pearson correlation"), ("spearman", "Spearman correlation"),
               ("rmse", "RMSE, dG (kcal/mol)")]

    for ax, (key, ylabel) in zip(axes, metrics):
        vals = [r[key] for r in RESULTS]
        ax.plot(ns, vals, "o-", color=color, markersize=11, linewidth=2, markeredgecolor="black",
                markeredgewidth=0.8)
        ax.axvline(24, color="gray", linewidth=1, linestyle="--", label="M=24 (used throughout)")
        ax.set_xlabel("Interface memory size, M")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ns)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=12.5, loc="lower right", framealpha=0.9)

    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    fig.tight_layout()
    out_path = "figures/memory_size_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    fig.savefig("figures/memory_size_sweep.pdf", bbox_inches="tight")
    print("saved figures/memory_size_sweep.pdf")

    pearsons = [r["pearson"] for r in RESULTS]
    print(f"\nPearson range across M=8-32: {min(pearsons):.4f}-{max(pearsons):.4f} "
          f"(spread={max(pearsons) - min(pearsons):.4f}); M=24 gives {dict(zip(ns, pearsons))[24]:.4f}, "
          f"not the best of the four values tested.")


if __name__ == "__main__":
    main()
