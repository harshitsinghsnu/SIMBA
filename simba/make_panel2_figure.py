"""Publication-style composite Panel 2: "Backbone choice and predictive
accuracy."

  (a) Predicted vs. true dG scatter, pooled 5-fold CV out-of-fold
      predictions, combined pool, complex (random) split
      (figures/scatter_random_split.png, embedded).
  (b) SIMBA vs. MINT's own reported SKEMPI method, same rows/folds
      (MODEL_REPORT.md S3.21).
  (c) PLM backbone ablation on the cold_target split: ESM2-650M vs. MINT
      vs. AbLang2 as drop-in backbones (MODEL_REPORT.md S3.20).
  (d) Zero-shot vs. few-shot (10% support) generalization to genuinely
      novel complexes (MODEL_REPORT.md S3.19) -- moved here from the old
      Panel 3A, since it's a backbone/model-behavior result, not an
      interpretability one.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.figure_style import PANEL_LABEL_KW, RCPARAMS  # noqa: E402

plt.rcParams.update(RCPARAMS)


def load_trimmed(path, pad=15):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    non_white = np.any(arr < 250, axis=-1)
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, arr.shape[0]), min(c1 + pad, arr.shape[1])
    return arr[r0:r1, c0:c1]


def embed(ax, path, aspect="equal"):
    ax.imshow(load_trimmed(path), aspect=aspect)
    ax.axis("off")


def main():
    # EXACT CHANGE: Changed from (15, 16) to (16, 12) for a side-by-side 2x2 layout
    fig = plt.figure(figsize=(16, 12))
    
    # EXACT CHANGE: Changed gridspec to 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.18, wspace=0.06,
                          top=0.98, bottom=0.02, left=0.03, right=0.98)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0]) # EXACT CHANGE: Bottom Left
    ax_d = fig.add_subplot(gs[1, 1]) # EXACT CHANGE: Bottom Right

    embed(ax_a, "figures/scatter_random_split.png")
    embed(ax_b, "figures/simba_vs_mint_skempi.png", aspect="auto")
    embed(ax_c, "figures/plm_ablation_cold_split.png", aspect="auto")
    embed(ax_d, "figures/fewshot_scaling_curve.png", aspect="auto")

    ax_a.text(0.0, 0.99, "A", transform=ax_a.transAxes, **PANEL_LABEL_KW)
    ax_b.text(0.0, 0.99, "B", transform=ax_b.transAxes, **PANEL_LABEL_KW)
    ax_c.text(0.0, 0.99, "C", transform=ax_c.transAxes, **PANEL_LABEL_KW)
    ax_d.text(0.0, 0.99, "D", transform=ax_d.transAxes, **PANEL_LABEL_KW)

    out_path = "figures/panel2_backbone_accuracy.png"
    fig.savefig(out_path, dpi=300)
    print(f"saved {out_path}")
    pdf_path = "figures/panel2_backbone_accuracy.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"[panel2] saved {pdf_path}")


if __name__ == "__main__":
    main()