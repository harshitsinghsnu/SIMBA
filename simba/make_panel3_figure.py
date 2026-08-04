"""Publication-style composite Panel 3: "Interpretability."

  (a) Integrated-Gradients attribution/true-interface overlap, 1CSE.
  (b) Integrated-Gradients attribution/true-interface overlap,
      trypsin-BPTI (MODEL_REPORT.md S3.18).

The zero-shot/few-shot generalization sub-panel that used to be Panel 3A
has moved to Panel 2D (see make_panel2_figure.py) -- this panel now holds
only the interpretability comparison, so it isn't diluted by an unrelated
third topic.
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


def embed(ax, path, aspect="auto"):
    ax.imshow(load_trimmed(path), aspect=aspect)
    ax.axis("off")


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 7.2))

    embed(ax_a, "figures/best_overlap_1cse_labeled.png")
    embed(ax_b, "figures/best_overlap_trypsin_bpti_labeled.png")

    ax_a.text(0.0, 0.99, "A", transform=ax_a.transAxes, **PANEL_LABEL_KW)
    ax_b.text(0.0, 0.99, "B", transform=ax_b.transAxes, **PANEL_LABEL_KW)

    fig.tight_layout()

    out_path = "figures/panel3_generalization_interpretability.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")
    pdf_path = "figures/panel3_generalization_interpretability.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"[panel3] saved {pdf_path}")


if __name__ == "__main__":
    main()
