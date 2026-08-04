"""Publication-style architecture schematic for Section 3.2 (Architecture):
two frozen ESM-2 encoders -> interface memory (cross-attention write) ->
Pairformer-lite refinement (row + column attention) -> MLP head -> scalar
dG, plus the ddG dual-call subtraction. Pure matplotlib box-and-arrow
diagram (no external drawing tool) so it's reproducible from this repo
alone. Box labels are deliberately terse -- the mechanism detail lives in
the LaTeX body text and caption, not crammed into the diagram itself.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from simba.figure_style import RCPARAMS, sequential_blues  # noqa: E402

plt.rcParams.update(RCPARAMS)

_BLUES = sequential_blues(5)  # lightest -> darkest, reused across boxes by role


def box(ax, xy, w, h, text, color, fontsize=14, fontweight="normal", text_color="black", lw=1.2):
    x, y = xy
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor="black", facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="black", lw=1.8, style="-|>", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18, linewidth=lw,
                        color=color, zorder=1, connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def center_right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def center_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    grey = "#e8e8e8"
    esm_color = _BLUES[1]
    mem_color = _BLUES[2]
    pair_color = _BLUES[3]
    head_color = _BLUES[4]

    # --- Input sequences ---
    b_self_seq = box(ax, (0.3, 6.7), 2.5, 1.0, "Self chain $A$\n(sequence)", grey)
    b_partner_seq = box(ax, (0.3, 3.9), 2.5, 1.0, "Partner chain $B$\n(sequence)", grey)

    # --- Frozen ESM-2 encoders (shared weights) ---
    b_esm_self = box(ax, (3.4, 6.7), 2.3, 1.0, "ESM-2-650M\n(frozen)", esm_color,
                     fontweight="bold", text_color="white")
    b_esm_partner = box(ax, (3.4, 3.9), 2.3, 1.0, "ESM-2-650M\n(frozen,\nshared)", esm_color,
                        fontweight="bold", text_color="white")

    arrow(ax, center_right(b_self_seq), center_left(b_esm_self))
    arrow(ax, center_right(b_partner_seq), center_left(b_esm_partner))

    # --- Linear projection ---
    b_proj_self = box(ax, (6.3, 6.7), 1.7, 1.0, "Projection", "white")
    b_proj_partner = box(ax, (6.3, 3.9), 1.7, 1.0, "Projection", "white")
    arrow(ax, center_right(b_esm_self), center_left(b_proj_self))
    arrow(ax, center_right(b_esm_partner), center_left(b_proj_partner))

    # --- Interface memory (partner side writes in) ---
    b_memory = box(ax, (8.6, 4.05), 2.9, 1.9, "Interface memory\n($M{=}24$ latents)",
                   mem_color, fontweight="bold", text_color="white")
    arrow(ax, center_right(b_proj_partner), (8.6, 4.4), connectionstyle="arc3,rad=-0.15")

    # --- Pairformer-lite block ---
    b_pair = box(ax, (8.6, 6.3), 2.9, 1.9, "Pairformer-lite\n(row + col attn)",
                pair_color, fontweight="bold", text_color="white")
    arrow(ax, center_right(b_proj_self), (8.6, 6.9), connectionstyle="arc3,rad=0.15")
    arrow(ax, center_top(b_memory), center_bottom(b_pair), color="#333333", lw=1.6,
          connectionstyle="arc3,rad=0.0")

    # --- Pool + head ---
    b_pool = box(ax, (11.9, 6.7), 1.3, 1.0, "Mean\npool", "white")
    arrow(ax, center_right(b_pair), center_left(b_pool))

    b_head = box(ax, (13.5, 6.7), 1.0, 1.0, "MLP\nhead", head_color,
                fontweight="bold", text_color="white")
    arrow(ax, center_right(b_pool), center_left(b_head))

    ax.annotate(r"$\Delta G(A,B)$", xy=(14.5, 7.2), xytext=(14.65, 7.2),
                fontsize=17, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))

    # --- ddG inset (bottom) ---
    inset_y = 0.4
    b_wt = box(ax, (0.3, inset_y), 2.6, 1.0, "$f_\\theta(A, B)$", "white")
    b_mut = box(ax, (4.6, inset_y), 2.6, 1.0, "$f_\\theta(A', B)$", "white")
    b_sub = box(ax, (8.9, inset_y), 2.2, 1.0, "subtract", grey)
    arrow(ax, center_right(b_wt), center_left(b_sub), connectionstyle="arc3,rad=-0.25")
    arrow(ax, center_right(b_mut), center_left(b_sub), connectionstyle="arc3,rad=0.15")
    ax.annotate(r"$\Delta\Delta G = f_\theta(A',B) - f_\theta(A,B)$",
                xy=(11.4, inset_y + 0.5), xytext=(11.6, inset_y + 0.5),
                fontsize=16, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))

    ax.text(0.1, 8.0, "A", fontsize=24, fontweight="bold")
    ax.text(0.1, 1.75, "B", fontsize=24, fontweight="bold")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = f"figures/architecture_diagram.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
