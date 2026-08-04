"""Shared publication-style settings for every make_*_figure.py script:
a real sequential colormap (not ad hoc hex, and not mixing a neutral grey
into an otherwise-blue progression) for ordered categories -- architecture
complexity, split strictness, backbone identity -- plus narrower bars with
visible gaps, which reads as "publication style" rather than the default
matplotlib wide-bar look.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 18,
    "axes.titlesize": 18.5,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15.5,
    "font.size": 16,
}

# Panel letters (A/B/C...) -- kept as the one "heading"-like element on any
# panel, since figures are referenced by sub-panel letter from the LaTeX
# caption and need one to be identifiable at all.
PANEL_LABEL_KW = dict(fontsize=22, fontweight="bold", family="sans-serif")

# A single accent hue (the project's existing SIMBA blue) used as a
# genuine sequential colormap rather than picking unrelated named colors
# per bar -- darker = "more of the thing being compared" (more capable
# architecture, our own method, deeper backbone), so hue order carries
# meaning instead of being decorative.
_CMAP = plt.get_cmap("Blues")
_CMAP_PURPLES = plt.get_cmap("Purples")


def _sequential(cmap, n: int, lo: float, hi: float) -> list:
    if n == 1:
        return [cmap(hi)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def sequential_blues(n: int, lo: float = 0.42, hi: float = 0.88) -> list:
    """n colors evenly spaced along the Blues colormap, lightest first.
    Reserved for panels that compare against MINT or plot the main
    architecture-ablation matrix (Panels 1-2) -- dark = SIMBA/proposed."""
    return _sequential(_CMAP, n, lo, hi)


def sequential_purples(n: int, lo: float = 0.42, hi: float = 0.88) -> list:
    """Same idea as `sequential_blues`, different hue -- used for figures
    whose bars/cells don't map onto the same "dark blue = SIMBA" comparison
    Panels 1-2 already use, so a reader can't misread the two figures as
    sharing one color legend when they don't (e.g. the reviewer-experiments
    figure's gate/head ablations, which are internal SIMBA variants, not a
    SIMBA-vs-baseline comparison)."""
    return _sequential(_CMAP_PURPLES, n, lo, hi)


# Neutral, desaturated tone for "baseline / not ours" series, kept outside
# the sequential blue family so it reads as qualitatively different (not
# just "a lighter version of us") without breaking the blue progression
# used for SIMBA's own ordered categories.
BASELINE_COLOR = "#b0b0b0"

BAR_WIDTH_SOLO = 0.5      # single bar per category
BAR_GROUP_WIDTH = 0.62    # total width used by a group of bars sharing one category
