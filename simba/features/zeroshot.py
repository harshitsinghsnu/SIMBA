"""Zero-shot background term: sum of per-site wildtype-marginal log-probability
deltas, log P(mutant aa) - log P(wildtype aa), under a frozen PLM. This is the
free, complex-agnostic 'how tolerated is this substitution' signal that
SIMBA's learned selection head corrects with partner/interface context."""

from __future__ import annotations

import numpy as np

from simba.features.plm import STANDARD_AAS, SequenceScore

_AA_TO_COL = {aa: i for i, aa in enumerate(STANDARD_AAS)}


def site_delta_logp(score: SequenceScore, seq_index: int, wt_aa: str, mut_aa: str) -> float | None:
    if wt_aa not in _AA_TO_COL or mut_aa not in _AA_TO_COL or seq_index is None:
        return None
    if not (0 <= seq_index < score.logprobs.shape[0]):
        return None
    wt_col, mut_col = _AA_TO_COL[wt_aa], _AA_TO_COL[mut_aa]
    return float(score.logprobs[seq_index, mut_col] - score.logprobs[seq_index, wt_col])


def mutation_delta_logp(score: SequenceScore, mutations: list[dict]) -> float | None:
    """Sum of per-site deltas (additive approximation). Returns None if any
    site can't be scored (non-standard residue or unresolved position), so
    callers can distinguish 'unscoreable' from a genuine zero delta."""
    total = 0.0
    for m in mutations:
        d = site_delta_logp(score, m.get("seq_index"), m["wt_aa"], m["mut_aa"])
        if d is None:
            return None
        total += d
    return total


def site_delta_logp_masked(
    logprobs_at_position: np.ndarray | None, wt_aa: str, mut_aa: str
) -> float | None:
    """Same as site_delta_logp but reading from a masked-marginal lookup
    (a single [20]-length vector for one specific position) rather than a
    full-sequence wildtype-marginal SequenceScore."""
    if logprobs_at_position is None or wt_aa not in _AA_TO_COL or mut_aa not in _AA_TO_COL:
        return None
    wt_col, mut_col = _AA_TO_COL[wt_aa], _AA_TO_COL[mut_aa]
    return float(logprobs_at_position[mut_col] - logprobs_at_position[wt_col])
