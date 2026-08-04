"""Builds the (wildtype self-sequence, mutant self-sequence, partner
sequence) triple needed to score mutation effects with
`AbsoluteAffinityModel.forward_ddg` -- the same absolute-affinity model used
for dG prediction, called twice (once per sequence) and diffed, instead of a
separate mutation-effect architecture.

Multi-site mutations are handled by substituting every site directly into
one shared string, so multi-chain cases (e.g. mutations on both the heavy
and light chain of one antibody) still produce a single fully-mutated
sequence rather than two half-mutated ones: each site's `self_sequence`/
`teammate_sequence` is matched by content against a canonical concatenation
of the complex's own-side chains (built from the first site), giving every
site the correct offset within that shared string regardless of which site
happens to be "self" vs "teammate" for it.
"""

from __future__ import annotations

from simba.data.dataset import Example


def full_wt_mut_sequences(example: Example) -> tuple[str, str, str]:
    """Returns (wildtype_self_sequence, mutant_self_sequence, partner_sequence)."""
    sites = example.sites
    first = sites[0]
    chain_order = [c for c in (first.self_sequence, first.teammate_sequence) if c]

    offsets: dict[str, int] = {}
    pos = 0
    for chain in chain_order:
        offsets[chain] = pos
        pos += len(chain)
    wt_full = "".join(chain_order)

    mut_chars = list(wt_full)
    for site in sites:
        offset = offsets.get(site.self_sequence)
        if offset is None:
            continue  # site's chain isn't part of this example's canonical self-side order
        mut_chars[offset + site.seq_index] = site.mut_aa
    mut_full = "".join(mut_chars)

    return wt_full, mut_full, first.partner_sequence


def paired_wt_mut_sequences(example: Example) -> tuple[str, str, str, str, str]:
    """Like `full_wt_mut_sequences`, but keeps the two self-side chains (e.g.
    heavy/light) SEPARATE instead of concatenating them -- needed for the
    hierarchical heavy<->light cross-attention model, which needs each
    chain's own token boundaries preserved before fusing them.

    Returns (wt_chain_a, wt_chain_b, mut_chain_a, mut_chain_b, partner_sequence).
    `chain_a` is whichever chain the example's first mutation site sits on
    (e.g. heavy); `chain_b` is the teammate chain (e.g. light), or "" if this
    example has no paired chain at all (single-chain mutation) -- in that
    case `chain_b`'s wt/mut strings are both "".
    """
    sites = example.sites
    first = sites[0]
    chain_a, chain_b = first.self_sequence, first.teammate_sequence

    a_chars = list(chain_a)
    b_chars = list(chain_b) if chain_b else []
    for site in sites:
        if site.self_sequence == chain_a:
            a_chars[site.seq_index] = site.mut_aa
        elif chain_b and site.self_sequence == chain_b:
            b_chars[site.seq_index] = site.mut_aa

    return chain_a, chain_b, "".join(a_chars), "".join(b_chars), first.partner_sequence
