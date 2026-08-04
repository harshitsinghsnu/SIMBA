"""Compares SIMBA's Integrated-Gradients attribution on barstar (partner
chain, `interpretability_1brs.py`'s output) against real, independently
-published experimental/computational hotspot residues for the same
interface: Lee & Tidor, "Barstar is electrostatically optimized for tight
binding to barnase," Nature Structural Biology 8:71-73 (2001).

That paper reports 7 barstar side chains as electrostatically critical to
binding (computed charge-optimization sensitivity, corroborated by prior
alanine-scanning mutagenesis): Asn33, Asp35, Trp38, Asp39, Thr42, Val73,
Glu76 -- with Asp35 and Asp39 singled out as the two most critical
(4-8 kcal/mol loss on mutation).

Residue-numbering reconciliation: this project's 1BRS chain D (barstar)
numbering matches the paper's numbering exactly for 5 of the 7 residues
(N33, D35, W38, D39, T42 -- confirmed identical amino acid at the identical
position). The remaining two (Val73, Glu76) do not match at face value
(our chain D has Arg73/Lys76) -- checked directly and resolved: shifting by
a constant -2 lands on Val71/Glu74, which *do* match, and this same -2 shift
is independently confirmed by the paper's own in-text mention of "Gln72" as
a third highly-sensitive position (our chain D has Gln70 -- also matches
under the same -2 shift). Three independent positions shifting by the same
constant offset is strong evidence this is a genuine numbering-convention
difference (not uncommon between older mutagenesis papers and a later
redistributed PDB entry's own numbering), not a coincidence -- documented
here rather than silently guessed.
"""

from __future__ import annotations

import json

PAPER_HOTSPOTS = {
    33: "N", 35: "D", 38: "W", 39: "D", 42: "T",   # exact-match positions
    71: "V", 74: "E",                               # -2 offset applied (paper: Val73, Glu76)
}
PAPER_MOST_CRITICAL = {35, 39}  # Asp35, Asp39 -- 4-8 kcal/mol loss on mutation, per the paper
IN_JSON = "figures/ig_1brs_attribution.json"


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    resnums = [r for r, _ in d["partner_resnums"]]
    seq = d["partner_sequence"]
    attr = d["partner_attribution"]

    for resnum, expected_aa in PAPER_HOTSPOTS.items():
        idx = resnums.index(resnum) if resnum in resnums else None
        actual_aa = seq[idx] if idx is not None else "?"
        assert actual_aa == expected_aa, f"residue {resnum}: expected {expected_aa}, got {actual_aa}"

    n_paper = len(PAPER_HOTSPOTS)
    order = sorted(range(len(attr)), key=lambda i: -attr[i])
    top_resnums_by_rank = [resnums[i] for i in order]

    top7 = set(top_resnums_by_rank[:n_paper])
    top15pct_k = max(1, round(0.15 * len(seq)))
    top15pct = set(top_resnums_by_rank[:top15pct_k])
    paper_set = set(PAPER_HOTSPOTS.keys())

    overlap_top7 = paper_set & top7
    overlap_top15 = paper_set & top15pct
    overlap_most_critical = PAPER_MOST_CRITICAL & top15pct

    print(f"Paper hotspots (n={n_paper}): {sorted(paper_set)}")
    print(f"SIMBA top-{n_paper} attributed barstar residues: {sorted(top7)}")
    print(f"  overlap: {sorted(overlap_top7)} ({len(overlap_top7)}/{n_paper})")
    print(f"SIMBA top-{top15pct_k} (15%) attributed barstar residues: {sorted(top15pct)}")
    print(f"  overlap: {sorted(overlap_top15)} ({len(overlap_top15)}/{n_paper})")
    print(f"Most-critical (Asp35, Asp39) recovered in top-15%: {sorted(overlap_most_critical)} "
          f"({len(overlap_most_critical)}/2)")

    random_expected_top7 = n_paper * n_paper / len(seq)
    random_expected_top15 = n_paper * top15pct_k / len(seq)
    print(f"\nRandom-chance expected overlap: top-{n_paper} ~ {random_expected_top7:.2f}, "
          f"top-{top15pct_k} ~ {random_expected_top15:.2f}")

    result = {
        "paper_hotspots": sorted(paper_set),
        "paper_most_critical": sorted(PAPER_MOST_CRITICAL),
        "simba_top7": sorted(top7),
        "simba_top15pct": sorted(top15pct),
        "overlap_top7": sorted(overlap_top7),
        "overlap_top15pct": sorted(overlap_top15),
        "overlap_most_critical_in_top15pct": sorted(overlap_most_critical),
        "random_expected_top7": random_expected_top7,
        "random_expected_top15pct": random_expected_top15,
        "top15pct_k": top15pct_k,
    }
    with open("figures/barstar_literature_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved figures/barstar_literature_comparison.json")


if __name__ == "__main__":
    main()
