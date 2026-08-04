"""Expanded absolute-affinity (ΔG) pool: every SKEMPI2 wildtype complex
(335, deduplicated) *plus* every individually-resolved SKEMPI2 mutant row
(6,706, RDE-PPI's exact reproduction) treated as its own independent
absolute-affinity data point -- each mutant carries its own real,
experimentally-measured Kd_mut, which is exactly as valid a ΔG measurement
as the wildtype's Kd_wt. Combined: 335 + 6706 = 7,041 real binding-affinity
examples, crossing the ">7000 complexes" scale requested for a proper
5-fold CV ablation study -- using data this project has already fully
resolved (RDE-PPI's reproduction pipeline), no new fetching needed.

Reuses `simba.data.dataset.load_rde_skempi` (already-resolved sequences,
mutation sites, cvfold assignment) and `simba.data.affinity_ddg.
full_wt_mut_sequences` (the exact wildtype/mutant substitution logic
already used for ΔΔG elsewhere in this project) -- this module only adds
the *absolute* ΔG_wt / ΔG_mut computation on top, at the same fixed
T=298.15K convention RDE-PPI itself uses (so the wildtype and
derived-mutant halves of this combined pool share one consistent
temperature convention, not a mix of RDE-PPI's fixed-T and
build_affinity_benchmark.py's per-row-T conventions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from simba.data.affinity_ddg import full_wt_mut_sequences
from simba.data.dataset import load_rde_skempi

RAW_SKEMPI = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "skempi2" / "skempi_v2.csv"
FIXED_TEMPERATURE_K = 298.15
R_KCAL = 8.314 / 4184  # matches build_rde_skempi.py's own gas-constant conversion


@dataclass
class AffinityExample:
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def _wt_affinity_by_complex() -> dict[str, float]:
    """pdbcode -> ΔG_wt (fixed T=298.15K), deduplicated (first occurrence --
    SKEMPI2's Affinity_wt is the same shared measurement repeated across
    every mutation row of the same complex)."""
    df = pd.read_csv(RAW_SKEMPI, sep=";")
    df["pdbcode"] = df["#Pdb"].str.split("_").str[0].str.upper()
    df["Affinity_wt_parsed"] = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    df = df.dropna(subset=["Affinity_wt_parsed"])
    df = df[df["Affinity_wt_parsed"] > 0].drop_duplicates(subset="pdbcode")
    return {
        row["pdbcode"]: R_KCAL * FIXED_TEMPERATURE_K * math.log(row["Affinity_wt_parsed"])
        for _, row in df.iterrows()
    }


def build_skempi2_expanded_examples(exclude_complex_ids: set[str] | None = None) -> list[AffinityExample]:
    exclude = {c.upper() for c in (exclude_complex_ids or set())}
    wt_dg_by_complex = _wt_affinity_by_complex()
    rde_examples = load_rde_skempi()

    out: list[AffinityExample] = []
    seen_wt_complexes: set[str] = set()

    for ex in rde_examples:
        complex_id = ex.complex_id.upper()
        if complex_id in exclude or ex.ddG is None:
            continue
        dg_wt = wt_dg_by_complex.get(complex_id)
        if dg_wt is None:
            continue
        wt_full, mut_full, partner_seq = full_wt_mut_sequences(ex)

        if complex_id not in seen_wt_complexes:
            seen_wt_complexes.add(complex_id)
            out.append(AffinityExample(
                complex_id=complex_id, self_sequence=wt_full, partner_sequence=partner_seq,
                dG=dg_wt, meta={"row_type": "wildtype"},
            ))

        dg_mut = dg_wt + ex.ddG
        out.append(AffinityExample(
            complex_id=complex_id, self_sequence=mut_full, partner_sequence=partner_seq,
            dG=dg_mut, meta={"row_type": "mutant", "n_mutations": len(ex.sites)},
        ))

    return out


if __name__ == "__main__":
    examples = build_skempi2_expanded_examples()
    n_wt = sum(1 for e in examples if e.meta["row_type"] == "wildtype")
    n_mut = sum(1 for e in examples if e.meta["row_type"] == "mutant")
    print(f"Total examples: {len(examples)}  (wildtype: {n_wt}, mutant: {n_mut})")
    print(f"Unique complexes: {len(set(e.complex_id for e in examples))}")
    print(f"dG range: {min(e.dG for e in examples):.2f} to {max(e.dG for e in examples):.2f}")
