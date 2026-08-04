"""Combined absolute-affinity (dG) pool: every license-free protein-protein
binding-affinity dataset from the Frontiers survey table (Wong et al. 2022,
10.3389/fbinf.2022.1065703, Table 1) this project has been able to acquire
without a gated registration step --

  SKEMPI 2.0 (expanded: 340 wildtype complexes + 6,706 individually-resolved
    mutant rows, see build_skempi2_expanded.py)          7,046 examples
  Kastritis-lineage "Corrected Benchmark 1.0"             46 examples
    (build_affinity_benchmark.py's build_test_set(); the same benchmark
    lineage as the survey's 144-complex Kastritis et al. 2011 row -- a
    corrected/reduced 46-complex version, license-free on GitHub, no
    PDBbind gate)
  ATLAS wildtype TCR:peptide-MHC set                      ~190 examples
    (build_atlas.py; a different structural paradigm -- TCR alpha+beta vs
    peptide+MHC -- but still a real, sequence-only ΔG binding-affinity task
    fitting the same self/partner framework)

Two of the survey's five rows are deliberately excluded, not silently
dropped:
  - SKEMPI 1.1 (Moal & Fernandez-Recio 2012): SKEMPI2 is its own published
    extension/superset over the same complex lineage (more mutations per
    complex, same core PDB structures) -- no separate raw file exists, and
    every complex SKEMPI 1.1 could contribute is already reachable through
    SKEMPI2.
  - PDBbind v2020 (protein-protein subset): requires a manual PDBbind
    license/registration step this environment doesn't have (the same
    constraint already documented in build_affinity_benchmark.py for
    PPI-Graphomer's training data) -- left out rather than worked around.

Deduplicates by complex_id (uppercase) across sources so no PDB structure is
double-counted -- SKEMPI2 wins on overlap (finest per-mutation granularity),
then benchmark46, then ATLAS. This matters for eval_skempi2_5fold.py's
complex-level fold construction: a complex must resolve to exactly one
source's version, or the same PDB could straddle two folds under different
example objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from simba.data.build_affinity_benchmark import build_test_set as build_benchmark46
from simba.data.build_skempi2_expanded import build_skempi2_expanded_examples

PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ATLAS_JSON = PROC_DIR / "atlas_wt.json"


@dataclass
class AffinityExample:
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def _load_atlas_examples() -> list[AffinityExample]:
    if not ATLAS_JSON.exists():
        return []
    with open(ATLAS_JSON) as f:
        rows = json.load(f)
    return [
        AffinityExample(complex_id=r["complex_id"], self_sequence=r["self_sequence"],
                         partner_sequence=r["partner_sequence"], dG=r["dG"], meta=r["meta"])
        for r in rows
    ]


def build_combined_examples() -> list[AffinityExample]:
    skempi = [
        AffinityExample(complex_id=e.complex_id, self_sequence=e.self_sequence,
                         partner_sequence=e.partner_sequence, dG=e.dG,
                         meta={**e.meta, "dataset": "skempi2"})
        for e in build_skempi2_expanded_examples()
    ]
    bm46 = [
        AffinityExample(complex_id=e.complex_id, self_sequence=e.self_sequence,
                         partner_sequence=e.partner_sequence, dG=e.dG,
                         meta={**e.meta, "dataset": "benchmark46"})
        for e in build_benchmark46()
    ]
    atlas = [
        AffinityExample(complex_id=e.complex_id, self_sequence=e.self_sequence,
                         partner_sequence=e.partner_sequence, dG=e.dG,
                         meta={**e.meta, "dataset": "atlas"})
        for e in _load_atlas_examples()
    ]

    seen_complexes: set[str] = set()
    out: list[AffinityExample] = []
    for pool in (skempi, bm46, atlas):
        for ex in pool:
            cid = ex.complex_id.upper()
            if ex.meta["dataset"] == "skempi2":
                # SKEMPI2 rows share complex_id across many mutant rows by
                # design (per-mutation examples) -- only dedup *other*
                # sources against SKEMPI2's complex set, not within it.
                out.append(ex)
                seen_complexes.add(cid)
            else:
                if cid in seen_complexes:
                    continue
                out.append(ex)
                seen_complexes.add(cid)
    return out


if __name__ == "__main__":
    examples = build_combined_examples()
    by_dataset: dict[str, int] = {}
    for e in examples:
        by_dataset[e.meta["dataset"]] = by_dataset.get(e.meta["dataset"], 0) + 1
    print(f"Total examples: {len(examples)}")
    print(f"By dataset: {by_dataset}")
    print(f"Unique complexes: {len(set(e.complex_id.upper() for e in examples))}")
    print(f"dG range: {min(e.dG for e in examples):.2f} to {max(e.dG for e in examples):.2f}")
