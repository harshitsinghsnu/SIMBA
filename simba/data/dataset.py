"""Turns a processed mutations table (+ its sequences.json) into a flat list
of Example objects the SIMBA model can consume directly.

Each mutation site is resolved independently to three sequences: `self` (the
chain the site sits on), `partner` (every chain on the *other* side of the
interface -- e.g. the antigen), and `teammate` (every *other* chain on the
*same* side -- e.g. the paired light chain when the mutation is on the heavy
chain, or vice versa). Earlier versions of this pipeline only modeled
self+partner, which meant a heavy-chain mutation never saw the light chain
(or any co-chain) at all. Resolving per-site rather than per-example handles
multi-site mutations spanning different chains without needing awkward
same-chain coordinate offsetting: every site gets its own local window +
cross-attention context, and the model sums per-site contributions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


@dataclass
class MutationSite:
    self_sequence: str
    partner_sequence: str     # concatenation of chains on the *other* side of the interface
    teammate_sequence: str    # concatenation of *other* chains on the *same* side (e.g. paired H/L chain)
    seq_index: int
    wt_aa: str
    mut_aa: str


@dataclass
class Example:
    dataset: str
    complex_id: str
    sites: list[MutationSite]
    ddG: float | None = None
    elisa_ratio: float | None = None
    meta: dict = field(default_factory=dict)


def _load_sequences(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text())
    return {
        complex_id: {
            chain: (info if isinstance(info, str) else info["sequence"])
            for chain, info in chains.items()
        }
        for complex_id, chains in raw.items()
    }


def build_examples(
    df: pd.DataFrame,
    sequences: dict[str, dict[str, str]],
    complex_id_col: str = "complex_id",
    dataset_col: str = "dataset",
) -> list[Example]:
    examples = []
    for _, row in df.iterrows():
        complex_id = row[complex_id_col]
        complex_seqs = sequences.get(complex_id, {})
        group_a = str(row.get("group_a_chains", "")).split(",") if pd.notna(row.get("group_a_chains")) else []
        group_b = str(row.get("group_b_chains", "")).split(",") if pd.notna(row.get("group_b_chains")) else []

        mutations = json.loads(row["mutations"])
        sites = []
        ok = True
        for m in mutations:
            if m.get("seq_index") is None:
                ok = False
                break
            chain = m["chain"]
            self_seq = complex_seqs.get(chain)
            if not self_seq:
                ok = False
                break
            own_group = group_a if chain in group_a else (group_b if chain in group_b else [])
            partner_group = group_b if chain in group_a else (group_a if chain in group_b else [])
            partner_seq = "".join(complex_seqs.get(c, "") for c in partner_group)
            teammate_seq = "".join(complex_seqs.get(c, "") for c in own_group if c != chain)
            sites.append(MutationSite(
                self_sequence=self_seq, partner_sequence=partner_seq, teammate_sequence=teammate_seq,
                seq_index=m["seq_index"], wt_aa=m["wt_aa"], mut_aa=m["mut_aa"],
            ))
        if not ok or not sites:
            continue

        ddg = row.get("ddG")
        ddg = float(ddg) if pd.notna(ddg) else None
        elisa = row.get("elisa_ratio")
        elisa = float(elisa) if pd.notna(elisa) else None
        if ddg is None and elisa is None:
            continue

        meta = {k: row[k] for k in row.index if k not in
                {"mutations", complex_id_col, dataset_col, "group_a_chains", "group_b_chains"}}
        # Raw (chain, resnum, icode) per site -- excluded from the flat meta
        # dict above (it's a list, not a scalar column) but needed for any
        # PDB-coordinate-based analysis (B-factor, distance-to-interface,
        # §3.13) that a plain `seq_index` (self_sequence-relative, already on
        # each MutationSite) can't drive on its own. Chain groupings kept
        # alongside for the same reason (which chains form the "partner"
        # side, needed to compute distance-to-partner-chain).
        meta["sites_raw"] = mutations
        meta["group_a_chains"] = group_a
        meta["group_b_chains"] = group_b
        examples.append(Example(
            dataset=row.get(dataset_col, "unknown"), complex_id=complex_id,
            sites=sites, ddG=ddg, elisa_ratio=elisa,
            meta=meta,
        ))
    return examples


def load_all_mutations() -> list[Example]:
    df = pd.read_csv(PROC_DIR / "all_mutations.csv")
    seqs: dict[str, dict[str, str]] = {}
    for name in ("skempi2", "abbind", "abdesign"):
        path = PROC_DIR / f"{name}_sequences.json"
        if path.exists():
            seqs.update(_load_sequences(path))
    return build_examples(df, seqs)


def load_rde_skempi() -> list[Example]:
    df = pd.read_csv(PROC_DIR / "rde_skempi_mutations.csv")
    seqs = _load_sequences(PROC_DIR / "rde_skempi_sequences.json")
    return build_examples(df, seqs, complex_id_col="pdbcode")


def load_mint_skempi() -> list[Example]:
    """Same rows/complexes/sequences as `load_rde_skempi`, but with a
    cvfold assignment built using MINT's own SkempiDataset split_seed
    (2023) instead of RDE-PPI's (2022) -- see
    simba/data/build_mint_skempi_split.py."""
    df = pd.read_csv(PROC_DIR / "mint_skempi_mutations.csv")
    seqs = _load_sequences(PROC_DIR / "rde_skempi_sequences.json")
    return build_examples(df, seqs, complex_id_col="pdbcode")


def load_protattba(name: str) -> list[Example]:
    """`name` in {"ab645", "ab1101", "s1131"}. Each Example's `meta` carries
    ProtAttBA's own `fold` assignment and `protattba_prediction` for a direct
    row-level comparison against their real predictions."""
    df = pd.read_csv(PROC_DIR / f"protattba_{name}_mutations.csv")
    seqs = _load_sequences(PROC_DIR / f"protattba_{name}_sequences.json")
    return build_examples(df, seqs)


def load_geoppi_m1707() -> list[Example]:
    """GeoPPI's multi-point mutation benchmark (general PPI, reused from
    MutaBind2). Each Example's `meta` carries `mutabind2_prediction` for a
    direct row-level comparison against that method's real predictions, and
    `mode` (forward/reverse). No official fold assignment is released, so
    eval_geoppi.py builds its own leave-complex-out folds."""
    df = pd.read_csv(PROC_DIR / "geoppi_m1707_mutations.csv")
    seqs = _load_sequences(PROC_DIR / "geoppi_m1707_sequences.json")
    return build_examples(df, seqs)


def load_geoppi_s4169() -> list[Example]:
    """GeoPPI's single-point mutation benchmark (general PPI). Only 58%
    resolved (see build_geoppi.py docstring -- legacy SKEMPI-1.0 numbering);
    `resolved`-filtering already happens inside build_examples via seq_index."""
    df = pd.read_csv(PROC_DIR / "geoppi_s4169_mutations.csv")
    seqs = _load_sequences(PROC_DIR / "geoppi_s4169_sequences.json")
    return build_examples(df, seqs)
