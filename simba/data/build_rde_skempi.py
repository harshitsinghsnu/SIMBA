"""Reproduce RDE-PPI's (Luo et al., ICLR 2023 -- the structural backbone
DSSA-PPI builds on) exact SKEMPI2 preprocessing and 3-fold complex-level CV
split, so SIMBA's numbers on this split are directly comparable to
RDE-PPI/DSSA-PPI-reported results rather than an approximation of them.

Reverse-engineered from their public `rde/datasets/skempi.py`
(github.com/luost26/RDE-PPI):
  - ddG computed from Affinity_wt_parsed/Affinity_mut_parsed using a FIXED
    T=298.15K for every row (ignoring SKEMPI2's per-row Temperature column) --
    deliberately replicated here even though our own build_skempi.py uses the
    real per-row temperature, precisely so the ddG values match theirs exactly.
  - Uses `Mutation(s)_cleaned` (not `Mutation(s)_PDB`), which -- unlike the PDB
    column -- numbers positions to match this redistributed PDB's own author
    numbering directly (no insertion codes, no SKEMPI2 .mapping-file
    indirection needed); resolved via direct PDB parsing (parse_pdb_chains).
  - block_list = {'1KBH'}.
  - "complex" unit = the full `#Pdb` string (e.g. '1CSE_E_I'), not just the
    bare PDB code.
  - 3-fold split: `complex_list = sorted(complex_to_entries.keys())`, then
    `random.Random(2022).shuffle(complex_list)`, divided into 3 contiguous
    folds of `ceil(n/3)` each -- fully deterministic given the same complex
    set, so this reproduces their exact fold assignment.

Output: data/processed/rde_skempi_mutations.csv with a `cvfold` column
(0/1/2, RDE-PPI's convention: fold i is validation, the other two are train).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

from simba.data.mutations import parse_skempi_mutation_string
from simba.data.pdb_utils import parse_pdb_chains

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "skempi2"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

BLOCK_LIST = {"1KBH"}
SPLIT_SEED = 2022
NUM_CVFOLDS = 3
FIXED_TEMPERATURE_K = 298.15
R_KCAL = 8.314 / 4184  # RDE-PPI's own gas-constant-in-kcal conversion factor


def load_chain_sequences(pdb_id: str) -> dict[str, dict]:
    pdb_path = RAW_DIR / "PDBs" / f"{pdb_id}.pdb"
    chains = parse_pdb_chains(pdb_path)
    return {
        cid: {"sequence": cs.sequence, "resnums": [[r.resnum, r.icode] for r in cs.residues]}
        for cid, cs in chains.items()
    }


def main() -> None:
    df = pd.read_csv(RAW_DIR / "skempi_v2.csv", sep=";")

    seq_cache: dict[str, dict[str, dict]] = {}
    rows = []
    complex_to_row_indices: dict[str, list[int]] = {}
    n_blocked = 0
    n_ddg_missing = 0
    n_unparseable = 0

    for _, row in df.iterrows():
        pdb_full = row["#Pdb"]
        parts = pdb_full.split("_")
        if len(parts) != 3:
            continue
        pdb_id, group_a, group_b = parts
        if pdb_id in BLOCK_LIST:
            n_blocked += 1
            continue

        aff_mut = row.get("Affinity_mut_parsed")
        aff_wt = row.get("Affinity_wt_parsed")
        if pd.isna(aff_mut) or pd.isna(aff_wt) or aff_mut <= 0 or aff_wt <= 0:
            n_ddg_missing += 1
            continue
        ddg = R_KCAL * FIXED_TEMPERATURE_K * (math.log(aff_mut) - math.log(aff_wt))

        if pdb_id not in seq_cache:
            try:
                seq_cache[pdb_id] = load_chain_sequences(pdb_id)
            except FileNotFoundError:
                seq_cache[pdb_id] = {}
        chains = seq_cache[pdb_id]

        try:
            sites = parse_skempi_mutation_string(row["Mutation(s)_cleaned"])
        except ValueError:
            n_unparseable += 1
            continue

        resolved = True
        mutation_records = []
        for site in sites:
            cs = chains.get(site.chain)
            idx = None
            if cs is not None:
                site_icode = (site.icode or " ").upper()
                for i, (resnum, icode) in enumerate(cs["resnums"]):
                    if resnum == site.resnum and icode.upper() == site_icode:
                        idx = i
                        break
            if idx is None or cs["sequence"][idx] != site.wt_aa:
                resolved = False
            mutation_records.append(
                {"chain": site.chain, "wt_aa": site.wt_aa, "resnum": site.resnum,
                 "icode": site.icode, "mut_aa": site.mut_aa, "seq_index": idx}
            )

        row_idx = len(rows)
        rows.append(
            {
                "dataset": "rde_skempi", "complex": pdb_full, "pdbcode": pdb_id,
                "group_a_chains": ",".join(group_a), "group_b_chains": ",".join(group_b),
                "mutations": json.dumps(mutation_records), "n_mutations": len(mutation_records),
                "ddG": ddg, "resolved": resolved,
                "interface_location": row.get("iMutation_Location(s)", None),
            }
        )
        complex_to_row_indices.setdefault(pdb_full, []).append(row_idx)

    out_df = pd.DataFrame(rows)

    complex_list = sorted(complex_to_row_indices.keys())
    random.Random(SPLIT_SEED).shuffle(complex_list)
    split_size = math.ceil(len(complex_list) / NUM_CVFOLDS)
    complex_splits = [
        complex_list[i * split_size:(i + 1) * split_size] for i in range(NUM_CVFOLDS)
    ]

    cvfold = pd.Series(-1, index=out_df.index)
    for fold_idx, fold_complexes in enumerate(complex_splits):
        for cplx in fold_complexes:
            cvfold.loc[complex_to_row_indices[cplx]] = fold_idx
    out_df["cvfold"] = cvfold

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "rde_skempi_mutations.csv", index=False)
    with open(PROC_DIR / "rde_skempi_sequences.json", "w") as f:
        json.dump(seq_cache, f)

    n_resolved = int(out_df["resolved"].sum())
    print(f"total rows in raw csv:      {len(df)}")
    print(f"blocked (1KBH):             {n_blocked}")
    print(f"dropped (missing affinity): {n_ddg_missing}")
    print(f"dropped (unparseable):      {n_unparseable}")
    print(f"kept rows:                  {len(out_df)}")
    print(f"  of which resolved:        {n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%)")
    print(f"unique complexes (#Pdb):    {len(complex_list)}")
    print(f"fold sizes (complexes):     {[len(f) for f in complex_splits]}")
    print("\ncvfold row counts:\n", out_df["cvfold"].value_counts())


if __name__ == "__main__":
    main()
