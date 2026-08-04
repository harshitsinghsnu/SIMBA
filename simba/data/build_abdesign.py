"""Ingest the AbDesign paper's released benchmark file (`datasets_mut.csv` /
`datasets_wt.csv`, manually downloaded from naturalantibody.com/ab-design/).

Unlike SKEMPI2/AB-Bind this file already ships full wildtype/mutant heavy,
light and antigen sequences plus a simple 1-based sequential mutation
position directly into the relevant chain sequence -- no PDB/mapping-file
resolution is needed.

The file bundles three sub-benchmarks the AbDesign paper evaluated together:
  - "AbDesign" (658 rows): their own new single-point CDR-H3 mutants,
    measured as an ELISA mut/wt signal ratio (affinity_type='elisa_mut_to_wt_ratio').
    NOT a thermodynamic ddG -- kept as a separate `elisa_ratio` column and
    used for zero-shot evaluation only (matching the paper's own protocol:
    they never trained on this data, only ran pretrained baselines against it).
  - "SKEMPIv2" (377 rows) and "AB-Bind" (268 rows): a curated antibody-antigen
    subset of those datasets that the AbDesign authors reused for their
    apples-to-apples comparison, already in ddG (kcal/mol), same sign
    convention as our own SKEMPI2/AB-Bind processing (positive = destabilizing).

Output:
  data/processed/abdesign_mutations.csv
  data/processed/abdesign_sequences.json  -- {pdb_name: {chain: sequence}}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "abdesign"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

ROLE_COLUMNS = {
    "heavy": ("orig_heavy_chain_id", "heavy_sequence"),
    "light": ("orig_light_chain_id", "light_sequence"),
    "antigen": ("orig_antigen_chain_id", "antigen_sequence"),
}


_MUTATION_CHAIN_LABEL_TO_ROLE = {"H": "heavy", "L": "light"}

# Standard IMGT CDR boundaries (same numbering scheme for heavy and light
# chains): https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
IMGT_CDR_RANGES = [(27, 38), (56, 65), (105, 117)]  # CDR1, CDR2, CDR3


def is_cdr_position(imgt_pos) -> bool | None:
    if pd.isna(imgt_pos):
        return None
    # IMGT numbering appends an insertion-code letter for CDR-loop insertions
    # (e.g. "112A"); strip it before parsing the integer position.
    digits = "".join(ch for ch in str(imgt_pos) if ch.isdigit())
    if not digits:
        return None
    imgt_pos = int(digits)
    return any(lo <= imgt_pos <= hi for lo, hi in IMGT_CDR_RANGES)


def chain_role(mut_row, wt_row) -> str | None:
    # `mutation_chain_id` is a generic antibody role label ("H"/"L"), not the
    # actual PDB chain letter (that's `orig_heavy_chain_id`/`orig_light_chain_id`).
    # All three sub-benchmarks here only contain antibody-side mutations.
    return _MUTATION_CHAIN_LABEL_TO_ROLE.get(str(mut_row["mutation_chain_id"]))


def main() -> None:
    mut_df = pd.read_csv(RAW_DIR / "datasets_mut.csv")
    wt_df = pd.read_csv(RAW_DIR / "datasets_wt.csv")
    wt_by_key = {(r["dataset"], r["pdb_name"]): r for _, r in wt_df.iterrows()}

    seq_cache: dict[str, dict[str, str]] = {}
    rows = []
    n_no_wt = 0
    n_unresolved_role = 0

    for _, row in mut_df.iterrows():
        key = (row["dataset"], row["pdb_name"])
        wt_row = wt_by_key.get(key)
        if wt_row is None:
            n_no_wt += 1
            continue

        complex_id = row["pdb_name"]
        if complex_id not in seq_cache:
            seq_cache[complex_id] = {
                str(wt_row[chain_col]): wt_row[seq_col]
                for role, (chain_col, seq_col) in ROLE_COLUMNS.items()
                if pd.notna(wt_row.get(chain_col)) and pd.notna(wt_row.get(seq_col))
            }

        role = chain_role(row, wt_row)
        if role is None:
            n_unresolved_role += 1
            resolved = False
            seq_index = None
            actual_chain_id = row["mutation_chain_id"]
        else:
            chain_col, seq_col = ROLE_COLUMNS[role]
            actual_chain_id = wt_row[chain_col]
            wt_seq = wt_row[seq_col]
            mut_seq = row[seq_col]
            pos = int(row["mutation_pos"])
            seq_index = pos - 1
            resolved = (
                isinstance(wt_seq, str) and isinstance(mut_seq, str)
                and 0 <= seq_index < len(wt_seq)
                and wt_seq[seq_index] == row["orig_residue"]
                and mut_seq[seq_index] == row["new_residue"]
            )

        antibody_chains = [
            str(wt_row[c]) for c in ("orig_heavy_chain_id", "orig_light_chain_id")
            if pd.notna(wt_row.get(c))
        ]
        antigen_chains = (
            [str(wt_row["orig_antigen_chain_id"])]
            if pd.notna(wt_row.get("orig_antigen_chain_id")) else []
        )

        is_ddg = row["affinity_type"] == "ddg"
        rows.append(
            {
                "dataset": "abdesign",
                "source": row["dataset"],  # AbDesign | SKEMPIv2 | AB-Bind (as curated by AbDesign authors)
                "complex_id": complex_id,
                "group_a_chains": ",".join(antibody_chains),
                "group_b_chains": ",".join(antigen_chains),
                "mutations": json.dumps([
                    {"chain": str(actual_chain_id), "wt_aa": row["orig_residue"],
                     "resnum": int(row["mutation_pos"]), "icode": "",
                     "mut_aa": row["new_residue"], "seq_index": seq_index}
                ]),
                "n_mutations": 1,
                "ddG": float(row["affinity"]) if is_ddg else None,
                "elisa_ratio": float(row["affinity"]) if not is_ddg else None,
                "label_type": "ddg" if is_ddg else "elisa_ratio",
                "assay_method": row.get("method"),
                "resolved": bool(resolved),
                "is_cdr": is_cdr_position(row.get("mutation_pos_imgt")),
            }
        )

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "abdesign_mutations.csv", index=False)
    with open(PROC_DIR / "abdesign_sequences.json", "w") as f:
        json.dump(seq_cache, f)

    n_resolved = int(out_df["resolved"].sum()) if len(out_df) else 0
    print(f"total rows in raw csv:          {len(mut_df)}")
    print(f"dropped (no matching wt row):   {n_no_wt}")
    print(f"unresolved chain role:          {n_unresolved_role}")
    print(f"kept rows:                      {len(out_df)}")
    print(f"  of which fully resolved:      {n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%)")
    print(f"unique complexes:               {out_df['complex_id'].nunique() if len(out_df) else 0}")
    print("\nby source/label_type:\n", out_df.groupby(["source", "label_type"]).size())


if __name__ == "__main__":
    main()
