"""Build a cleaned, resolved AB-Bind mutation table + per-complex sequence cache.

Output:
  data/processed/abbind_mutations.csv
  data/processed/abbind_sequences.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simba.data.mutations import parse_abbind_mutation_string
from simba.data.pdb_utils import parse_pdb_chains

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "abbind"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_chain_sequences(pdb_id: str) -> dict[str, dict]:
    pdb_path = RAW_DIR / "pdbs" / f"{pdb_id}.pdb"
    chains = parse_pdb_chains(pdb_path)
    return {
        cid: {"sequence": cs.sequence, "resnums": [[r.resnum, r.icode] for r in cs.residues]}
        for cid, cs in chains.items()
    }


def main() -> None:
    df = pd.read_csv(RAW_DIR / "AB-Bind_experimental_data.csv", encoding="latin1")

    seq_cache: dict[str, dict[str, dict]] = {}
    rows = []
    n_ddg_missing = 0
    n_unresolved = 0

    for _, row in df.iterrows():
        pdb_id = str(row["#PDB"]).strip()
        partners = str(row["Partners(A_B)"]).strip()
        if "_" not in partners:
            continue
        group_a, group_b = partners.split("_", 1)

        ddg = row.get("ddG(kcal/mol)")
        if pd.isna(ddg):
            n_ddg_missing += 1
            continue

        if pdb_id not in seq_cache:
            try:
                seq_cache[pdb_id] = load_chain_sequences(pdb_id)
            except FileNotFoundError:
                seq_cache[pdb_id] = {}
        chains = seq_cache[pdb_id]

        try:
            sites = parse_abbind_mutation_string(row["Mutation"])
        except ValueError:
            n_unresolved += 1
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
        if not resolved:
            n_unresolved += 1

        rows.append(
            {
                "dataset": "abbind",
                "complex_id": pdb_id,
                "group_a_chains": ",".join(group_a),
                "group_b_chains": ",".join(group_b),
                "mutations": json.dumps(mutation_records),
                "n_mutations": len(mutation_records),
                "ddG": float(ddg),
                "resolved": resolved,
            }
        )

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "abbind_mutations.csv", index=False)
    with open(PROC_DIR / "abbind_sequences.json", "w") as f:
        json.dump(seq_cache, f)

    n_resolved = int(out_df["resolved"].sum()) if len(out_df) else 0
    print(f"total rows in raw csv:        {len(df)}")
    print(f"dropped (missing ddG):        {n_ddg_missing}")
    print(f"dropped (unparseable mutation): {n_unresolved}")
    print(f"kept rows:                    {len(out_df)}")
    print(f"  of which fully resolved:    {n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%)")
    print(f"unique complexes:             {out_df['complex_id'].nunique() if len(out_df) else 0}")


if __name__ == "__main__":
    main()
