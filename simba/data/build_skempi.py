"""Build a cleaned, resolved SKEMPI2 mutation table + per-complex sequence cache.

Output:
  data/processed/skempi2_mutations.csv   -- one row per mutation record
  data/processed/skempi2_sequences.json  -- {complex_id: {chain: {"sequence": str,
                                             "resnums": [[resnum, icode], ...]}}}
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from simba.data.mutations import parse_skempi_mutation_string
from simba.data.pdb_utils import parse_skempi_mapping

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "skempi2"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
R_KCAL = 1.9872036e-3  # kcal / (mol . K)


def parse_temperature(raw: str | float) -> float:
    if isinstance(raw, float) and math.isnan(raw):
        return 298.0
    match = re.search(r"\d+", str(raw))
    return float(match.group()) if match else 298.0


def load_chain_sequences(pdb_id: str) -> dict[str, dict]:
    mapping_path = RAW_DIR / "PDBs" / f"{pdb_id}.mapping"
    chains = parse_skempi_mapping(mapping_path)
    return {
        cid: {"sequence": cs.sequence, "resnums": [[r.resnum, r.icode] for r in cs.residues]}
        for cid, cs in chains.items()
    }


def main() -> None:
    df = pd.read_csv(RAW_DIR / "skempi_v2.csv", sep=";")

    seq_cache: dict[str, dict[str, dict]] = {}
    rows = []
    n_ddg_missing = 0
    n_unresolved = 0

    for _, row in df.iterrows():
        pdb_full = row["#Pdb"]
        parts = pdb_full.split("_")
        if len(parts) != 3:
            continue
        pdb_id, group_a, group_b = parts

        aff_mut = row.get("Affinity_mut_parsed")
        aff_wt = row.get("Affinity_wt_parsed")
        if pd.isna(aff_mut) or pd.isna(aff_wt) or aff_mut <= 0 or aff_wt <= 0:
            n_ddg_missing += 1
            continue
        temp_k = parse_temperature(row.get("Temperature"))
        ddg = R_KCAL * temp_k * (math.log(aff_mut) - math.log(aff_wt))

        if pdb_id not in seq_cache:
            try:
                seq_cache[pdb_id] = load_chain_sequences(pdb_id)
            except FileNotFoundError:
                seq_cache[pdb_id] = {}
        chains = seq_cache[pdb_id]

        try:
            sites = parse_skempi_mutation_string(row["Mutation(s)_PDB"])
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
                "dataset": "skempi2",
                "complex_id": pdb_id,
                "group_a_chains": ",".join(group_a),
                "group_b_chains": ",".join(group_b),
                "mutations": json.dumps(mutation_records),
                "n_mutations": len(mutation_records),
                "ddG": ddg,
                "resolved": resolved,
                "hold_out_type": row.get("Hold_out_type"),
            }
        )

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "skempi2_mutations.csv", index=False)
    with open(PROC_DIR / "skempi2_sequences.json", "w") as f:
        json.dump(seq_cache, f)

    n_resolved = int(out_df["resolved"].sum()) if len(out_df) else 0
    print(f"total rows in raw csv:        {len(df)}")
    print(f"dropped (missing affinity):   {n_ddg_missing}")
    print(f"dropped (unparseable mutation): {n_unresolved}")
    print(f"kept rows:                    {len(out_df)}")
    print(f"  of which fully resolved:    {n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%)")
    print(f"unique complexes:             {out_df['complex_id'].nunique() if len(out_df) else 0}")


if __name__ == "__main__":
    main()
