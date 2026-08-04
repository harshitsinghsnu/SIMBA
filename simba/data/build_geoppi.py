"""Resolve GeoPPI's own benchmark CSVs (S4169, M1707) against our already
-downloaded SKEMPI2 PDB `.mapping` files, so SIMBA can be evaluated on the
exact same mutation sets GeoPPI (and papers citing it) report numbers on.

S4169: 4169 single-point SKEMPI mutations, mutation format 'A:D8A' (colon
style, like AB-Bind). M1707: 1707 multi-point mutations reused from the
MutaBind2 benchmark, mutation format 'DA11A,MA14V,...' (SKEMPI2's own
concatenated WT+CHAIN+POS+MUT style) plus a `MutaBind2` column with that
method's own precomputed predictions -- a free baseline reference, no
re-running required. Both files' PDB ids are 100% (S4169) / 99% (M1707)
covered by the SKEMPI2_PDBs.tgz structures already in data/raw/skempi2/PDBs.

Output:
  data/processed/geoppi_s4169_mutations.csv
  data/processed/geoppi_m1707_mutations.csv
(sequences are looked up directly from data/processed/skempi2_sequences.json
plus any PDBs not already cached there, resolved fresh via the same
.mapping-file pipeline as build_skempi.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simba.data.mutations import (
    parse_concatenated_abbind_mutation_string,
    parse_skempi_mutation_string,
)
from simba.data.pdb_utils import parse_skempi_mapping

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "geoppi"
SKEMPI_PDB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "skempi2" / "PDBs"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_chain_sequences(pdb_id: str) -> dict[str, dict]:
    mapping_path = SKEMPI_PDB_DIR / f"{pdb_id}.mapping"
    chains = parse_skempi_mapping(mapping_path)
    return {
        cid: {"sequence": cs.sequence, "resnums": [[r.resnum, r.icode] for r in cs.residues]}
        for cid, cs in chains.items()
    }


WINDOW = 10  # +/- residues searched when a numbering convention mismatches


def resolve_sites(sites, chains) -> tuple[bool, list[dict]]:
    """Resolve each site's exact-numbering match first; if that fails (some
    older benchmarks like S4169 use a numbering convention that's neither this
    PDB's own author numbering nor SKEMPI2's canonical .mapping numbering),
    fall back to a +/-WINDOW residue search on the same chain and accept only
    an *unambiguous* single match on wt_aa. Anything still unresolved after
    that is left unresolved rather than guessed."""
    resolved = True
    records = []
    for site in sites:
        cs = chains.get(site.chain)
        idx = None
        used_fallback = False
        if cs is not None:
            site_icode = (site.icode or " ").upper()
            for i, (resnum, icode) in enumerate(cs["resnums"]):
                if resnum == site.resnum and icode.upper() == site_icode:
                    idx = i
                    break
            if idx is not None and cs["sequence"][idx] != site.wt_aa:
                idx = None  # exact position exists but wrong residue -- try fallback
            if idx is None:
                candidates = [
                    i for i, (resnum, _) in enumerate(cs["resnums"])
                    if abs(resnum - site.resnum) <= WINDOW and cs["sequence"][i] == site.wt_aa
                ]
                if len(candidates) == 1:
                    idx = candidates[0]
                    used_fallback = True
        site_resolved = idx is not None and cs is not None and cs["sequence"][idx] == site.wt_aa
        resolved = resolved and site_resolved
        records.append({"chain": site.chain, "wt_aa": site.wt_aa, "resnum": site.resnum,
                         "icode": site.icode, "mut_aa": site.mut_aa, "seq_index": idx,
                         "resolved_via_window_fallback": used_fallback})
    return resolved, records


def build_s4169() -> None:
    df = pd.read_csv(RAW_DIR / "S4169.csv")
    seq_cache: dict[str, dict[str, dict]] = {}
    rows = []
    n_unparseable = 0

    for _, row in df.iterrows():
        pdb_id = row["protein"]
        partners = str(row["Partners(A_B)"])
        group_a, group_b = partners.split("_", 1) if "_" in partners else (partners, "")

        if pdb_id not in seq_cache:
            try:
                seq_cache[pdb_id] = load_chain_sequences(pdb_id)
            except FileNotFoundError:
                seq_cache[pdb_id] = {}

        try:
            sites = parse_concatenated_abbind_mutation_string(str(row["mutation"]))
        except ValueError:
            n_unparseable += 1
            continue

        resolved, records = resolve_sites(sites, seq_cache[pdb_id])
        rows.append({
            "dataset": "geoppi_s4169", "complex_id": pdb_id,
            "group_a_chains": ",".join(group_a), "group_b_chains": ",".join(group_b),
            "mutations": json.dumps(records), "n_mutations": len(records),
            "ddG": float(row["DDG"]), "mode": row.get("mode"), "resolved": resolved,
        })

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "geoppi_s4169_mutations.csv", index=False)
    with open(PROC_DIR / "geoppi_s4169_sequences.json", "w") as f:
        json.dump(seq_cache, f)
    n_resolved = int(out_df["resolved"].sum())
    print(f"[S4169] rows={len(out_df)}/{len(df)} (dropped {n_unparseable} unparseable) "
          f"resolved={n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%) "
          f"complexes={out_df['complex_id'].nunique()}")


def build_m1707() -> None:
    df = pd.read_csv(RAW_DIR / "M1707.csv")
    seq_cache: dict[str, dict[str, dict]] = {}
    rows = []
    n_unparseable = 0

    for _, row in df.iterrows():
        pdb_id = row["PDB id"]
        partners = str(row["Partner1"])
        group_a, group_b = partners.split("_", 1) if "_" in partners else (partners, "")

        if pdb_id not in seq_cache:
            try:
                seq_cache[pdb_id] = load_chain_sequences(pdb_id)
            except FileNotFoundError:
                seq_cache[pdb_id] = {}

        try:
            sites = parse_skempi_mutation_string(str(row["Mutation(s)_PDB"]))
        except ValueError:
            n_unparseable += 1
            continue

        resolved, records = resolve_sites(sites, seq_cache[pdb_id])
        rows.append({
            "dataset": "geoppi_m1707", "complex_id": pdb_id,
            "group_a_chains": ",".join(group_a), "group_b_chains": ",".join(group_b),
            "mutations": json.dumps(records), "n_mutations": len(records),
            "ddG": float(row["DDGexp"]), "mode": row.get("Label"),
            "mutabind2_prediction": float(row["MutaBind2"]) if pd.notna(row.get("MutaBind2")) else None,
            "resolved": resolved,
        })

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / "geoppi_m1707_mutations.csv", index=False)
    with open(PROC_DIR / "geoppi_m1707_sequences.json", "w") as f:
        json.dump(seq_cache, f)
    n_resolved = int(out_df["resolved"].sum())
    print(f"[M1707] rows={len(out_df)}/{len(df)} (dropped {n_unparseable} unparseable) "
          f"resolved={n_resolved} ({100 * n_resolved / max(len(out_df), 1):.1f}%) "
          f"complexes={out_df['complex_id'].nunique()}")


def main() -> None:
    build_s4169()
    build_m1707()


if __name__ == "__main__":
    main()
