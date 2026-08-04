"""ATLAS (Borrman et al. 2017, atlas.ibbr.umd.edu) -- a TCR:peptide-MHC
binding-affinity database, one of the five datasets the user asked to
combine from the Frontiers in Bioinformatics survey table (Table 1,
10.3389/fbinf.2022.1065703). Full table (`ATLAS.tsv`, downloaded from
atlas.ibbr.umd.edu/web/tables/ATLAS.tsv, license-free, no registration) has
1,001 rows including CDR point-mutants; this script uses only the **pure
wildtype** rows (`TCR_mut`/`MHC_mut`/`PEP_mut` all "WT", a real solved
`true_PDB`) -- 221 rows / ~197 unique structures -- because ATLAS itself
carries a precomputed real `DeltaG_kcal_per_mol` per row that needs no
recomputation, and restricting to unperturbed wildtype structures avoids
having to reconstruct CDR-mutant sequences from a template PDB (the mutant
rows only give the mutated CDR loop's sequence, not the full mutant chain --
a separate, larger effort out of scope here).

Chain grouping (self=TCR alpha+beta, partner=peptide+MHC[+beta2m]) uses
RCSB's polymer-entity description text, not naive "first entity vs rest"
(build_ppi_affinity_test2.py's convention) -- TCR-pMHC entries have 4-5
protein entities and a positional split would arbitrarily cut the TCR itself
across self/partner. Peptide entities (<=15 residues) always go to partner.
Classifying the *remaining* entities by matching "t cell receptor" text
turned out unreliable (checked against RCSB directly: many entries name TCR
entities by clone, e.g. "CF34 alpha chain", "C25 beta", "Ani2.3 TCR A
chain" -- no shared keyword at all), so this instead matches the MHC/
beta-2-microglobulin family (HLA/MHC/histocompatibility/beta-2-
microglobulin nomenclature is standardized across PDB, unlike TCR clone
names) -> partner, and defaults every other non-peptide entity -> self
(TCR). Verified against 4 previously-failing entries (3FFC, 3MV7, 4H1L,
5D2N) before adopting.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from simba.data.build_ppi_affinity_test2 import _fetch_json, fetch_structure

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "atlas"
PDB_DIR = RAW_DIR / "PDBs"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

BACKSLASH_N = "\\N"
PEPTIDE_MAX_LEN = 15
MHC_KEYWORDS = ("hla", "mhc", "histocompatibility", "beta-2-microglobulin",
                "beta 2 microglobulin", "beta2-microglobulin", "b2-microglobulin")


@dataclass
class AffinityExample:
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def _is_real_pdb_id(pdb_id: str) -> bool:
    return len(pdb_id) == 4 and pdb_id[0].isdigit() and pdb_id[1:].isalnum()


def load_wt_rows() -> list[dict]:
    with open(RAW_DIR / "ATLAS.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        if r["TCR_mut"].strip() != "WT" or r["MHC_mut"].strip() != "WT":
            continue
        if r["PEP_mut"].strip() not in ("WT", ""):
            continue
        pdb_id = r["true_PDB"].strip().upper()
        if not _is_real_pdb_id(pdb_id):
            continue
        dg_raw = r["DeltaG_kcal_per_mol"].strip()
        try:
            float(dg_raw)
        except ValueError:
            continue
        out.append(r)
    return out


def entity_description(pdb_id: str, entity_num: str) -> str:
    data = _fetch_json(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_num}")
    if data is None:
        return ""
    desc = data.get("rcsb_polymer_entity", {}).get("pdbx_description", "")
    return (desc or "").lower()


def classify_chains(pdb_id: str, seqs_by_chain: dict[str, str],
                     groups: list[list[str]]) -> tuple[str, str] | None:
    """Returns (self_sequence, partner_sequence) or None if classification
    fails (e.g. no TCR-looking entity found)."""
    self_chains: list[str] = []
    partner_chains: list[str] = []

    for entity_num, chains in enumerate(groups, start=1):
        # `chains` may hold >1 chain ID per entity when the asymmetric unit
        # has multiple identical copies (RCSB's FASTA groups them under one
        # header, e.g. "Chains A, F" for a 2-copy complex) -- keep only one
        # representative chain per entity, or self/partner sequences would
        # be duplicated per extra copy (real cause of a CUDA OOM: doubled
        # sequence length blows up ESM2-650M's attention memory).
        rep_chain = chains[0]
        seq = seqs_by_chain.get(rep_chain, "")
        if len(seq) <= PEPTIDE_MAX_LEN:
            partner_chains.append(rep_chain)
            continue
        desc = entity_description(pdb_id, str(entity_num))
        time.sleep(0.15)
        if any(kw in desc for kw in MHC_KEYWORDS):
            partner_chains.append(rep_chain)
        else:
            self_chains.append(rep_chain)

    if not self_chains or not partner_chains:
        return None
    self_seq = "".join(seqs_by_chain[c] for c in self_chains)
    partner_seq = "".join(seqs_by_chain[c] for c in partner_chains)
    if not self_seq or not partner_seq:
        return None
    return self_seq, partner_seq


def build_atlas_wt_examples() -> list[AffinityExample]:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_wt_rows()

    by_pdb: dict[str, dict] = {}
    for r in rows:
        by_pdb.setdefault(r["true_PDB"].strip().upper(), r)  # first occurrence per PDB

    examples: list[AffinityExample] = []
    n_failed = 0
    for i, (pdb_id, row) in enumerate(sorted(by_pdb.items())):
        print(f"[{i+1}/{len(by_pdb)}] {pdb_id}", flush=True)
        cache_path = PDB_DIR / f"{pdb_id}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get("skip"):
                n_failed += 1
                continue
            self_seq, partner_seq = cached["self_seq"], cached["partner_seq"]
        else:
            result = fetch_structure(pdb_id)
            if result is None:
                cache_path.write_text(json.dumps({"skip": True}))
                n_failed += 1
                print(f"  FAILED to fetch {pdb_id}", flush=True)
                continue
            seqs_by_chain, groups = result
            classified = classify_chains(pdb_id, seqs_by_chain, groups)
            if classified is None:
                cache_path.write_text(json.dumps({"skip": True}))
                n_failed += 1
                print(f"  SKIP {pdb_id}: chain classification failed", flush=True)
                continue
            self_seq, partner_seq = classified
            cache_path.write_text(json.dumps({"self_seq": self_seq, "partner_seq": partner_seq}))
            time.sleep(0.2)

        dg = float(row["DeltaG_kcal_per_mol"])
        temp_k = float(row["Temperature_K"]) if row["Temperature_K"].strip() not in ("", BACKSLASH_N) else 298.15
        examples.append(AffinityExample(
            complex_id=pdb_id, self_sequence=self_seq, partner_sequence=partner_seq, dG=dg,
            meta={"source": "atlas_wt", "temperature_k": temp_k, "tcr": row["TCRname"], "mhc": row["MHCname"]},
        ))

    print(f"\nTotal wt-unique PDBs: {len(by_pdb)}  Resolved: {len(examples)}  Failed: {n_failed}")
    return examples


if __name__ == "__main__":
    examples = build_atlas_wt_examples()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "atlas_wt.json"
    with open(out_path, "w") as f:
        json.dump([
            {"complex_id": e.complex_id, "self_sequence": e.self_sequence,
             "partner_sequence": e.partner_sequence, "dG": e.dG, "meta": e.meta}
            for e in examples
        ], f, indent=2)
    print(f"Saved {len(examples)} examples to {out_path}")
    if examples:
        print("dG range:", min(e.dG for e in examples), max(e.dG for e in examples))
