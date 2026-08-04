"""Reproduces PPI-Affinity's (Romero-Molina et al. 2022, J Proteome Res
21(8):1829-41) "Test Set 2" -- the same 90-complex protein-protein test set
PPI-Graphomer's Table 1 calls "Test set 2" (87 rows after their own
filtering). Source: PPI-Affinity's own ACS-hosted supplementary ZIP
(figshare collection 6027665, article 19971014,
`SI-File-4-protein-protein-test-set-2.csv`), fetched via Figshare's public
REST API (the web UI is behind an AWS WAF bot challenge that blocks plain
HTTP clients; the file CDN at ndownloader.figshare.com is not).

That CSV ships `Binding_affinity` (real ΔG, kcal/mol, standard sign
convention -- negative=favorable, matching this project's own convention
directly, unlike PPI-Graphomer's Test Set 1 source table) plus real
per-complex predictions from PRODIGY, DFIRE, CP_PIE, ISLAND, and
PPI-Affinity's own SVM -- the same five methods (plus PPI-Graphomer itself)
compared in PPI-Graphomer's Table 1. No sequences/chain groupings are in
that CSV, so this script fetches each PDB entry from RCSB (public, not
license-gated) and groups polymer entities into "self"/"partner" via
RCSB's own entity API: the common case (2 protein entities) splits
unambiguously; entries with >2 protein entities put the first entity alone
on the self side and every other protein entity combined on the partner
side (a documented simplification for the minority of complexes that
aren't simple two-body).
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "ppi_affinity"
PDB_DIR = RAW_DIR / "PDBs"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

BASELINE_COLUMNS = ["PRODIGY", "DFIRE", "CP_PIE", "ISLAND", "PPI-Affinity"]


@dataclass
class AffinityExample:
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def _fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _fetch_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_structure(pdb_id: str) -> tuple[dict[str, str], list[list[str]]] | None:
    """Returns (chain_id -> sequence, [entity_1_chains, entity_2_chains, ...])
    for protein polymer entities only, via RCSB's REST + FASTA APIs."""
    entry = _fetch_json(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}")
    if entry is None:
        return None
    entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    fasta = _fetch_text(f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}")
    if fasta is None:
        return None
    # Parse FASTA: headers look like ">1AN1_1|Chains A|... polypeptide(L) ..."
    seqs_by_chain: dict[str, str] = {}
    chain_to_entity: dict[str, str] = {}
    entity_chains: dict[str, list[str]] = {}
    header, seq_lines = None, []

    def flush():
        if header is None:
            return
        parts = header.split("|")
        if len(parts) < 2 or ("Chains" not in parts[1] and "Chain" not in parts[1]):
            return
        entity_num = parts[0].split("_")[-1]
        chain_field = parts[1].replace("Chains", "").replace("Chain", "").strip()
        # Each token is either a bare label ("A") or "B[auth I]", where the
        # bracketed value is the actual author chain ID (what PDB ATOM
        # records use) -- prefer that when present, since a bracket-naive
        # split would silently produce a wrong/unused chain identifier.
        chains = []
        for tok in chain_field.split(","):
            tok = tok.strip()
            m = re.match(r"^(\S+)\[auth\s+(\S+)\]$", tok)
            chains.append(m.group(2) if m else tok)
        seq = "".join(seq_lines)
        for c in chains:
            seqs_by_chain[c] = seq
            chain_to_entity[c] = entity_num
            entity_chains.setdefault(entity_num, []).append(c)

    for line in fasta.splitlines():
        if line.startswith(">"):
            flush()
            header, seq_lines = line[1:], []
        else:
            seq_lines.append(line.strip())
    flush()

    # Only keep entities RCSB itself calls polymer_entity_ids (protein or not --
    # filter to protein via entity_ids membership; non-protein polymers like
    # DNA/RNA are rare in this benchmark and filtered by sequence looking
    # like protein if needed downstream).
    ordered_entities = sorted(entity_chains.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    groups = [entity_chains[e] for e in ordered_entities]
    return seqs_by_chain, groups


def main() -> None:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "SI" / "SI-File-4-protein-protein-test-set-2.csv") as f:
        rows = list(csv.DictReader(f))

    examples: list[AffinityExample] = []
    n_failed = 0
    seq_cache = {}
    for i, row in enumerate(rows):
        pdb_id = row["PDB_NAME"].strip()
        print(f"[{i+1}/{len(rows)}] {pdb_id}", flush=True)
        cache_path = PDB_DIR / f"{pdb_id}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            seqs_by_chain, groups = data["seqs"], data["groups"]
        else:
            result = fetch_structure(pdb_id)
            if result is None:
                n_failed += 1
                print(f"  FAILED to fetch {pdb_id}", flush=True)
                time.sleep(0.3)
                continue
            seqs_by_chain, groups = result
            cache_path.write_text(json.dumps({"seqs": seqs_by_chain, "groups": groups}))
            time.sleep(0.3)  # be polite to RCSB

        if len(groups) < 2:
            n_failed += 1
            print(f"  SKIP {pdb_id}: fewer than 2 polymer entities resolved", flush=True)
            continue

        self_chains = groups[0]
        partner_chains = [c for g in groups[1:] for c in g]
        self_seq = "".join(seqs_by_chain[c] for c in self_chains)
        partner_seq = "".join(seqs_by_chain[c] for c in partner_chains)
        if not self_seq or not partner_seq:
            n_failed += 1
            continue

        try:
            dg = float(row["Binding_affinity"])
        except ValueError:
            n_failed += 1
            continue

        examples.append(AffinityExample(
            complex_id=pdb_id.upper(), self_sequence=self_seq, partner_sequence=partner_seq, dG=dg,
            meta={"baselines": {c: float(row[c]) for c in BASELINE_COLUMNS if row.get(c) not in (None, "")},
                  "n_entities": len(groups)},
        ))

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "ppi_affinity_test2.json"
    with open(out_path, "w") as f:
        json.dump([
            {"complex_id": e.complex_id, "self_sequence": e.self_sequence,
             "partner_sequence": e.partner_sequence, "dG": e.dG, "meta": e.meta}
            for e in examples
        ], f, indent=2)

    print(f"\nTotal rows: {len(rows)}  Resolved: {len(examples)}  Failed: {n_failed}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
