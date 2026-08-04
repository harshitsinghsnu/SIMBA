"""Utilities for resolving (chain, author residue number, insertion code) -> amino
acid identity and full chain sequence from a PDB structure file.

We only ever use these coordinates to resolve *which sequence position* a mutation
notation refers to (SKEMPI2/AB-Bind both use author PDB numbering). No 3D geometry
is passed to any model -- the resulting objects are plain 1D sequences plus a
residue-number lookup table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from Bio.SeqUtils import seq1


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resnum: int
    icode: str  # " " if none


@dataclass
class ChainSequence:
    chain: str
    residues: list[ResidueKey]
    sequence: str  # 1-letter, same length/order as `residues`

    def index_of(self, resnum: int, icode: str = " ") -> int | None:
        key = ResidueKey(self.chain, resnum, icode)
        for i, r in enumerate(self.residues):
            if r == key:
                return i
        return None


def parse_pdb_chains(pdb_path: str | Path) -> dict[str, ChainSequence]:
    """Parse a PDB file into one ChainSequence per polymer chain (first model only).

    Reads fixed-width ATOM records directly instead of using Bio.PDB's structure
    builder: several real-world PDB files (e.g. some AB-Bind entries) have blank
    occupancy fields on disordered/altloc atoms that crash Bio.PDB's disordered-atom
    comparison logic. We only need the CA trace to resolve one amino acid per
    (chain, resnum, icode), so a manual line parser sidesteps that entirely.
    """
    seen: dict[str, dict[tuple[int, str], str]] = {}
    order: dict[str, list[tuple[int, str]]] = {}

    with open(pdb_path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            record = line[0:6]
            if record != "ATOM  ":
                if record == "ENDMDL":
                    break  # first model only
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue  # keep only the primary altloc conformer
            resname = line[17:20].strip()
            chain_id = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip() or " "

            key = (resnum, icode)
            chain_seen = seen.setdefault(chain_id, {})
            if key in chain_seen:
                continue  # duplicate CA for this residue (e.g. repeated altloc)
            aa = seq1(resname, custom_map={}, undef_code="X") or "X"
            chain_seen[key] = aa
            order.setdefault(chain_id, []).append(key)

    chains: dict[str, ChainSequence] = {}
    for chain_id, keys in order.items():
        residues = [ResidueKey(chain_id, resnum, icode) for resnum, icode in keys]
        sequence = "".join(seen[chain_id][k] for k in keys)
        chains[chain_id] = ChainSequence(chain_id, residues, sequence)
    return chains


def parse_pdb_bfactors(pdb_path: str | Path) -> dict[tuple[str, int, str], float]:
    """CA B-factor per (chain, resnum, icode), first model only. Used as a
    cheap, no-new-pipeline proxy for local backbone flexibility (high
    B-factor ~ more mobile/disordered in the crystal) -- an alternative to
    running MD or a dedicated flexibility predictor, consistent with this
    project's preference for reusing data already on disk over building new
    infrastructure for a single analysis. Same fixed-width-record, first
    -model-only, primary-altloc-only parsing convention as `parse_pdb_chains`.
    """
    bfactors: dict[tuple[str, int, str], float] = {}
    with open(pdb_path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            record = line[0:6]
            if record != "ATOM  ":
                if record == "ENDMDL":
                    break
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain_id = line[21]
            try:
                resnum = int(line[22:26])
                bfactor = float(line[60:66])
            except ValueError:
                continue
            icode = line[26].strip() or " "
            bfactors[(chain_id, resnum, icode)] = bfactor
    return bfactors


def parse_pdb_ca_coords(pdb_path: str | Path) -> dict[str, dict[tuple[int, str], tuple[float, float, float]]]:
    """CA (x, y, z) per chain, keyed by (resnum, icode), first model only,
    primary altloc only -- same parsing convention as `parse_pdb_chains`.
    Used only for a post-hoc geometric distance-to-interface analysis (§3.13
    figure), never as model input -- this project's model stays sequence
    -only; this is a diagnostic tool, not a feature.
    """
    coords: dict[str, dict[tuple[int, str], tuple[float, float, float]]] = {}
    with open(pdb_path, encoding="ascii", errors="replace") as fh:
        for line in fh:
            record = line[0:6]
            if record != "ATOM  ":
                if record == "ENDMDL":
                    break
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            chain_id = line[21]
            try:
                resnum = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            icode = line[26].strip() or " "
            coords.setdefault(chain_id, {})[(resnum, icode)] = (x, y, z)
    return coords


def min_distance_to_chains(
    point: tuple[float, float, float],
    coords_by_chain: dict[str, dict[tuple[int, str], tuple[float, float, float]]],
    target_chains: set[str],
) -> float | None:
    """Minimum Euclidean CA-CA distance from `point` to any residue in any
    of `target_chains`. Returns None if none of those chains have coordinates."""
    best = None
    px, py, pz = point
    for chain_id in target_chains:
        for (x, y, z) in coords_by_chain.get(chain_id, {}).values():
            d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
            if best is None or d < best:
                best = d
    return best


def parse_skempi_mapping(mapping_path: str | Path) -> dict[str, ChainSequence]:
    """Parse a SKEMPI2 `<PDB>.mapping` file.

    Each line is: RESNAME(3-letter) CHAIN CANONICAL_RESNUM AUTH_RESNUM
    where CANONICAL_RESNUM is the numbering used in SKEMPI2's
    `Mutation(s)_PDB` column, which can differ from the actual auth
    numbering in the redistributed PDB file (observed offsets in several
    classic complexes, e.g. 1CSE). Building sequences straight from this
    file sidesteps the offset entirely -- no need to cross-reference the
    .pdb file's own numbering.
    """
    resnum_re = re.compile(r"^(\d+)([A-Za-z]*)$")
    by_chain: dict[str, list[tuple[int, str, str]]] = {}
    for line in Path(mapping_path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        resname, chain, canonical_resnum, _auth_resnum = parts
        m = resnum_re.match(canonical_resnum)
        if not m:
            continue
        resnum, icode = m.groups()
        aa = seq1(resname, custom_map={}, undef_code="X")
        by_chain.setdefault(chain, []).append((int(resnum), icode.upper(), aa or "X"))

    chains: dict[str, ChainSequence] = {}
    for chain, entries in by_chain.items():
        entries.sort(key=lambda t: (t[0], t[1]))
        residues = [ResidueKey(chain, resnum, icode or " ") for resnum, icode, _ in entries]
        sequence = "".join(aa for _, _, aa in entries)
        chains[chain] = ChainSequence(chain, residues, sequence)
    return chains
