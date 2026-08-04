from pathlib import Path

import pytest

from simba.data.pdb_utils import parse_pdb_chains, parse_skempi_mapping

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SKEMPI_1CSE_MAPPING = DATA_ROOT / "skempi2" / "PDBs" / "1CSE.mapping"
ABBIND_1AK4_PDB = DATA_ROOT / "abbind" / "pdbs" / "1AK4.pdb"

pytestmark = pytest.mark.skipif(
    not SKEMPI_1CSE_MAPPING.exists() or not ABBIND_1AK4_PDB.exists(),
    reason="raw data not downloaded; run scripts/download_data.py first",
)


def test_skempi_mapping_resolves_known_offset_case():
    # 1CSE chain I: SKEMPI2's canonical numbering (used in Mutation(s)_PDB, e.g.
    # "LI45G") is offset from this redistributed PDB's own author numbering.
    # The .mapping file is the documented way to bridge the two; canonical
    # position 45 on chain I must resolve to 'L' as SKEMPI2 asserts.
    chains = parse_skempi_mapping(SKEMPI_1CSE_MAPPING)
    idx = chains["I"].index_of(45)
    assert idx is not None
    assert chains["I"].sequence[idx] == "L"


def test_abbind_pdb_parses_auth_numbering_directly():
    # AB-Bind mutations reference the deposited PDB's own author numbering
    # (no offset/mapping file involved), e.g. "D:A488G" and "D:L483T".
    chains = parse_pdb_chains(ABBIND_1AK4_PDB)
    idx_488 = chains["D"].index_of(488)
    idx_483 = chains["D"].index_of(483)
    assert chains["D"].sequence[idx_488] == "A"
    assert chains["D"].sequence[idx_483] == "L"


def test_parse_pdb_chains_skips_alternate_altlocs():
    chains = parse_pdb_chains(ABBIND_1AK4_PDB)
    # exactly one residue per (resnum, icode) despite any disordered/altloc atoms
    seen = set()
    for r in chains["D"].residues:
        key = (r.resnum, r.icode)
        assert key not in seen
        seen.add(key)
