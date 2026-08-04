import pytest

from simba.data.mutations import parse_abbind_mutation_string, parse_skempi_mutation_string


def test_skempi_single_mutation():
    sites = parse_skempi_mutation_string("LI45G")
    assert len(sites) == 1
    s = sites[0]
    assert (s.wt_aa, s.chain, s.resnum, s.icode, s.mut_aa) == ("L", "I", 45, "", "G")


def test_skempi_multi_mutation():
    sites = parse_skempi_mutation_string("RI67A,RI65A")
    assert [(s.wt_aa, s.resnum, s.mut_aa) for s in sites] == [("R", 67, "A"), ("R", 65, "A")]


def test_skempi_insertion_code():
    sites = parse_skempi_mutation_string("RD100bA")
    s = sites[0]
    assert (s.wt_aa, s.chain, s.resnum, s.icode, s.mut_aa) == ("R", "D", 100, "b", "A")


def test_skempi_unparseable_raises():
    with pytest.raises(ValueError):
        parse_skempi_mutation_string("not-a-mutation")


def test_abbind_single_mutation():
    sites = parse_abbind_mutation_string("D:A488G")
    s = sites[0]
    assert (s.wt_aa, s.chain, s.resnum, s.icode, s.mut_aa) == ("A", "D", 488, "", "G")


def test_abbind_multi_mutation():
    sites = parse_abbind_mutation_string("D:L483T,D:V486P,D:H487A")
    assert [(s.chain, s.resnum, s.mut_aa) for s in sites] == [
        ("D", 483, "T"), ("D", 486, "P"), ("D", 487, "A"),
    ]


def test_abbind_unparseable_raises():
    with pytest.raises(ValueError):
        parse_abbind_mutation_string("bad-token")
