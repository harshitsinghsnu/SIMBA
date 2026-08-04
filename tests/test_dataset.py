import json

import pandas as pd

from simba.data.dataset import build_examples


def _row(chain, group_a="H,L", group_b="A"):
    return {
        "dataset": "test",
        "complex_id": "1XYZ",
        "group_a_chains": group_a,
        "group_b_chains": group_b,
        "mutations": json.dumps([
            {"chain": chain, "wt_aa": "A", "mut_aa": "G", "seq_index": 3}
        ]),
        "ddG": 1.0,
    }


SEQS = {"1XYZ": {"H": "HEAVYCHAINSEQ", "L": "LIGHTCHAINSEQ", "A": "ANTIGENSEQXYZ"}}


def test_heavy_chain_mutation_gets_light_chain_as_teammate():
    df = pd.DataFrame([_row("H")])
    examples = build_examples(df, SEQS)
    site = examples[0].sites[0]
    assert site.self_sequence == "HEAVYCHAINSEQ"
    assert site.partner_sequence == "ANTIGENSEQXYZ"
    assert site.teammate_sequence == "LIGHTCHAINSEQ"


def test_light_chain_mutation_gets_heavy_chain_as_teammate():
    df = pd.DataFrame([_row("L")])
    examples = build_examples(df, SEQS)
    site = examples[0].sites[0]
    assert site.self_sequence == "LIGHTCHAINSEQ"
    assert site.partner_sequence == "ANTIGENSEQXYZ"
    assert site.teammate_sequence == "HEAVYCHAINSEQ"


def test_antigen_mutation_has_no_teammate_when_alone_in_its_group():
    df = pd.DataFrame([_row("A")])
    examples = build_examples(df, SEQS)
    site = examples[0].sites[0]
    assert site.self_sequence == "ANTIGENSEQXYZ"
    assert site.partner_sequence in ("HEAVYCHAINSEQLIGHTCHAINSEQ", "LIGHTCHAINSEQHEAVYCHAINSEQ")
    assert site.teammate_sequence == ""


def test_single_chain_group_has_no_teammate():
    """General PPI (single chain per side) should never get a spurious
    teammate -- this must match prior single-partner behavior exactly."""
    df = pd.DataFrame([_row("H", group_a="H", group_b="A")])
    examples = build_examples(df, SEQS)
    site = examples[0].sites[0]
    assert site.teammate_sequence == ""
    assert site.partner_sequence == "ANTIGENSEQXYZ"
