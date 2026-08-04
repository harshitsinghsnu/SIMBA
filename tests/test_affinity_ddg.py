from simba.data.affinity_ddg import full_wt_mut_sequences, paired_wt_mut_sequences
from simba.data.dataset import Example, MutationSite


def test_single_site_single_chain():
    site = MutationSite(self_sequence="ACDEFG", partner_sequence="PPP",
                         teammate_sequence="", seq_index=2, wt_aa="D", mut_aa="K")
    ex = Example(dataset="x", complex_id="c1", sites=[site], ddG=1.0)
    wt, mut, partner = full_wt_mut_sequences(ex)
    assert wt == "ACDEFG"
    assert mut == "ACKEFG"
    assert partner == "PPP"


def test_two_sites_same_chain():
    seq = "ACDEFGHIK"
    site1 = MutationSite(self_sequence=seq, partner_sequence="PP", teammate_sequence="",
                          seq_index=1, wt_aa="C", mut_aa="W")
    site2 = MutationSite(self_sequence=seq, partner_sequence="PP", teammate_sequence="",
                          seq_index=5, wt_aa="G", mut_aa="Y")
    ex = Example(dataset="x", complex_id="c2", sites=[site1, site2], ddG=1.0)
    wt, mut, partner = full_wt_mut_sequences(ex)
    assert wt == seq
    assert mut == "AWDEFYHIK"
    assert partner == "PP"


def test_two_sites_different_chains_heavy_light():
    heavy = "HHHHH"
    light = "LLLLL"
    # site on heavy chain: self=heavy, teammate=light
    site_h = MutationSite(self_sequence=heavy, partner_sequence="AG", teammate_sequence=light,
                           seq_index=1, wt_aa="H", mut_aa="Q")
    # site on light chain: self=light, teammate=heavy
    site_l = MutationSite(self_sequence=light, partner_sequence="AG", teammate_sequence=heavy,
                           seq_index=3, wt_aa="L", mut_aa="R")
    ex = Example(dataset="x", complex_id="c3", sites=[site_h, site_l], ddG=1.0)
    wt, mut, partner = full_wt_mut_sequences(ex)
    assert wt == heavy + light  # canonical order fixed by first site (site_h): self+teammate
    assert mut == "HQHHH" + "LLLRL"
    assert partner == "AG"


def test_no_teammate_chain():
    site = MutationSite(self_sequence="MNOP", partner_sequence="XY", teammate_sequence="",
                         seq_index=0, wt_aa="M", mut_aa="A")
    ex = Example(dataset="x", complex_id="c4", sites=[site], ddG=1.0)
    wt, mut, partner = full_wt_mut_sequences(ex)
    assert wt == "MNOP"
    assert mut == "ANOP"
    assert partner == "XY"


def test_paired_single_site_single_chain():
    site = MutationSite(self_sequence="ACDEFG", partner_sequence="PPP",
                         teammate_sequence="", seq_index=2, wt_aa="D", mut_aa="K")
    ex = Example(dataset="x", complex_id="c1", sites=[site], ddG=1.0)
    wt_a, wt_b, mut_a, mut_b, partner = paired_wt_mut_sequences(ex)
    assert wt_a == "ACDEFG" and wt_b == ""
    assert mut_a == "ACKEFG" and mut_b == ""
    assert partner == "PPP"


def test_paired_two_sites_heavy_and_light():
    heavy = "HHHHH"
    light = "LLLLL"
    site_h = MutationSite(self_sequence=heavy, partner_sequence="AG", teammate_sequence=light,
                           seq_index=1, wt_aa="H", mut_aa="Q")
    site_l = MutationSite(self_sequence=light, partner_sequence="AG", teammate_sequence=heavy,
                           seq_index=3, wt_aa="L", mut_aa="R")
    ex = Example(dataset="x", complex_id="c3", sites=[site_h, site_l], ddG=1.0)
    wt_a, wt_b, mut_a, mut_b, partner = paired_wt_mut_sequences(ex)
    assert wt_a == heavy and wt_b == light
    assert mut_a == "HQHHH"
    assert mut_b == "LLLRL"
    assert partner == "AG"


def test_paired_mutation_only_on_light_chain():
    heavy = "HHHHH"
    light = "LLLLL"
    site_l = MutationSite(self_sequence=light, partner_sequence="AG", teammate_sequence=heavy,
                           seq_index=0, wt_aa="L", mut_aa="W")
    ex = Example(dataset="x", complex_id="c5", sites=[site_l], ddG=1.0)
    wt_a, wt_b, mut_a, mut_b, partner = paired_wt_mut_sequences(ex)
    # canonical chain_a is whichever chain the FIRST site sits on -- here that's light
    assert wt_a == light and wt_b == heavy
    assert mut_a == "WLLLL"
    assert mut_b == heavy  # unmutated teammate passes through unchanged
