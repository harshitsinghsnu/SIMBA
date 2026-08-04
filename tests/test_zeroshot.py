import numpy as np

from simba.features.plm import STANDARD_AAS, SequenceScore
from simba.features.zeroshot import mutation_delta_logp, site_delta_logp


def _fake_score(length: int = 10) -> SequenceScore:
    rng = np.random.RandomState(0)
    logprobs = rng.uniform(-5, -0.1, size=(length, len(STANDARD_AAS))).astype(np.float32)
    embedding = rng.randn(length, 8).astype(np.float16)
    return SequenceScore(embedding=embedding, logprobs=logprobs)


def test_site_delta_logp_basic():
    score = _fake_score()
    wt_col = STANDARD_AAS.index("A")
    mut_col = STANDARD_AAS.index("G")
    score.logprobs[3, wt_col] = -1.0
    score.logprobs[3, mut_col] = -2.0
    d = site_delta_logp(score, seq_index=3, wt_aa="A", mut_aa="G")
    assert d == -1.0  # log P(mut) - log P(wt) = -2.0 - (-1.0)


def test_site_delta_logp_none_for_unresolved_or_nonstandard():
    score = _fake_score()
    assert site_delta_logp(score, seq_index=None, wt_aa="A", mut_aa="G") is None
    assert site_delta_logp(score, seq_index=3, wt_aa="X", mut_aa="G") is None
    assert site_delta_logp(score, seq_index=999, wt_aa="A", mut_aa="G") is None


def test_mutation_delta_logp_is_additive_over_sites():
    score = _fake_score()
    muts = [
        {"seq_index": 1, "wt_aa": "A", "mut_aa": "G"},
        {"seq_index": 2, "wt_aa": "C", "mut_aa": "D"},
    ]
    expected = (
        site_delta_logp(score, 1, "A", "G") + site_delta_logp(score, 2, "C", "D")
    )
    assert mutation_delta_logp(score, muts) == expected


def test_mutation_delta_logp_none_if_any_site_unscoreable():
    score = _fake_score()
    muts = [
        {"seq_index": 1, "wt_aa": "A", "mut_aa": "G"},
        {"seq_index": None, "wt_aa": "C", "mut_aa": "D"},
    ]
    assert mutation_delta_logp(score, muts) is None
