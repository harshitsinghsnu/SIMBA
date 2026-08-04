import shutil

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"


def _model_cached() -> bool:
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
    except Exception:
        return False
    return any(MODEL_NAME in repo.repo_id for repo in cache.repos)


pytestmark = pytest.mark.skipif(
    not _model_cached(),
    reason=f"{MODEL_NAME} not cached locally; would require a network download",
)


@pytest.fixture(scope="module")
def scorer():
    from simba.features.plm import ESMScorer
    return ESMScorer(model_name=MODEL_NAME, device="cpu")


def test_score_shapes(scorer):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"
    result = scorer.score(seq)
    assert result.embedding.shape == (len(seq), scorer.model.config.hidden_size)
    assert result.logprobs.shape == (len(seq), 20)


def test_logprobs_are_valid_log_probabilities(scorer):
    import numpy as np
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"
    result = scorer.score(seq)
    assert np.all(result.logprobs <= 0.0)


def test_cache_round_trip(tmp_path, scorer):
    from simba.features.cache import SequenceScoreCache
    cache = SequenceScoreCache(scorer, cache_dir=tmp_path)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"

    first = cache.get(seq)
    cache_file = cache._path(seq)
    assert cache_file.exists()

    mtime_before = cache_file.stat().st_mtime
    second = cache.get(seq)  # should hit the cache, not recompute
    assert cache_file.stat().st_mtime == mtime_before
    assert (first.logprobs == second.logprobs).all()

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_masked_marginal_shapes_and_validity(scorer):
    import numpy as np
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"
    positions = [0, 5, 20, len(seq) - 1]
    out = scorer.masked_marginal_logprobs(seq, positions)
    assert out.shape == (len(positions), 20)
    assert np.all(out <= 0.0)


def test_masked_marginal_differs_from_wildtype_marginal(scorer):
    """Sanity check the two scoring modes are actually different computations
    (masking really does hide the wildtype identity from the model)."""
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"
    wildtype_marginal = scorer.score(seq).logprobs[10]
    masked_marginal = scorer.masked_marginal_logprobs(seq, [10])[0]
    assert not (wildtype_marginal == masked_marginal).all()


def test_masked_marginal_empty_positions(scorer):
    out = scorer.masked_marginal_logprobs("MKTAYIAKQ", [])
    assert out.shape == (0, 20)


def test_masked_marginal_cache_round_trip(tmp_path, scorer):
    from simba.features.cache import MaskedMarginalCache
    cache = MaskedMarginalCache(scorer, cache_dir=tmp_path)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"

    cache.ensure(seq, {3, 7})
    v3_first = cache.get(seq, 3)
    assert v3_first is not None and v3_first.shape == (20,)

    # a fresh cache instance pointed at the same dir should hit disk, not recompute
    cache2 = MaskedMarginalCache(scorer, cache_dir=tmp_path)
    v3_second = cache2.get(seq, 3)
    assert (v3_first == v3_second).all()
    assert cache2.get(seq, 999) is None  # never requested -- not present

    shutil.rmtree(tmp_path, ignore_errors=True)
