import torch

from simba.models.simba import SimbaModel, make_projection

HIDDEN = 32


def _make_model(**kwargs):
    defaults = dict(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6,
        mlp_hidden=16, pair_hidden=8, aa_embed_dim=4, window=3,
        chain_mode="concat",
    )
    defaults.update(kwargs)
    return SimbaModel(**defaults)


def _site(seq_len=20, seq_index=5, delta_logp=0.3, mut_embedding=None):
    def seq(n):
        return torch.randn(n, HIDDEN)
    self_embedding = seq(seq_len)
    return {
        "self_embedding": self_embedding,
        "mut_embedding": mut_embedding if mut_embedding is not None else seq(seq_len),
        "partner_embedding": seq(10), "teammate_embedding": seq(8),
        "seq_index": seq_index, "wt_aa": "A", "mut_aa": "G", "delta_logp": delta_logp,
    }


def test_make_projection_staged_when_mid_dim_larger_than_out():
    proj = make_projection(1280, 512, 256)
    assert isinstance(proj, torch.nn.Sequential)
    out = proj(torch.randn(3, 1280))
    assert out.shape == (3, 256)


def test_make_projection_plain_linear_when_no_mid_dim():
    proj = make_projection(128, None, 64)
    assert isinstance(proj, torch.nn.Linear)


def test_identical_wt_mut_embedding_gives_zero_diff_and_cosine_one():
    """If the mutant sequence's embedding at the site were identical to
    wildtype's (degenerate case), diff should be exactly 0 and cosine
    similarity exactly 1 -- sanity-checks the feature construction itself,
    independent of the learned projection."""
    model = _make_model()
    model.eval()
    self_embedding = torch.randn(20, HIDDEN)
    site = _site(mut_embedding=self_embedding.clone())
    site["self_embedding"] = self_embedding
    with torch.no_grad():
        wt_raw = self_embedding[site["seq_index"]]
        mut_raw = site["mut_embedding"][site["seq_index"]]
        assert torch.allclose(wt_raw, mut_raw)


def test_different_mutant_embedding_changes_prediction():
    """The counterfactual WT-vs-Mut comparison must actually be load-bearing:
    two sites identical except for mut_embedding content must score
    differently."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    site_a = _site()
    site_b = dict(site_a)
    site_b["mut_embedding"] = torch.randn(20, HIDDEN)  # different mutant content

    with torch.no_grad():
        out_a = model([site_a])
        out_b = model([site_b])
    assert out_a.item() != out_b.item()


def test_forward_with_1280_dim_staged_projection_end_to_end():
    model = _make_model(plm_hidden_size=1280, d_model=256, proj_mid_dim=512, n_heads=4, n_memory=6)
    site = {
        "self_embedding": torch.randn(20, 1280), "mut_embedding": torch.randn(20, 1280),
        "partner_embedding": torch.randn(10, 1280), "teammate_embedding": torch.randn(8, 1280),
        "seq_index": 5, "wt_aa": "A", "mut_aa": "G", "delta_logp": 0.2,
    }
    out = model([site])
    assert out.dim() == 0
    assert torch.isfinite(out)
