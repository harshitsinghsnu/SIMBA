import torch

from simba.models.simba import Epinet, SimbaModel

HIDDEN = 32


def _make_model(use_epinet=True, epinet_index_dim=4, epinet_hidden=8):
    return SimbaModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6,
        mlp_hidden=16, pair_hidden=8, aa_embed_dim=4, window=3,
        core="static", chain_mode="concat",
        use_epinet=use_epinet, epinet_index_dim=epinet_index_dim, epinet_hidden=epinet_hidden,
    )


def _site(seq_len=20, seq_index=5, partner_len=15, teammate_len=12, delta_logp=0.3):
    def seq(n):
        return torch.randn(n, HIDDEN) if n else torch.zeros(0, HIDDEN)
    return {
        "self_embedding": seq(seq_len), "mut_embedding": seq(seq_len),
        "partner_embedding": seq(partner_len),
        "teammate_embedding": seq(teammate_len), "seq_index": seq_index,
        "wt_aa": "A", "mut_aa": "G", "delta_logp": delta_logp,
    }


def test_epinet_module_has_frozen_prior():
    epinet = Epinet(feature_dim=16, index_dim=4, hidden=8)
    for p in epinet.prior.parameters():
        assert not p.requires_grad
    for p in epinet.trainable.parameters():
        assert p.requires_grad


def test_epinet_forward_finite_and_index_dependent():
    epinet = Epinet(feature_dim=16, index_dim=4, hidden=8)
    feature = torch.randn(16)
    z1 = torch.randn(4)
    z2 = torch.randn(4)
    out1 = epinet(feature, z1)
    out2 = epinet(feature, z2)
    assert out1.dim() == 0
    assert torch.isfinite(out1) and torch.isfinite(out2)
    assert not torch.isclose(out1, out2)


def test_model_with_epinet_runs_and_varies_across_z():
    model = _make_model(use_epinet=True)
    model.eval()
    sites = [_site()]
    z1 = torch.randn(4)
    z2 = torch.randn(4)
    out1 = model(sites, epinet_z=z1)
    out2 = model(sites, epinet_z=z2)
    assert torch.isfinite(out1) and torch.isfinite(out2)
    assert not torch.isclose(out1, out2)


def test_predict_with_uncertainty_returns_mean_and_positive_std():
    model = _make_model(use_epinet=True)
    model.eval()
    mean, std = model.predict_with_uncertainty([_site()], n_samples=25)
    assert torch.isfinite(mean)
    assert std.item() >= 0.0


def test_predict_with_uncertainty_requires_epinet():
    model = _make_model(use_epinet=False)
    try:
        model.predict_with_uncertainty([_site()])
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_model_without_epinet_has_no_epinet_module():
    model = _make_model(use_epinet=False)
    assert not hasattr(model, "epinet")
