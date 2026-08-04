import torch

from simba.models.simba import SimbaModel

HIDDEN = 32


def _make_model(chain_mode):
    return SimbaModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6,
        mlp_hidden=16, pair_hidden=8, aa_embed_dim=4, window=3,
        core="static", chain_mode=chain_mode,
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


def test_all_chain_modes_run_and_return_finite_scalar():
    for mode in ("dual_memory", "concat", "self_attn", "none", "pairformer"):
        model = _make_model(mode)
        out = model([_site()])
        assert out.dim() == 0
        assert torch.isfinite(out), mode


def test_all_chain_modes_handle_empty_context():
    for mode in ("dual_memory", "concat", "self_attn", "none", "pairformer"):
        model = _make_model(mode)
        out = model([_site(partner_len=0, teammate_len=0)])
        assert torch.isfinite(out), mode


def test_pairformer_mode_has_pairwise_block_not_role_embed():
    model = _make_model("pairformer")
    assert hasattr(model, "combined_memory")
    assert hasattr(model, "pairwise_block")
    assert not hasattr(model, "role_embed")
    assert not hasattr(model, "interface_memory")


def test_none_mode_has_no_memory_modules_at_all():
    model = _make_model("none")
    assert not hasattr(model, "interface_memory")
    assert not hasattr(model, "combined_memory")
    assert not hasattr(model, "role_embed")


def test_self_attn_mode_has_role_embedding_and_fusion_attn():
    model = _make_model("self_attn")
    assert hasattr(model, "role_embed")
    assert hasattr(model, "fusion_attn")
    assert not hasattr(model, "interface_memory")


def test_concat_mode_has_single_combined_memory():
    model = _make_model("concat")
    assert hasattr(model, "combined_memory")
    assert not hasattr(model, "interface_memory")
    assert not hasattr(model, "role_embed")


def test_dual_memory_mode_unchanged_structure():
    model = _make_model("dual_memory")
    assert hasattr(model, "interface_memory")
    assert hasattr(model, "chain_memory")
    assert not hasattr(model, "combined_memory")


def test_training_step_reduces_loss_for_each_chain_mode():
    for mode in ("dual_memory", "concat", "self_attn", "none", "pairformer"):
        torch.manual_seed(0)
        model = _make_model(mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        sites = [_site(seq_index=4), _site(seq_index=10)]
        target = torch.tensor(2.0)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pred = model(sites)
            loss = (pred - target) ** 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], mode
