import torch

from simba.models.affinity import AbsoluteAffinityModel

HIDDEN = 32


def _make_model(use_pairformer=True):
    return AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=use_pairformer,
    )


def test_forward_returns_finite_scalar():
    for use_pairformer in (True, False):
        model = _make_model(use_pairformer)
        self_emb = torch.randn(20, HIDDEN)
        partner_emb = torch.randn(15, HIDDEN)
        out = model(self_emb, partner_emb)
        assert out.dim() == 0
        assert torch.isfinite(out), use_pairformer


def test_forward_varies_with_input_once_gates_are_open():
    # Every read/attention gate is Flamingo-style zero-initialized (model
    # starts equivalent to ignoring context entirely), so at true init the
    # output is input-invariant by design -- open the gates manually to test
    # that the underlying computation actually depends on the input content.
    for use_pairformer in (True, False):
        model = _make_model(use_pairformer)
        model.eval()
        model.partner_memory.gate.data.fill_(1.0)
        if use_pairformer:
            for layer in model.pairwise_block.layers:
                layer.gate_row.data.fill_(1.0)
                layer.gate_col.data.fill_(1.0)
        self_emb = torch.randn(20, HIDDEN)
        partner_emb1 = torch.randn(15, HIDDEN)
        partner_emb2 = torch.randn(15, HIDDEN)
        out1 = model(self_emb, partner_emb1)
        out2 = model(self_emb, partner_emb2)
        assert not torch.isclose(out1, out2), use_pairformer


def test_training_step_reduces_loss():
    for use_pairformer in (True, False):
        torch.manual_seed(0)
        model = _make_model(use_pairformer)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        self_emb = torch.randn(20, HIDDEN)
        partner_emb = torch.randn(15, HIDDEN)
        target = torch.tensor(-8.0)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            pred = model(self_emb, partner_emb)
            loss = (pred - target) ** 2
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], use_pairformer


def test_pairformer_variant_has_pairwise_block_not_readout_query():
    model = _make_model(use_pairformer=True)
    assert hasattr(model, "pairwise_block")
    assert not hasattr(model, "readout_query")


def test_simple_variant_has_readout_query_not_pairwise_block():
    model = _make_model(use_pairformer=False)
    assert hasattr(model, "readout_query")
    assert not hasattr(model, "pairwise_block")


def test_forward_ddg_is_difference_of_two_forward_calls():
    model = _make_model(use_pairformer=True)
    model.eval()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    ddg = model.forward_ddg(wt_emb, mut_emb, partner_emb)
    expected = model(mut_emb, partner_emb) - model(wt_emb, partner_emb)
    assert torch.isclose(ddg, expected)


def test_forward_ddg_is_zero_when_wt_equals_mut():
    model = _make_model(use_pairformer=True)
    model.eval()
    emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    ddg = model.forward_ddg(emb, emb, partner_emb)
    assert torch.isclose(ddg, torch.tensor(0.0), atol=1e-6)


def test_n_pairformer_layers_stacks_independent_layers():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=True, n_pairformer_layers=3,
    )
    assert len(model.pairwise_block.layers) == 3
    # each layer has its own parameters, not shared/tied
    ids = {id(layer) for layer in model.pairwise_block.layers}
    assert len(ids) == 3
    self_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(self_emb, partner_emb)
    assert out.dim() == 0 and torch.isfinite(out)


def test_epinet_forward_and_uncertainty():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=True, use_epinet=True, epinet_index_dim=4, epinet_hidden=8,
    )
    model.eval()
    self_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(self_emb, partner_emb)
    assert out.dim() == 0 and torch.isfinite(out)
    mean, std = model.predict_with_uncertainty(self_emb, partner_emb, n_samples=10)
    assert torch.isfinite(mean) and std.item() >= 0.0


def test_epinet_ddg_shares_index_between_wt_and_mut():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=True, use_epinet=True, epinet_index_dim=4, epinet_hidden=8,
    )
    model.eval()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    z = torch.randn(4)
    ddg = model.forward_ddg(wt_emb, mut_emb, partner_emb, epinet_z=z)
    expected = model(mut_emb, partner_emb, epinet_z=z) - model(wt_emb, partner_emb, epinet_z=z)
    assert torch.isclose(ddg, expected)
    mean, std = model.predict_ddg_with_uncertainty(wt_emb, mut_emb, partner_emb, n_samples=10)
    assert torch.isfinite(mean) and std.item() >= 0.0


def test_cosine_head_forward_in_target_range():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        use_pairformer=True, head_type="cosine", target_min=-15.0, target_max=-2.0,
    )
    model.eval()
    self_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(self_emb, partner_emb)
    assert torch.isfinite(out)
    assert -15.0 - 1e-4 <= out.item() <= -2.0 + 1e-4


def test_cosine_head_forward_cos_sim_matches_mapped_forward():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        use_pairformer=True, head_type="cosine", target_min=-15.0, target_max=-2.0,
    )
    model.eval()
    self_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    cos_sim = model.forward_cos_sim(self_emb, partner_emb)
    expected = (cos_sim + 1) / 2 * (model.target_max - model.target_min) + model.target_min
    out = model(self_emb, partner_emb)
    assert torch.isclose(out, expected)


def test_cosine_head_ddg_still_works():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        use_pairformer=True, head_type="cosine", target_min=-15.0, target_max=-2.0,
    )
    model.eval()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    ddg = model.forward_ddg(wt_emb, mut_emb, partner_emb)
    assert torch.isfinite(ddg)


def _make_hl_model():
    return AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=True, use_hl_cross_attention=True,
    )


def test_hl_model_has_cross_attn_module():
    model = _make_hl_model()
    assert hasattr(model, "hl_cross_attn")


def test_forward_paired_returns_finite_scalar():
    model = _make_hl_model()
    heavy_emb = torch.randn(12, HIDDEN)
    light_emb = torch.randn(10, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model.forward_paired(heavy_emb, light_emb, partner_emb)
    assert out.dim() == 0 and torch.isfinite(out)


def test_forward_paired_handles_empty_light_chain():
    # single-chain mutation (no paired chain) should still work: the
    # cross-attention block passes the heavy chain through unchanged.
    model = _make_hl_model()
    model.eval()
    heavy_emb = torch.randn(12, HIDDEN)
    empty_light = torch.zeros(0, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model.forward_paired(heavy_emb, empty_light, partner_emb)
    assert torch.isfinite(out)


def test_forward_paired_varies_with_light_chain_once_gates_open():
    model = _make_hl_model()
    model.eval()
    model.hl_cross_attn.gate_heavy.data.fill_(1.0)
    model.hl_cross_attn.gate_light.data.fill_(1.0)
    model.partner_memory.gate.data.fill_(1.0)
    for layer in model.pairwise_block.layers:
        layer.gate_row.data.fill_(1.0)
        layer.gate_col.data.fill_(1.0)
    heavy_emb = torch.randn(12, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    light_emb1 = torch.randn(10, HIDDEN)
    light_emb2 = torch.randn(10, HIDDEN)
    out1 = model.forward_paired(heavy_emb, light_emb1, partner_emb)
    out2 = model.forward_paired(heavy_emb, light_emb2, partner_emb)
    assert not torch.isclose(out1, out2)


def test_forward_ddg_paired_is_difference_of_two_paired_calls():
    model = _make_hl_model()
    model.eval()
    wt_heavy = torch.randn(12, HIDDEN)
    wt_light = torch.randn(10, HIDDEN)
    mut_heavy = torch.randn(12, HIDDEN)
    mut_light = torch.randn(10, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    ddg = model.forward_ddg_paired(wt_heavy, wt_light, mut_heavy, mut_light, partner_emb)
    expected = (model.forward_paired(mut_heavy, mut_light, partner_emb)
                - model.forward_paired(wt_heavy, wt_light, partner_emb))
    assert torch.isclose(ddg, expected)


def test_forward_ddg_paired_is_zero_when_wt_equals_mut():
    model = _make_hl_model()
    model.eval()
    heavy_emb = torch.randn(12, HIDDEN)
    light_emb = torch.randn(10, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    ddg = model.forward_ddg_paired(heavy_emb, light_emb, heavy_emb, light_emb, partner_emb)
    assert torch.isclose(ddg, torch.tensor(0.0), atol=1e-6)


def test_hl_epinet_ddg_shares_index():
    model = AbsoluteAffinityModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6, pair_hidden=8,
        mlp_hidden=16, use_pairformer=True, use_hl_cross_attention=True,
        use_epinet=True, epinet_index_dim=4, epinet_hidden=8,
    )
    model.eval()
    wt_heavy = torch.randn(12, HIDDEN)
    wt_light = torch.randn(10, HIDDEN)
    mut_heavy = torch.randn(12, HIDDEN)
    mut_light = torch.randn(10, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    z = torch.randn(4)
    ddg = model.forward_ddg_paired(wt_heavy, wt_light, mut_heavy, mut_light, partner_emb, epinet_z=z)
    expected = (model.forward_paired(mut_heavy, mut_light, partner_emb, epinet_z=z)
                - model.forward_paired(wt_heavy, wt_light, partner_emb, epinet_z=z))
    assert torch.isclose(ddg, expected)
