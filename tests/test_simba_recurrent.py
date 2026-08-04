import torch

from simba.models.simba import SimbaModel

HIDDEN = 32


def _make_model(n_rounds=2, recurrent_use_pairwise=False):
    return SimbaModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6,
        mlp_hidden=16, pair_hidden=8, aa_embed_dim=4, window=3,
        core="recurrent", n_rounds=n_rounds, recurrent_use_pairwise=recurrent_use_pairwise,
    )


def _site(seq_len=20, seq_index=5, partner_len=15, teammate_len=12, delta_logp=0.3,
          wt_aa="A", mut_aa="G"):
    def seq(n):
        return torch.randn(n, HIDDEN) if n else torch.zeros(0, HIDDEN)
    return {
        "self_embedding": seq(seq_len), "mut_embedding": seq(seq_len),
        "partner_embedding": seq(partner_len),
        "teammate_embedding": seq(teammate_len), "seq_index": seq_index,
        "wt_aa": wt_aa, "mut_aa": mut_aa, "delta_logp": delta_logp,
    }


def test_recurrent_forward_returns_scalar():
    model = _make_model()
    out = model([_site()])
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_recurrent_handles_multi_site_and_no_context():
    model = _make_model()
    out = model([_site(partner_len=0, teammate_len=0), _site(seq_index=12)])
    assert torch.isfinite(out)


def test_field_is_noop_when_gates_are_zero():
    """At initialization all three round gates are 0, so `run` must return
    site_states completely unchanged regardless of field content or rounds."""
    model = _make_model(n_rounds=3)
    model.eval()
    site_states = torch.randn(4, model.d_model)
    field = torch.randn(6, model.d_model)
    refined = model.field.run(field, site_states)
    assert torch.allclose(refined, site_states)


def test_epistasis_emerges_from_field_once_gates_open_no_explicit_pairwise():
    """With recurrent_use_pairwise=False, any interaction between sites must
    come purely from the shared field. Force the gates open and check the
    joint prediction differs from the sum of isolated predictions -- i.e.
    the field-mediated pathway alone is sufficient to create epistasis."""
    model = _make_model(n_rounds=2, recurrent_use_pairwise=False)
    model.eval()
    with torch.no_grad():
        model.field.site_write_gate.fill_(1.0)
        model.field.field_update_gate.fill_(1.0)
        model.field.site_read_gate.fill_(1.0)
    torch.manual_seed(0)
    site_a = _site(seq_index=4, delta_logp=0.2)
    site_b = _site(seq_index=10, delta_logp=-0.1)

    out_joint = model([site_a, site_b])
    out_a_alone = model([site_a])
    out_b_alone = model([site_b])
    assert out_joint.item() != (out_a_alone + out_b_alone).item()


def test_backward_produces_gradients_for_field_gates():
    model = _make_model()
    sites = [_site(seq_index=4), _site(seq_index=10)]
    out = model(sites)
    loss = (out - torch.tensor(1.5)) ** 2
    loss.backward()
    assert model.field.site_write_gate.grad is not None
    assert model.field.field_update_gate.grad is not None
    assert model.field.site_read_gate.grad is not None


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = _make_model()
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
    assert losses[-1] < losses[0]


def test_cost_scales_linearly_not_quadratically_in_num_sites():
    """Sanity check the design property: doubling the number of sites should
    not require the pairwise-count growth the static/explicit model has.
    This just checks a large site count runs without error in reasonable
    tensor shapes (functional, not a timing assertion)."""
    model = _make_model()
    many_sites = [_site(seq_index=i) for i in range(15)]
    out = model(many_sites)
    assert torch.isfinite(out)
