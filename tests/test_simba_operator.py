import torch

from simba.models.operator import ResponsePropagationOperator, SimbaOperatorModel

HIDDEN = 32


def _make_model(n_rounds=4):
    return SimbaOperatorModel(
        plm_hidden_size=HIDDEN, d_model=16, proj_mid_dim=None, operator_hidden=8,
        coupling_kernel_size=3, n_rounds=n_rounds, decoder_hidden=8,
    )


def test_forward_returns_finite_scalar():
    model = _make_model()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(wt_emb, mut_emb, partner_emb)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_zero_when_wt_equals_mut():
    # delta = mut_proj - wt_proj = 0 everywhere when the sequences are
    # identical, and the operator/decoder are purely functions of delta (and
    # a partner summary that doesn't depend on the self chain), so ddG should
    # be a fixed function of an all-zero field regardless of round count --
    # in particular it does NOT depend on which arbitrary embedding was used
    # for "both" wt and mut, since delta collapses to zero either way.
    model = _make_model()
    model.eval()
    emb_a = torch.randn(20, HIDDEN)
    emb_b = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out_a = model(emb_a, emb_a, partner_emb)
    out_b = model(emb_b, emb_b, partner_emb)
    assert torch.isclose(out_a, out_b, atol=1e-5)


def test_equilibrium_field_equals_delta_at_init():
    # The update MLP's last layer is zero-initialized, so at true init the
    # propagation contributes nothing: R stays exactly equal to delta across
    # every round, regardless of the (randomly initialized) coupling conv.
    op = ResponsePropagationOperator(d_model=16, hidden=8, coupling_kernel_size=3, n_rounds=4)
    op.eval()
    delta = torch.randn(10, 16)
    partner_summary = torch.randn(16)
    equilibrium = op(delta, partner_summary)
    assert torch.allclose(equilibrium, delta, atol=1e-6)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    target = torch.tensor(2.5)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        pred = model(wt_emb, mut_emb, partner_emb)
        loss = (pred - target) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_forward_varies_with_mutant_once_trained():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb1 = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    target = torch.tensor(3.0)
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(wt_emb, mut_emb1, partner_emb)
        loss = (pred - target) ** 2
        loss.backward()
        optimizer.step()
    model.eval()
    mut_emb2 = torch.randn(20, HIDDEN)
    out1 = model(wt_emb, mut_emb1, partner_emb)
    out2 = model(wt_emb, mut_emb2, partner_emb)
    assert not torch.isclose(out1, out2)


def test_handles_empty_partner():
    model = _make_model()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    empty_partner = torch.zeros(0, HIDDEN)
    out = model(wt_emb, mut_emb, empty_partner)
    assert torch.isfinite(out)


def test_more_rounds_still_finite_and_runs():
    model = _make_model(n_rounds=8)
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(wt_emb, mut_emb, partner_emb)
    assert torch.isfinite(out)
