import torch

from simba.models.ncpd import ContinuousDynamicsIntegrator, NCPDModel

HIDDEN = 32


def _make_model(n_steps=4):
    return NCPDModel(
        plm_hidden_size=HIDDEN, d_model=16, proj_mid_dim=None,
        mutation_op_hidden=8, dynamics_hidden=8, n_steps=n_steps, decoder_hidden=8,
    )


def test_forward_returns_finite_scalar():
    model = _make_model()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(wt_emb, mut_emb, partner_emb)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_dynamics_leaves_state_unchanged_at_init():
    # DynamicsField's last layer is zero-initialized, so dh/dt is exactly
    # zero at init regardless of delta/partner -- the trajectory shouldn't
    # move the state at all, for any n_steps.
    integrator = ContinuousDynamicsIntegrator(d_model=16, hidden=8, n_steps=4)
    h0 = torch.randn(16)
    delta = torch.randn(16)
    partner_summary = torch.randn(16)
    h_relaxed = integrator(h0, delta, partner_summary)
    assert torch.allclose(h_relaxed, h0, atol=1e-6)


def test_output_is_mutant_invariant_at_init():
    # Since the trajectory doesn't move at init, ddG_pred is a function of
    # h_wt alone -- different mutants (with the same WT/partner) should give
    # the identical prediction until training opens the dynamics up.
    model = _make_model()
    model.eval()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb1 = torch.randn(20, HIDDEN)
    mut_emb2 = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out1 = model(wt_emb, mut_emb1, partner_emb)
    out2 = model(wt_emb, mut_emb2, partner_emb)
    assert torch.isclose(out1, out2, atol=1e-5)


def test_output_varies_with_wt_at_init():
    model = _make_model()
    model.eval()
    wt_emb1 = torch.randn(20, HIDDEN)
    wt_emb2 = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out1 = model(wt_emb1, mut_emb, partner_emb)
    out2 = model(wt_emb2, mut_emb, partner_emb)
    assert not torch.isclose(out1, out2)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    target = torch.tensor(2.2)
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
    target = torch.tensor(3.5)
    for _ in range(30):
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
