import torch

from simba.models.nfer import NeuralEnergyFunctional, NeuralEnergySolver, NFERModel

HIDDEN = 32


def _make_model(n_steps=4):
    return NFERModel(
        plm_hidden_size=HIDDEN, d_model=16, proj_mid_dim=None, solver_hidden=8,
        window=2, n_steps=n_steps, decoder_hidden=8,
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
    model = _make_model()
    model.eval()
    emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    out = model(emb, emb, partner_emb)
    assert torch.isclose(out, torch.tensor(0.0), atol=1e-5)


def test_energy_functional_is_zero_at_init():
    fn = NeuralEnergyFunctional(d_model=16, hidden=8, window=2)
    e = torch.randn(10, 16)
    energy = fn(e)
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-6)


def test_energy_functional_handles_empty_field():
    fn = NeuralEnergyFunctional(d_model=16, hidden=8, window=2)
    e = torch.zeros(0, 16)
    energy = fn(e)
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-6)


def test_solver_leaves_field_unchanged_at_init():
    # energy_fn is identically zero at init, so its gradient is zero, so the
    # relaxation should not move the field at all, regardless of n_steps.
    solver = NeuralEnergySolver(d_model=16, hidden=8, window=2, n_steps=4)
    e_init = torch.randn(10, 16)
    equilibrium = solver(e_init)
    assert torch.allclose(equilibrium, e_init, atol=1e-6)


def test_solver_handles_empty_field():
    solver = NeuralEnergySolver(d_model=16, hidden=8, window=2, n_steps=4)
    e_init = torch.zeros(0, 16)
    out = solver(e_init)
    assert out.shape == (0, 16)


def test_solver_works_under_outer_no_grad_context():
    # The evaluation loops in this project wrap inference in torch.no_grad();
    # the solver's internal torch.enable_grad() must still let it compute
    # correctly rather than erroring or silently returning garbage.
    solver = NeuralEnergySolver(d_model=16, hidden=8, window=2, n_steps=3)
    solver.eval()
    e_init = torch.randn(10, 16)
    with torch.no_grad():
        out = solver(e_init)
    assert torch.isfinite(out).all()


def test_forward_works_under_outer_no_grad_context():
    model = _make_model()
    model.eval()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    with torch.no_grad():
        out = model(wt_emb, mut_emb, partner_emb)
    assert torch.isfinite(out)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    partner_emb = torch.randn(15, HIDDEN)
    target = torch.tensor(1.8)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        pred = model(wt_emb, mut_emb, partner_emb)
        loss = (pred - target) ** 2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_handles_empty_partner():
    model = _make_model()
    wt_emb = torch.randn(20, HIDDEN)
    mut_emb = torch.randn(20, HIDDEN)
    empty_partner = torch.zeros(0, HIDDEN)
    out = model(wt_emb, mut_emb, empty_partner)
    assert torch.isfinite(out)
