import torch

from simba.models.simba import SimbaModel

HIDDEN = 32


def _make_model():
    # This file specifically exercises the dual_memory chain-handling design
    # (separate interface/chain gated memories) -- see test_simba_chain_modes.py
    # for the other modes, and note "concat" is the empirically-better default
    # used elsewhere (Phase B ablation, see the project report).
    return SimbaModel(
        plm_hidden_size=HIDDEN, d_model=16, n_heads=2, n_memory=6,
        mlp_hidden=16, pair_hidden=8, aa_embed_dim=4, window=3,
        chain_mode="dual_memory",
    )


def _site(seq_len=20, seq_index=5, partner_len=15, teammate_len=12, delta_logp=0.3,
          wt_aa="A", mut_aa="G", device="cpu"):
    def seq(n):
        return torch.randn(n, HIDDEN, device=device) if n else torch.zeros(0, HIDDEN, device=device)
    return {
        "self_embedding": seq(seq_len),
        "mut_embedding": seq(seq_len),
        "partner_embedding": seq(partner_len),
        "teammate_embedding": seq(teammate_len),
        "seq_index": seq_index,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
        "delta_logp": delta_logp,
    }


def test_forward_returns_scalar_single_site():
    model = _make_model()
    out = model([_site()])
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_forward_handles_no_partner_or_teammate_sequence():
    model = _make_model()
    out = model([_site(partner_len=0, teammate_len=0)])
    assert torch.isfinite(out)


def test_forward_handles_varying_context_lengths_same_memory_size():
    """The interface/chain memory bottlenecks should make the model
    indifferent to raw partner/teammate sequence length (always compressed
    to n_memory tokens)."""
    model = _make_model()
    out_short = model([_site(partner_len=5, teammate_len=3)])
    out_long = model([_site(partner_len=200, teammate_len=150)])
    assert torch.isfinite(out_short) and torch.isfinite(out_long)


def test_gates_start_at_zero_so_memory_reads_start_at_zero():
    """Flamingo-style gating: at initialization tanh(0)=0, so neither memory
    contributes anything regardless of its content -- verifies the
    stabilization property directly rather than just trusting tanh(0)==0."""
    model = _make_model()
    model.eval()
    query = torch.randn(3, model.d_model)
    memory = torch.randn(6, model.d_model)
    assert torch.allclose(model.interface_memory.read(query, memory), torch.zeros(model.d_model))
    assert torch.allclose(model.chain_memory.read(query, memory), torch.zeros(model.d_model))


def test_teammate_context_affects_prediction_once_gate_is_nonzero():
    """After the chain-memory gate opens up, two otherwise-identical sites
    differing only in their teammate sequence must produce different h_i --
    i.e. heavy/light (or any same-side co-chain) context is actually load
    -bearing, not silently ignored."""
    model = _make_model()
    with torch.no_grad():
        model.chain_memory.gate.fill_(1.0)
    torch.manual_seed(0)
    site_a = _site(teammate_len=10)
    site_b = dict(site_a)
    site_b["teammate_embedding"] = torch.randn(10, HIDDEN)  # different teammate content

    out_a = model([site_a])
    out_b = model([site_b])
    assert out_a.item() != out_b.item()


def test_single_site_has_no_interaction_term():
    """With exactly one mutated site there are no pairs, so the pair_mlp must
    contribute nothing -- output should equal background + independent term."""
    model = _make_model()
    model.eval()  # disable dropout for a deterministic comparison
    site = _site()
    with torch.no_grad():
        interface_mem = model.interface_memory.write(model.proj(site["partner_embedding"]))
        chain_mem = model.chain_memory.write(model.proj(site["teammate_embedding"]))
        h = model._site_feature_static(site["self_embedding"], site["mut_embedding"], (interface_mem, chain_mem),
                                        site["seq_index"], site["delta_logp"])
        expected = site["delta_logp"] * model.background_scale.item() + model.independent_head(h).item()
    out = model([site])
    assert abs(out.item() - expected) < 1e-4


def test_two_site_example_includes_nonzero_interaction_term():
    """Sanity check the epistasis pathway actually fires: total for a 2-site
    example should differ from the sum of each site scored in isolation
    (i.e. the pairwise term is doing something, not silently zero)."""
    model = _make_model()
    model.eval()  # disable dropout so the comparison isn't confounded by independent random masks
    torch.manual_seed(0)
    site_a = _site(seq_index=4, delta_logp=0.2)
    site_b = _site(seq_index=10, delta_logp=-0.1)

    out_joint = model([site_a, site_b])
    out_a_alone = model([site_a])
    out_b_alone = model([site_b])

    # background + independent heads are additive by construction; only the
    # pairwise term can make the joint prediction differ from naive summation
    assert out_joint.item() != (out_a_alone + out_b_alone).item()


def test_backward_produces_gradients_including_pair_mlp_and_gates():
    model = _make_model()
    sites = [_site(seq_index=4), _site(seq_index=10)]
    out = model(sites)
    loss = (out - torch.tensor(1.5)) ** 2
    loss.backward()
    pair_grads = [p.grad for p in model.pair_mlp.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in pair_grads)
    assert model.interface_memory.gate.grad is not None
    assert model.chain_memory.gate.grad is not None


def test_training_step_reduces_loss_on_single_example():
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
