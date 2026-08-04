import torch

from simba.models.simba import HeavyLightCrossAttention

D_MODEL = 16


def _make():
    return HeavyLightCrossAttention(d_model=D_MODEL, n_heads=2)


def test_returns_concatenated_shape():
    block = _make()
    heavy = torch.randn(5, D_MODEL)
    light = torch.randn(4, D_MODEL)
    out = block(heavy, light)
    assert out.shape == (9, D_MODEL)


def test_empty_light_passes_heavy_through_unchanged():
    block = _make()
    heavy = torch.randn(5, D_MODEL)
    light = torch.zeros(0, D_MODEL)
    out = block(heavy, light)
    assert torch.equal(out, heavy)


def test_empty_heavy_passes_light_through_unchanged():
    block = _make()
    heavy = torch.zeros(0, D_MODEL)
    light = torch.randn(4, D_MODEL)
    out = block(heavy, light)
    assert torch.equal(out, light)


def test_gates_zero_init_means_no_cross_chain_contribution():
    # At init (gates = 0), each chain's output half should equal its own
    # input unchanged -- cross-attention contributes exactly zero.
    block = _make()
    block.eval()
    heavy = torch.randn(5, D_MODEL)
    light = torch.randn(4, D_MODEL)
    out = block(heavy, light)
    assert torch.allclose(out[:5], heavy, atol=1e-6)
    assert torch.allclose(out[5:], light, atol=1e-6)


def test_output_varies_once_gates_open():
    block = _make()
    block.eval()
    block.gate_heavy.data.fill_(1.0)
    block.gate_light.data.fill_(1.0)
    heavy = torch.randn(5, D_MODEL)
    light1 = torch.randn(4, D_MODEL)
    light2 = torch.randn(4, D_MODEL)
    out1 = block(heavy, light1)
    out2 = block(heavy, light2)
    assert not torch.allclose(out1[:5], out2[:5])
