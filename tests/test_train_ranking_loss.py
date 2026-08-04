"""Unit tests for train.py's within-complex pairwise ranking loss. Uses a
fake FeatureProvider (no real ESM backbone) so this stays fast and offline,
consistent with how the rest of the test suite avoids downloading models."""

import torch

from simba.data.dataset import Example, MutationSite
from simba.train import run_training

HIDDEN = 16


class FakeProvider:
    """Returns fixed-shape random-but-deterministic site dicts, standing in
    for FeatureProvider.featurize without needing a real PLM backbone."""

    def featurize(self, ex: Example):
        sites = []
        for i, site in enumerate(ex.sites):
            g = torch.Generator().manual_seed(hash((ex.complex_id, i)) % (2**31))

            def seq(n, g=g):
                return torch.randn(n, HIDDEN, generator=g) if n else torch.zeros(0, HIDDEN)

            sites.append({
                "self_embedding": seq(20), "mut_embedding": seq(20),
                "partner_embedding": seq(10), "teammate_embedding": seq(0),
                "seq_index": 5, "wt_aa": site.wt_aa, "mut_aa": site.mut_aa,
                "delta_logp": 0.1,
            })
        return sites


def _example(complex_id, ddg, idx):
    site = MutationSite(self_sequence="A" * 20, partner_sequence="A" * 10,
                         teammate_sequence="", seq_index=5, wt_aa="A", mut_aa="G")
    return Example(dataset="fake", complex_id=complex_id, sites=[site], ddG=ddg,
                   meta={"idx": idx})


def _dataset(n_complexes=3, per_complex=4):
    examples = []
    for c in range(n_complexes):
        for i in range(per_complex):
            examples.append(_example(f"complex_{c}", ddg=float(c + i * 0.5), idx=i))
    return examples


def test_run_training_with_ranking_loss_weight_zero_runs_fine():
    examples = _dataset()
    provider = FakeProvider()
    model, best_score, test_report, _ = run_training(
        examples[:8], examples[8:10], examples[10:],
        provider=provider, plm_hidden_size=HIDDEN, epochs=1, batch_size=4,
        device="cpu", verbose=False, model_kwargs={"chain_mode": "concat", "d_model": 8,
                                                      "n_heads": 2, "n_memory": 4, "mlp_hidden": 8,
                                                      "pair_hidden": 4, "aa_embed_dim": 4, "window": 3},
        ranking_loss_weight=0.0,
    )
    assert test_report is not None


def test_run_training_with_ranking_loss_weight_positive_runs_fine():
    examples = _dataset()
    provider = FakeProvider()
    model, best_score, test_report, _ = run_training(
        examples[:8], examples[8:10], examples[10:],
        provider=provider, plm_hidden_size=HIDDEN, epochs=1, batch_size=4,
        device="cpu", verbose=False, model_kwargs={"chain_mode": "concat", "d_model": 8,
                                                      "n_heads": 2, "n_memory": 4, "mlp_hidden": 8,
                                                      "pair_hidden": 4, "aa_embed_dim": 4, "window": 3},
        ranking_loss_weight=0.5,
    )
    assert test_report is not None
