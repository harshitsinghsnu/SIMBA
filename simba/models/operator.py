"""SIMBA-Operator: a genuinely different computational primitive for ddG
prediction, built around three components (per the design this ablation
implements directly, not a paraphrase of it):

  1. Counterfactual Encoder -- WT and mutant self-chain sequences each go
     through the frozen PLM and a shared staged projection; the per-residue
     DIFFERENCE FIELD `Delta = mut_proj - wt_proj` is the only signal that
     enters the operator below. No symbolic amino-acid lookup, no separate
     "site window" -- the perturbation is read directly off the frozen
     backbone's own representational shift, at every self-chain position.

  2. Response Propagation Operator -- the actual new primitive. Every
     self-chain position holds a RESPONSE state R_i (not an identity
     embedding), initialized to R_i^0 = Delta_i (the raw local perturbation
     *is* the response at t=0). For `n_rounds` steps:

         R_i^{t+1} = R_i^t + f(R_i^t, (phi * R^t)_i, Delta_i, partner_summary)

     where `phi * R^t` is a FIXED-width learned 1D convolution over the
     sequence axis (translation-invariant coupling between nearby
     positions -- the sequence-only proxy for "neighbors" this project has
     access to, since there is no 3D structure), not attention: there is no
     query, no key, no value, no content-addressed lookup anywhere in this
     module. `partner_summary` is a single pooled vector from the partner
     chain (mean-pool, not attention either), injected identically at every
     position and every round as a constant external field, not something
     that itself evolves -- a deliberate scope-down from a fully mutual
     self<->partner dynamical system, kept for tractability. `f` is one
     small MLP with SHARED weights across every position and every round
     (translation- and time-invariant, like a real local update rule).
     No Flamingo-style scalar gate anywhere in this operator (the rest of
     this project uses one everywhere; this ablation deliberately doesn't,
     per the design it's testing) -- only the update MLP's own weights,
     zero-initialized on its last layer purely as a standard training
     -stability trick, not a gate in the interaction-processing sense.

  3. Energy Decoder -- pool the equilibrium field R^{n_rounds} (mean over
     self-chain positions) and read out a scalar ddG directly with one
     small MLP. No pairwise term over mutated-site pairs, no memory
     bottleneck, no attention anywhere in the whole model.

This predicts ddG *directly* (not as a difference of two absolute-affinity
calls) -- the counterfactual difference field already IS the input, so
there's no separate "wildtype complex" scored on its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from simba.models.simba import make_projection


class ResponsePropagationOperator(nn.Module):
    def __init__(self, d_model: int, hidden: int, coupling_kernel_size: int = 7, n_rounds: int = 4):
        super().__init__()
        assert coupling_kernel_size % 2 == 1, "coupling_kernel_size must be odd (symmetric local window)"
        self.n_rounds = n_rounds
        self.coupling = nn.Conv1d(d_model, d_model, kernel_size=coupling_kernel_size,
                                   padding=coupling_kernel_size // 2)
        self.update_fn = nn.Sequential(
            nn.Linear(d_model * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.update_fn[-1].weight)
        nn.init.zeros_(self.update_fn[-1].bias)

    def forward(self, delta: torch.Tensor, partner_summary: torch.Tensor) -> torch.Tensor:
        """`delta`: [L, d_model] per-residue counterfactual difference field.
        `partner_summary`: [d_model] pooled partner-chain vector. Returns the
        equilibrium response field [L, d_model] after `n_rounds` updates."""
        L = delta.shape[0]
        if L == 0:
            return delta

        r = delta
        partner_broadcast = partner_summary.unsqueeze(0).expand(L, -1)  # [L, d_model]
        for _ in range(self.n_rounds):
            coupled = self.coupling(r.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)  # [L, d_model]
            update_input = torch.cat([r, coupled, delta, partner_broadcast], dim=-1)  # [L, 4*d_model]
            r = r + self.update_fn(update_input)
        return r


class SimbaOperatorModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        proj_mid_dim: int | None = 512,
        operator_hidden: int = 128,
        coupling_kernel_size: int = 7,
        n_rounds: int = 4,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.operator = ResponsePropagationOperator(d_model, operator_hidden, coupling_kernel_size, n_rounds)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """`wt_self_embedding`/`mut_self_embedding`: [Ls, plm_hidden] frozen
        -backbone embeddings of the wildtype/mutant self-chain sequence
        (same length, since a substitution doesn't change sequence length).
        `partner_embedding`: [Lp, plm_hidden]. Returns a scalar predicted ddG."""
        wt_proj = self.proj(wt_self_embedding)    # [Ls, d_model]
        mut_proj = self.proj(mut_self_embedding)  # [Ls, d_model]
        delta = mut_proj - wt_proj                # [Ls, d_model] -- the counterfactual difference field

        partner_proj = self.proj(partner_embedding)  # [Lp, d_model]
        partner_summary = (partner_proj.mean(dim=0) if partner_proj.shape[0] > 0
                            else torch.zeros(delta.shape[-1], device=delta.device, dtype=delta.dtype))

        equilibrium = self.operator(delta, partner_summary)  # [Ls, d_model]
        pooled = equilibrium.mean(dim=0) if equilibrium.shape[0] > 0 else equilibrium.new_zeros(equilibrium.shape[-1])
        return self.decoder(pooled).squeeze(-1)
