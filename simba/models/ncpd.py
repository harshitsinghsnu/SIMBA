"""NCPD (Neural Counterfactual Protein Dynamics): a fourth computational
primitive, distinct from the pairformer (dense cross-attention), SIMBA
-Operator (local convolutional propagation, simba/models/operator.py), and
NFER (learned energy-functional minimization, simba/models/nfer.py). Here
the hypothesis is that a mutation induces a latent **dynamical trajectory**
from the wildtype protein state toward a mutant-influenced state, and ddG is
read off the *endpoint* of that trajectory -- not from aggregation, not from
propagation, not from energy minimization, but from **continuous-time
evolution** of a single pooled latent state:

    WT PLM -> Latent State -> Mutation Operator -> Continuous Dynamics
           -> Relaxed State -> Energy Decoder -> ddG

  1. Latent State -- WT self-chain and partner chain each go through the
     frozen backbone, staged projection, and mean-pooling to a single
     [d_model] vector each (`h_wt`, `partner_summary`). Unlike the operator
     and NFER (which keep a per-residue field), NCPD's state is a single
     pooled vector -- its novelty is about *time*, not per-residue spatial
     structure, so it's kept as simple as possible on that axis.
  2. Mutation Operator -- the mutant self-chain is pooled the same way
     (`h_mut_raw`), and a small learned MLP transforms the raw
     `h_mut_raw - h_wt` difference into a perturbation signal `Delta` that
     will drive the dynamics below (a genuine transform, not just the raw
     difference vector, giving the model room to reshape what "the
     mutation's effect" looks like before it becomes a forcing term).
  3. Continuous Dynamics -- `h` evolves from `h(0) = h_wt` under a learned
     vector field:

         dh/dt = f_theta(h, Delta, partner_summary)

     with `Delta` and `partner_summary` held constant throughout as
     external conditioning (not part of the evolving state itself),
     integrated with fixed-step forward Euler over `n_steps` steps (chosen
     over RK4 for computational simplicity -- the point being tested is
     the *existence* of a continuous-trajectory primitive, not integrator
     accuracy). `f_theta`'s last layer is zero-initialized, so at true init
     `dh/dt` is exactly zero regardless of `Delta` -- the trajectory doesn't
     move at all, and every mutation predicts the *same* ddG (a function of
     `h_wt` alone) until training opens the dynamics up. This is the same
     zero-contribution-at-init property used throughout this project,
     achieved here without any Flamingo-style gate.
  4. Energy Decoder -- one small MLP maps the relaxed endpoint state
     directly to a scalar ddG. Unlike the operator/NFER, there is no
     separate wildtype-complex score to subtract: the WT-to-mutant
     trajectory *is* the model, and its endpoint is decoded once.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from simba.models.simba import make_projection


class MutationOperator(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, d_model))

    def forward(self, raw_delta: torch.Tensor) -> torch.Tensor:
        return self.net(raw_delta)


class DynamicsField(nn.Module):
    """f_theta(h, Delta, partner_summary) = dh/dt."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model * 3, hidden), nn.ReLU(), nn.Linear(hidden, d_model))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, delta: torch.Tensor, partner_summary: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, delta, partner_summary], dim=-1))


class ContinuousDynamicsIntegrator(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_steps: int = 4):
        super().__init__()
        self.field = DynamicsField(d_model, hidden)
        self.n_steps = n_steps

    def forward(
        self, h0: torch.Tensor, delta: torch.Tensor, partner_summary: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward-Euler integration of dh/dt = f_theta(h, delta, partner)
        from t=0 (h0) to t=1, `n_steps` fixed-size steps. `return_trajectory`
        (diagnostic/visualization only, §3.14 -- not used during training)
        returns the full list [h0, h1, ..., h_{n_steps}] instead of just the
        endpoint."""
        h = h0
        dt = 1.0 / self.n_steps
        trajectory = [h] if return_trajectory else None
        for _ in range(self.n_steps):
            h = h + dt * self.field(h, delta, partner_summary)
            if return_trajectory:
                trajectory.append(h)
        return trajectory if return_trajectory else h


class NCPDModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        proj_mid_dim: int | None = 512,
        mutation_op_hidden: int = 128,
        dynamics_hidden: int = 128,
        n_steps: int = 4,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.mutation_operator = MutationOperator(d_model, mutation_op_hidden)
        self.dynamics = ContinuousDynamicsIntegrator(d_model, dynamics_hidden, n_steps)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )

    def _pooled_proj(self, embedding: torch.Tensor) -> torch.Tensor:
        proj = self.proj(embedding)  # [L, d_model], L may be 0
        if proj.shape[0] == 0:
            return proj.new_zeros(proj.shape[-1])
        return proj.mean(dim=0)

    def forward(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        h_wt = self._pooled_proj(wt_self_embedding)              # [d_model] -- "Latent State"
        h_mut_raw = self._pooled_proj(mut_self_embedding)
        partner_summary = self._pooled_proj(partner_embedding)

        raw_delta = h_mut_raw - h_wt
        delta = self.mutation_operator(raw_delta)                # "Mutation Operator"

        h_relaxed = self.dynamics(h_wt, delta, partner_summary)   # "Continuous Dynamics" -> "Relaxed State"
        return self.decoder(h_relaxed).squeeze(-1)                # "Energy Decoder" -> ddG

    def forward_trajectory(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> dict:
        """Diagnostic-only variant of `forward` (§3.14 trajectory
        visualization) -- returns every intermediate latent state h(t) for
        t=0..n_steps, plus the decoded ddG at each step (not just the
        endpoint), so the WT->mutant relaxation can be plotted step by step."""
        h_wt = self._pooled_proj(wt_self_embedding)
        h_mut_raw = self._pooled_proj(mut_self_embedding)
        partner_summary = self._pooled_proj(partner_embedding)
        raw_delta = h_mut_raw - h_wt
        delta = self.mutation_operator(raw_delta)
        trajectory = self.dynamics(h_wt, delta, partner_summary, return_trajectory=True)
        ddg_per_step = [self.decoder(h).squeeze(-1).item() for h in trajectory]
        return {"h_wt": h_wt, "h_mut_raw": h_mut_raw, "trajectory": trajectory, "ddg_per_step": ddg_per_step}
