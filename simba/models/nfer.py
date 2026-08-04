"""NFER (Neural Free Energy Relaxation Network): a third computational
primitive, distinct from both the pairformer (dense cross-attention) and
SIMBA-Operator (local convolutional propagation, simba/models/operator.py).
Here the primitive is OPTIMIZATION, not aggregation: the model learns a
scalar energy functional over a per-residue field, and the prediction is
the field that (approximately) minimizes it.

  Stage 1: Frozen PLM + staged projection -> initial per-residue field.
  Stage 2: Predicts F_wt and F_mut *separately* -- each is its own
           relaxation to equilibrium under the SAME learned energy
           functional (shared weights, Siamese-style), so the mutation's
           effect is entirely in how its different initial field relaxes
           differently, not in an explicit "perturbation" step.
  Stage 3: Neural Energy Solver -- unrolled gradient descent on a learned
           energy functional:

               E_theta(e) = sum_i Psi(e_i) + sum_{i,j neighbors} Phi(e_i, e_j)

           (Psi: per-residue "local strain" MLP -> scalar. Phi: pairwise
           "interaction energy" MLP over neighboring residues within a
           fixed local window -- the sequence-only proxy for spatial
           neighbors used everywhere else in this project that lacks 3D
           structure.) Then:

               e_{t+1} = e_t - eta * grad_e E_theta(e_t)

           for a fixed number of steps, `eta` itself a learned scalar. This
           is genuinely different from attention or convolution: there is
           no learned aggregation rule at all, just a learned scalar
           functional whose gradient defines the dynamics -- the model
           never directly outputs an update, it outputs an ENERGY, and the
           update is *derived* from that energy via autograd every step.
  Stage 4: Equilibrium decoder -- pool each equilibrium field and read out
           a scalar free energy F with one small MLP; `ddG_pred = F_mut -
           F_wt`.

Psi and Phi's last layers are zero-initialized, so `E_theta` is identically
zero at true init -- the gradient is exactly zero, and the "relaxation"
leaves the initial field completely unchanged (equilibrium = e_init at
init), the same zero-contribution-at-init property used throughout this
project's other primitives, achieved here without any Flamingo-style gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from simba.models.simba import make_projection


class NeuralEnergyFunctional(nn.Module):
    def __init__(self, d_model: int, hidden: int, window: int):
        super().__init__()
        self.window = window
        self.psi = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.phi = nn.Sequential(nn.Linear(d_model * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for net in (self.psi, self.phi):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """`e`: [L, d_model]. Returns a scalar total energy."""
        L = e.shape[0]
        if L == 0:
            return e.new_zeros(())
        energy = self.psi(e).sum()
        for offset in range(1, self.window + 1):
            if L <= offset:
                break
            pair_in = torch.cat([e[:-offset], e[offset:]], dim=-1)  # [L-offset, 2*d_model]
            energy = energy + self.phi(pair_in).sum()
        return energy


class NeuralEnergySolver(nn.Module):
    def __init__(self, d_model: int, hidden: int, window: int, n_steps: int = 4, init_step_size: float = 0.05):
        super().__init__()
        self.energy_fn = NeuralEnergyFunctional(d_model, hidden, window)
        self.n_steps = n_steps
        self.log_step_size = nn.Parameter(torch.log(torch.tensor(float(init_step_size))))

    def forward(self, e_init: torch.Tensor) -> torch.Tensor:
        """Unrolled gradient descent on `self.energy_fn`, `n_steps` updates.
        Uses `torch.enable_grad()` internally so this works correctly even
        when called from inside a `torch.no_grad()` evaluation loop -- the
        gradient here is w.r.t. the *field*, not the model parameters, and
        is needed to define the dynamics regardless of whether the outer
        call is for training or inference."""
        if e_init.shape[0] == 0:
            return e_init
        e = e_init
        for _ in range(self.n_steps):
            with torch.enable_grad():
                e_req = e if e.requires_grad else e.detach().requires_grad_(True)
                energy = self.energy_fn(e_req)
                (grad_e,) = torch.autograd.grad(energy, e_req, create_graph=self.training)
            e = e - torch.exp(self.log_step_size) * grad_e
        return e


class NFERModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        proj_mid_dim: int | None = 512,
        solver_hidden: int = 128,
        window: int = 3,
        n_steps: int = 4,
        init_step_size: float = 0.05,
        decoder_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.solver = NeuralEnergySolver(d_model, solver_hidden, window, n_steps, init_step_size)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )

    def _free_energy(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_proj = self.proj(self_embedding)  # [Ls, d_model]
        partner_proj = self.proj(partner_embedding)
        partner_summary = (partner_proj.mean(dim=0) if partner_proj.shape[0] > 0
                            else torch.zeros(self_proj.shape[-1], device=self_proj.device, dtype=self_proj.dtype))
        e_init = self_proj + partner_summary.unsqueeze(0)  # partner enters as a constant field bias
        equilibrium = self.solver(e_init)
        pooled = equilibrium.mean(dim=0) if equilibrium.shape[0] > 0 else equilibrium.new_zeros(equilibrium.shape[-1])
        return self.decoder(pooled).squeeze(-1)

    def forward(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ddG_pred = F(mutant equilibrium) - F(wildtype equilibrium),
        each F obtained by relaxing that sequence's own initial field to
        equilibrium under the identical (shared-weight) energy functional."""
        f_wt = self._free_energy(wt_self_embedding, partner_embedding)
        f_mut = self._free_energy(mut_self_embedding, partner_embedding)
        return f_mut - f_wt
