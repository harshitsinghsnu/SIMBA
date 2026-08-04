"""Concatenation + MLP baseline: the simplest possible architecture that
still uses both chains, no cross-attention, no interface memory, no
pairformer. Each chain's frozen-PLM per-residue embeddings are mean-pooled
to a single vector, projected through a shared small linear layer, the two
pooled vectors are concatenated, and an MLP maps the concatenation directly
to a scalar. Exists specifically to test whether the pairformer's
cross-attention machinery is doing anything a naive concatenation baseline
couldn't already do -- the most basic ablation of the "combine self and
partner via cross-attention" design choice (§1.1 of the model report).

Same `forward`/`forward_ddg` interface as `AbsoluteAffinityModel` so it
drops into the existing eval scripts unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatMLPModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(plm_hidden_size, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _pooled(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_vec = self.proj(self_embedding.mean(dim=0))       # [d_model]
        partner_vec = self.proj(partner_embedding.mean(dim=0))  # [d_model]
        return torch.cat([self_vec, partner_vec], dim=-1)       # [2*d_model]

    def forward(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled(self_embedding, partner_embedding)
        return self.head(pooled).squeeze(-1)

    def forward_ddg(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        dg_wt = self.forward(wt_self_embedding, partner_embedding)
        dg_mut = self.forward(mut_self_embedding, partner_embedding)
        return dg_mut - dg_wt
