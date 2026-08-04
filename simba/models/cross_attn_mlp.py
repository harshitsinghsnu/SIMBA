"""Cross-attention + MLP: sits between ConcatMLPModel (no cross-chain
interaction at all) and AbsoluteAffinityModel's full pairformer (cross
-attention into a fixed-size memory, *then* a dense pairwise map refined by
several row/column-attention layers). This model has the cross-attention
step and nothing else: self chain tokens attend once, directly, to the raw
partner chain tokens (`nn.MultiheadAttention`, no GatedMemory bottleneck, no
PairwiseInterfaceBlock refinement), then the attended self tokens are
mean-pooled and passed to an MLP head.

Exists to separate two possible explanations for the pairformer's win over
concat+MLP (§3.8/§3.15): (a) any form of self<->partner cross-attention
already captures most of the benefit, or (b) the extra dense-pairwise-map +
row/column-attention refinement machinery is doing real additional work
beyond plain cross-attention. Comparing all three models on the same data
answers that directly.

Same `forward`/`forward_ddg` interface as `AbsoluteAffinityModel` and
`ConcatMLPModel` so it drops into the existing eval scripts unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttnMLPModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        n_heads: int = 4,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(plm_hidden_size, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _pooled(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_tokens = self.proj(self_embedding)        # [Ls, d_model]
        partner_tokens = self.proj(partner_embedding)  # [Lp, d_model]
        q = self_tokens.unsqueeze(0)
        kv = partner_tokens.unsqueeze(0)
        attn_out, _ = self.cross_attn(q, kv, kv)        # [1, Ls, d_model]
        fused = self_tokens + attn_out.squeeze(0)        # residual, [Ls, d_model]
        return fused.mean(dim=0)                         # [d_model]

    def forward(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled(self_embedding, partner_embedding)
        return self.head(pooled).squeeze(-1)

    def forward_ddg(
        self, wt_self_embedding: torch.Tensor, mut_self_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        dg_wt = self.forward(wt_self_embedding, partner_embedding)
        dg_mut = self.forward(mut_self_embedding, partner_embedding)
        return dg_mut - dg_wt
