"""Absolute protein-protein binding-affinity (dG) model. The partner chain
is compressed into a fixed-size interface memory (GatedMemory, the same
primitive used for the mutation-effect models' "concat" mode), and the self
chain's own tokens act as the query side of a dense pairwise map against
that memory (PairwiseInterfaceBlock -- the same Pairformer-lite row/column
-attention block already validated as chain_mode="pairformer" on the
mutation-effect models). `use_pairformer=False` keeps the simpler single
-query GatedMemory readout as a cheap ablation baseline -- though note it
never looks at the self chain at all (see `forward_ddg`'s docstring), so it
is only a meaningful baseline for single-complex dG scoring, not for ddG.

Which chain is "self" vs "partner" is an arbitrary but fixed convention
inherited from build_affinity_benchmark.py (chain_id_1 -> self, chain_id_2 ->
partner) -- the same asymmetric-but-fixed convention used throughout the
rest of this project's ddG pipeline, not a claim that the task is physically
asymmetric.

`forward_ddg` reuses this exact same absolute-affinity model for mutation
-effect (ddG) prediction, unchanged: score the wildtype complex, score the
mutant complex (mutation(s) already substituted into the self sequence),
and take the difference. No separate architecture, no background zero-shot
term, no per-site counterfactual machinery -- the same model that predicts
one complex's dG, called twice.

Four ablatable extensions on top of the base design:
  `n_pairformer_layers` -- stack more than one PairformerLayer (row+column
      attention pass) before pooling, instead of just one.
  `use_epinet`          -- epistemic-index residual head (Osband et al.) for
      calibrated uncertainty, same design as the mutation-effect models'
      Epinet. `forward_ddg` shares one epistemic index `z` between its two
      internal calls, so the sampled residual reflects one consistent
      posterior draw across both the wildtype and mutant score, rather than
      two independent (and therefore partly self-cancelling-by-accident)
      draws.
  `head_type="cosine"`  -- instead of an MLP regression head, project the
      pooled representation and compare it via cosine similarity to a single
      learned reference vector, then map that similarity into the dataset's
      real-valued target range via a fixed linear transform (matching a
      cosine-similarity-based affinity-regression scheme): `pred = (cos_sim +
      1) / 2 * (target_max - target_min) + target_min`. Training uses
      `forward_cos_sim` directly with the true target rescaled the same way
      into [-1, 1], via a cosine-MSE loss, rather than MSE on `forward()`'s
      already-rescaled output.
  `use_hl_cross_attention` -- hierarchical fusion for a two-chain "self"
      side (e.g. an antibody's heavy and light chain): heavy and light
      chains are embedded and projected *separately*, then fused via
      `HeavyLightCrossAttention` (bidirectional, gated cross-attention)
      *before* that fused representation attends to the partner (e.g. the
      antigen) through the unchanged pairformer/GatedMemory pathway below.
      Everything downstream of the fused self-tokens (partner memory,
      pairformer, head, epinet) is exactly the same code path as the
      single-self-sequence model -- only how `self_tokens` gets built
      changes. Used via `forward_paired`/`forward_ddg_paired` instead of
      `forward`/`forward_ddg`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from simba.models.simba import Epinet, GatedMemory, HeavyLightCrossAttention, PairwiseInterfaceBlock, make_projection


class AbsoluteAffinityModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        proj_mid_dim: int | None = 512,
        n_heads: int = 4,
        n_memory: int = 24,
        pair_hidden: int = 64,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
        use_pairformer: bool = True,
        n_pairformer_layers: int = 1,
        use_epinet: bool = False,
        epinet_index_dim: int = 8,
        epinet_hidden: int = 32,
        epinet_prior_scale: float = 1.0,
        head_type: str = "mlp",
        cosine_ref_dim: int | None = None,
        target_min: float = -10.0,
        target_max: float = 10.0,
        use_hl_cross_attention: bool = False,
    ):
        super().__init__()
        assert head_type in ("mlp", "cosine")
        self.use_pairformer = use_pairformer
        self.use_epinet = use_epinet
        self.head_type = head_type
        self.use_hl_cross_attention = use_hl_cross_attention

        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.partner_memory = GatedMemory(d_model, n_heads, n_memory)
        if use_hl_cross_attention:
            self.hl_cross_attn = HeavyLightCrossAttention(d_model, n_heads)
        if use_pairformer:
            self.pairwise_block = PairwiseInterfaceBlock(
                d_model, n_heads, pair_hidden, dropout, n_layers=n_pairformer_layers)
        else:
            self.readout_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )
        else:
            ref_dim = cosine_ref_dim or d_model
            self.cos_proj = nn.Linear(d_model, ref_dim)
            self.cos_ref = nn.Parameter(torch.randn(ref_dim) * 0.02)
            self.register_buffer("target_min", torch.tensor(float(target_min)))
            self.register_buffer("target_max", torch.tensor(float(target_max)))

        if use_epinet:
            self.epinet = Epinet(d_model, epinet_index_dim, epinet_hidden, epinet_prior_scale)

    def _pooled_from_tokens(self, self_tokens: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        """Shared tail: `self_tokens` (already projected to d_model, however
        it was constructed) attends to the partner side. Identical code path
        for the plain single-self-sequence model and the heavy/light
        hierarchical variant -- only how `self_tokens` is built differs."""
        partner_tokens = self.proj(partner_embedding)  # [Lp, d_model]
        memory = self.partner_memory.write(partner_tokens)  # [n_memory, d_model]

        if self.use_pairformer:
            return self.pairwise_block(self_tokens, memory)  # [d_model]
        return self.partner_memory.read(self.readout_query, memory)  # [d_model]

    def _pooled(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        self_tokens = self.proj(self_embedding)  # [Ls, d_model]
        return self._pooled_from_tokens(self_tokens, partner_embedding)

    def _pooled_paired(
        self, heavy_embedding: torch.Tensor, light_embedding: torch.Tensor, partner_embedding: torch.Tensor,
    ) -> torch.Tensor:
        assert self.use_hl_cross_attention
        heavy_tokens = self.proj(heavy_embedding)  # [Lh, d_model]
        light_tokens = self.proj(light_embedding)  # [Ll, d_model] (Ll may be 0)
        self_tokens = self.hl_cross_attn(heavy_tokens, light_tokens)  # [Lh+Ll, d_model]
        return self._pooled_from_tokens(self_tokens, partner_embedding)

    def _scalar_from_pooled(self, pooled: torch.Tensor, epinet_z: torch.Tensor | None) -> torch.Tensor:
        """Shared head + epinet dispatch, used by both `forward` and
        `forward_paired` so the two pathways can never silently diverge in
        how the pooled representation becomes a scalar prediction."""
        if self.head_type == "mlp":
            out = self.head(pooled).squeeze(-1)
        else:
            v = self.cos_proj(pooled)
            cos_sim = F.cosine_similarity(v.unsqueeze(0), self.cos_ref.unsqueeze(0), dim=-1).squeeze(0)
            out = (cos_sim + 1) / 2 * (self.target_max - self.target_min) + self.target_min

        if self.use_epinet:
            z = epinet_z if epinet_z is not None else torch.randn(self.epinet.index_dim, device=pooled.device)
            out = out + self.epinet(pooled, z)
        return out

    def forward_cos_sim(self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor) -> torch.Tensor:
        """Raw cosine similarity in [-1, 1], before mapping to the target
        range. Only valid when `head_type="cosine"`; used directly by the
        cosine-MSE training loss (the true target is rescaled to [-1, 1]
        the same way, so the loss never needs `forward()`'s mapped output)."""
        assert self.head_type == "cosine"
        pooled = self._pooled(self_embedding, partner_embedding)
        v = self.cos_proj(pooled)
        return F.cosine_similarity(v.unsqueeze(0), self.cos_ref.unsqueeze(0), dim=-1).squeeze(0)

    def forward(
        self,
        self_embedding: torch.Tensor,
        partner_embedding: torch.Tensor,
        epinet_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`self_embedding`/`partner_embedding`: [L, plm_hidden] frozen-backbone
        token embeddings for the two chain groups making up one complex.
        Returns a scalar predicted dG (kcal/mol), already in real units
        regardless of `head_type`."""
        pooled = self._pooled(self_embedding, partner_embedding)
        return self._scalar_from_pooled(pooled, epinet_z)

    def forward_paired(
        self,
        heavy_embedding: torch.Tensor,
        light_embedding: torch.Tensor,
        partner_embedding: torch.Tensor,
        epinet_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hierarchical variant of `forward`: heavy and light chains are
        embedded/projected separately and fused via `HeavyLightCrossAttention`
        before attending to the partner -- everything downstream (partner
        memory, pairformer, head, epinet) is identical to `forward`. Requires
        `use_hl_cross_attention=True`."""
        pooled = self._pooled_paired(heavy_embedding, light_embedding, partner_embedding)
        return self._scalar_from_pooled(pooled, epinet_z)

    def predict_with_uncertainty(
        self, self_embedding: torch.Tensor, partner_embedding: torch.Tensor, n_samples: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draws `n_samples` fresh epistemic indices and returns (mean, std)
        of the resulting predictions. Requires `use_epinet=True`."""
        assert self.use_epinet, "predict_with_uncertainty requires use_epinet=True"
        preds = torch.stack([self.forward(self_embedding, partner_embedding) for _ in range(n_samples)])
        return preds.mean(), preds.std()

    def forward_ddg(
        self,
        wt_self_embedding: torch.Tensor,
        mut_self_embedding: torch.Tensor,
        partner_embedding: torch.Tensor,
        epinet_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ddG_pred = dG(mutant complex) - dG(wildtype complex), both scored
        by the identical absolute-affinity forward pass above. Handles
        multi-site and multi-chain mutations for free: `mut_self_embedding`
        is just the frozen-backbone embedding of the self-side sequence with
        every mutation already substituted in, however many sites/chains
        that spans -- no per-site enumeration or pairwise epistasis term
        needed, since the whole mutant sequence is scored at once.

        When `use_epinet=True`, the *same* epistemic index is used for both
        the wildtype and mutant calls (sampled once here if not given), so a
        single posterior draw scores both complexes consistently rather than
        two independently-noisy ones."""
        if self.use_epinet and epinet_z is None:
            epinet_z = torch.randn(self.epinet.index_dim, device=wt_self_embedding.device)
        dg_wt = self.forward(wt_self_embedding, partner_embedding, epinet_z=epinet_z)
        dg_mut = self.forward(mut_self_embedding, partner_embedding, epinet_z=epinet_z)
        return dg_mut - dg_wt

    def predict_ddg_with_uncertainty(
        self,
        wt_self_embedding: torch.Tensor,
        mut_self_embedding: torch.Tensor,
        partner_embedding: torch.Tensor,
        n_samples: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same idea as `predict_with_uncertainty`, for ddG: each of the
        `n_samples` draws uses one shared epistemic index across the paired
        wildtype/mutant calls (see `forward_ddg`)."""
        assert self.use_epinet, "predict_ddg_with_uncertainty requires use_epinet=True"
        preds = torch.stack([
            self.forward_ddg(wt_self_embedding, mut_self_embedding, partner_embedding)
            for _ in range(n_samples)
        ])
        return preds.mean(), preds.std()

    def forward_ddg_paired(
        self,
        wt_heavy_embedding: torch.Tensor,
        wt_light_embedding: torch.Tensor,
        mut_heavy_embedding: torch.Tensor,
        mut_light_embedding: torch.Tensor,
        partner_embedding: torch.Tensor,
        epinet_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hierarchical variant of `forward_ddg`: heavy and light chains are
        kept separate (each with wildtype/mutant versions) and fused via
        `HeavyLightCrossAttention` before each of the two dG scoring passes,
        rather than being pre-concatenated into one "self" string. Same
        shared-epinet-index behavior as `forward_ddg`."""
        if self.use_epinet and epinet_z is None:
            epinet_z = torch.randn(self.epinet.index_dim, device=wt_heavy_embedding.device)
        dg_wt = self.forward_paired(wt_heavy_embedding, wt_light_embedding, partner_embedding, epinet_z=epinet_z)
        dg_mut = self.forward_paired(mut_heavy_embedding, mut_light_embedding, partner_embedding, epinet_z=epinet_z)
        return dg_mut - dg_wt
