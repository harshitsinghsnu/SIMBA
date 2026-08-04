"""SIMBA model.

  ddG_pred = a * sum(delta_logp)   -- frozen-PLM zero-shot background
           + sum_i readout(h_i)    -- per-site effect after interaction

Two interchangeable "core" mechanisms compute each site's feature h_i,
selected via `core=`:

  "static"    -- the original design: two separate Perceiver-style gated
                 memories (interface = partner side, chain = teammate side,
                 e.g. paired H/L chain), each written once and read once,
                 plus an explicit O(k^2) pairwise MLP over mutated-site pairs
                 for epistasis.

  "recurrent" -- a genuinely different computational primitive (not another
                 attention variant dressed up): a single shared latent FIELD
                 is initialized from the rest of the complex, then for
                 `n_rounds` weight-shared iterations, every mutated site
                 WRITES its current state into the field, the field
                 SELF-UPDATES (letting different sites' perturbations already
                 written in mix), and every site READS the updated field
                 back. Multiple mutations interact *indirectly*, through
                 shared state that evolves over rounds -- not via an explicit
                 pairwise enumeration -- so epistasis emerges from iterative
                 propagation (recycling, AlphaFold-style) rather than being
                 hard-coded as a second term, and cost is linear in the
                 number of mutated sites instead of quadratic.

Every attention/read step in both cores is Flamingo-gated (`tanh(alpha) *
attention_output`, alpha initialized to 0), so the model always starts
training equivalent to having no cross-attention/field contribution at all.

Site representation (static core): rather than describing a mutation only
symbolically (wildtype/mutant amino acid identity looked up in a 20-entry
table), the mutant sequence is actually run through the frozen backbone too
(cheap: it's just another sequence in the same embedding cache), and the site
feature is built from the *counterfactual* wildtype-vs-mutant comparison at
that position: their difference vector (in projected space) plus their raw
cosine similarity (in the backbone's own un-projected space, so it measures
the frozen model's actual representational shift, not something entangled
with a trainable projection). This replaces the old symbolic AA-embedding
lookup with a real, context-dependent signal of "what changed" -- motivated
by a control ablation showing every interaction-processing mechanism tried
(dual memory, self-attention fusion, the recurrent field) landed within noise
of a context-free baseline, i.e. the bottleneck was the input representation,
not the processing.

The projection from the frozen backbone's hidden size down to d_model is
staged (e.g. 1280 -> 512 -> 256 for ESM2-650M) rather than a single large
jump, so compression is gradual rather than a hard bottleneck.

Everything below the frozen ESM2 backbone is a few million parameters at
most -- this is the "lightweight" half of the design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from simba.features.plm import STANDARD_AAS

AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}


def make_projection(in_dim: int, mid_dim: int, out_dim: int) -> nn.Module:
    """Staged dimensionality reduction (e.g. 1280 -> 512 -> 256) instead of a
    single large linear jump, so a big backbone hidden size isn't collapsed
    too aggressively in one step."""
    if mid_dim is None or mid_dim <= out_dim:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, out_dim))


class GatedMemory(nn.Module):
    """A fixed-size set of learned latent tokens, written to by a variable
    -length context sequence (cross-attention, memory as query) and read from
    by a query sequence (cross-attention, memory as key/value), with the read
    step's contribution gated by a learnable scalar initialized to 0."""

    def __init__(self, d_model: int, n_heads: int, n_memory: int):
        super().__init__()
        self.memory_init = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.write_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.read_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def write(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[0] == 0:
            return self.memory_init
        kv = context.unsqueeze(0)
        mem_q = self.memory_init.unsqueeze(0)
        mem_out, _ = self.write_attn(mem_q, kv, kv)
        return mem_out.squeeze(0)

    def read(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem_kv = memory.unsqueeze(0)
        read_out, _ = self.read_attn(query.unsqueeze(0), mem_kv, mem_kv)
        return torch.tanh(self.gate) * read_out.mean(dim=1).squeeze(0)


class HeavyLightCrossAttention(nn.Module):
    """Bidirectional cross-attention between the two chains on one "self"
    side of a complex (e.g. an antibody's heavy and light chain) before that
    fused representation goes on to attend with the partner side (e.g. the
    antigen) via the existing GatedMemory/PairwiseInterfaceBlock pathway,
    unchanged. Each direction (heavy attends to light, light attends to
    heavy) is its own MultiheadAttention with its own Flamingo-style gate
    (init 0), so the block starts equivalent to no heavy/light interaction
    at all. If one chain is empty (a single-chain mutation with no paired
    chain), that chain's cross-attention step is skipped and the other
    chain passes through unchanged -- so this reduces to plain
    concatenation for every example that has no second self-side chain.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.heavy_attends_light = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.light_attends_heavy = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate_heavy = nn.Parameter(torch.tensor(0.0))
        self.gate_light = nn.Parameter(torch.tensor(0.0))

    def forward(self, heavy_tokens: torch.Tensor, light_tokens: torch.Tensor) -> torch.Tensor:
        """`heavy_tokens`: [Lh, d]. `light_tokens`: [Ll, d] (Ll may be 0).
        Returns the concatenated, fused [Lh+Ll, d] representation."""
        if heavy_tokens.shape[0] == 0:
            return light_tokens
        if light_tokens.shape[0] == 0:
            return heavy_tokens

        h_kv = light_tokens.unsqueeze(0)
        h_out, _ = self.heavy_attends_light(heavy_tokens.unsqueeze(0), h_kv, h_kv)
        heavy_fused = heavy_tokens + torch.tanh(self.gate_heavy) * h_out.squeeze(0)

        l_kv = heavy_tokens.unsqueeze(0)
        l_out, _ = self.light_attends_heavy(light_tokens.unsqueeze(0), l_kv, l_kv)
        light_fused = light_tokens + torch.tanh(self.gate_light) * l_out.squeeze(0)

        return torch.cat([heavy_fused, light_fused], dim=0)


class RecurrentInteractionField(nn.Module):
    """The new core operator (see module docstring). Sites communicate only
    through this shared field, iteratively, never directly with each other."""

    def __init__(self, d_model: int, n_heads: int, n_field: int, n_rounds: int = 3):
        super().__init__()
        self.field_init = nn.Parameter(torch.randn(n_field, d_model) * 0.02)
        self.env_write_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.site_write_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.field_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.site_read_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n_rounds = n_rounds
        self.site_write_gate = nn.Parameter(torch.tensor(0.0))
        self.field_update_gate = nn.Parameter(torch.tensor(0.0))
        self.site_read_gate = nn.Parameter(torch.tensor(0.0))

    def init_field(self, env_tokens: torch.Tensor) -> torch.Tensor:
        if env_tokens.shape[0] == 0:
            return self.field_init
        kv = env_tokens.unsqueeze(0)
        f0 = self.field_init.unsqueeze(0)
        out, _ = self.env_write_attn(f0, kv, kv)
        return out.squeeze(0)

    def run(self, field: torch.Tensor, site_states: torch.Tensor) -> torch.Tensor:
        """`site_states`: [K, d], one row per mutated site in the example
        (K can be 1). Returns refined site_states [K, d]."""
        for _ in range(self.n_rounds):
            write_out, _ = self.site_write_attn(
                field.unsqueeze(0), site_states.unsqueeze(0), site_states.unsqueeze(0))
            field = field + torch.tanh(self.site_write_gate) * write_out.squeeze(0)

            self_out, _ = self.field_self_attn(field.unsqueeze(0), field.unsqueeze(0), field.unsqueeze(0))
            field = field + torch.tanh(self.field_update_gate) * self_out.squeeze(0)

            read_out, _ = self.site_read_attn(
                site_states.unsqueeze(0), field.unsqueeze(0), field.unsqueeze(0))
            site_states = site_states + torch.tanh(self.site_read_gate) * read_out.squeeze(0)
        return site_states


class PairformerLayer(nn.Module):
    """One row+column attention update on a persistent [Wq, M, pair_dim]
    pairwise representation, Flamingo-gated (each layer starts equivalent to
    zero contribution). Stacking several of these (PairwiseInterfaceBlock's
    `n_layers`) is the sequence-only analog of a multi-block Pairformer/
    Evoformer stack -- no triangle-multiplicative update (that enforces 3D
    geometric consistency this project doesn't have access to), just
    row attention (per window-position, across memory slots) and column
    attention (per memory-slot, across window positions) per layer.
    """

    def __init__(self, pair_dim: int, n_heads: int):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(pair_dim, n_heads, batch_first=True)
        self.gate_row = nn.Parameter(torch.tensor(0.0))
        self.gate_col = nn.Parameter(torch.tensor(0.0))

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """`pair`: [Wq, M, pair_dim]. Returns the same shape."""
        row_out, _ = self.row_attn(pair, pair, pair)  # batch=Wq, seq=M
        pair = pair + torch.tanh(self.gate_row) * row_out

        pair_t = pair.transpose(0, 1)  # [M, Wq, pair_dim]: batch=M, seq=Wq
        col_out, _ = self.col_attn(pair_t, pair_t, pair_t)
        pair_t = pair_t + torch.tanh(self.gate_col) * col_out
        return pair_t.transpose(0, 1)  # back to [Wq, M, pair_dim]


class PairwiseInterfaceBlock(nn.Module):
    """Pairformer-lite: instead of collapsing the mutation-site window's
    attention over the interface memory straight to a single averaged context
    vector, build an explicit dense (window-position x memory-slot) pairwise
    representation and refine it with `n_layers` row/column-attention passes
    (`PairformerLayer`, each with its own weights) before pooling. Operates
    over the bounded-size interface memory (not raw partner-length tokens),
    so cost stays independent of partner sequence length like every other
    chain_mode.
    """

    def __init__(self, d_model: int, n_heads: int, pair_dim: int, dropout: float = 0.1, n_layers: int = 1):
        super().__init__()
        self.pair_proj = nn.Sequential(nn.Linear(d_model * 2, pair_dim), nn.ReLU())
        self.layers = nn.ModuleList([PairformerLayer(pair_dim, n_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(pair_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """`query`: [Wq, d] mutation-site window tokens. `memory`: [M, d]
        interface-memory latents. Returns a pooled [d] context vector."""
        d_model = query.shape[-1]
        wq, m = query.shape[0], memory.shape[0]
        if wq == 0 or m == 0:
            return torch.zeros(d_model, device=query.device, dtype=query.dtype)

        q_exp = query.unsqueeze(1).expand(wq, m, d_model)
        m_exp = memory.unsqueeze(0).expand(wq, m, d_model)
        pair = self.pair_proj(torch.cat([q_exp, m_exp], dim=-1))  # [Wq, M, pair_dim]

        for layer in self.layers:
            pair = layer(pair)

        pooled = pair.mean(dim=(0, 1))
        return self.dropout(self.out_proj(pooled))


class Epinet(nn.Module):
    """Epistemic neural network residual head (Osband et al.), TerraBind
    -inspired: a small trainable MLP plus a same-shaped *frozen random-prior*
    MLP (its untrained, fixed-weight twin), both taking the pooled per
    -complex feature concatenated with a random "epistemic index" z and
    producing an index-dim vector, combined via dot-product with z itself.
    Trained with an ordinary MSE step per example (z resampled fresh every
    forward call, matching standard ENN training), so nothing about
    `run_training`'s loop needs to change -- the prior network's fixed
    randomness is what keeps predictions diverse across z after convergence,
    which is what turns repeated sampling at inference into a calibrated
    posterior instead of the trainable net just memorizing a single z."""

    def __init__(self, feature_dim: int, index_dim: int, hidden: int, prior_scale: float = 1.0):
        super().__init__()
        self.index_dim = index_dim
        self.prior_scale = prior_scale
        self.trainable = nn.Sequential(
            nn.Linear(feature_dim + index_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, index_dim),
        )
        self.prior = nn.Sequential(
            nn.Linear(feature_dim + index_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, index_dim),
        )
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def forward(self, feature: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feature, z], dim=-1)
        out = self.trainable(x) + self.prior_scale * self.prior(x)
        return (out * z).sum(-1)


class SimbaModel(nn.Module):
    def __init__(
        self,
        plm_hidden_size: int = 1280,
        d_model: int = 256,
        proj_mid_dim: int | None = 512,
        n_heads: int = 4,
        n_memory: int = 24,
        mlp_hidden: int = 128,
        pair_hidden: int = 64,
        aa_embed_dim: int = 16,
        window: int = 8,
        dropout: float = 0.1,
        core: str = "static",
        n_rounds: int = 3,
        recurrent_use_pairwise: bool = False,
        chain_mode: str = "concat",
        use_epinet: bool = False,
        epinet_index_dim: int = 8,
        epinet_hidden: int = 32,
        epinet_prior_scale: float = 1.0,
    ):
        """`chain_mode` (static core only) controls how partner (opposite
        -side, e.g. antigen) and teammate (same-side, e.g. paired H/L chain)
        context are combined:
          "dual_memory" -- two separate gated memories, read separately (the
                            original design).
          "concat"      -- both sequences concatenated with no role
                            distinction, written into one shared memory.
          "self_attn"   -- both sequences get a learned role embedding added,
                            are self-attended together (letting partner and
                            teammate tokens mix directly), then written into
                            one shared memory.
          "none"        -- no interface/chain context at all (control
                            ablation: background term + local window only,
                            to check whether any of the above mechanisms are
                            earning their keep over a context-free baseline).
          "pairformer"  -- like "concat", but the site window reads the
                            shared memory through a PairwiseInterfaceBlock
                            (dense window x memory pairwise map + row/column
                            attention) instead of a single collapsed
                            attention-weighted average.
        """
        super().__init__()
        assert core in ("static", "recurrent")
        assert chain_mode in ("dual_memory", "concat", "self_attn", "none", "pairformer")
        self.core = core
        self.chain_mode = chain_mode
        self.window = window
        self.d_model = d_model
        self.recurrent_use_pairwise = recurrent_use_pairwise
        self.use_epinet = use_epinet

        self.proj = make_projection(plm_hidden_size, proj_mid_dim, d_model)
        self.aa_embed = nn.Embedding(len(STANDARD_AAS), aa_embed_dim)

        if core == "static":
            if chain_mode == "dual_memory":
                self.interface_memory = GatedMemory(d_model, n_heads, n_memory)
                self.chain_memory = GatedMemory(d_model, n_heads, n_memory)
                context_dim = d_model * 2
            elif chain_mode == "none":
                context_dim = 0
            else:
                self.combined_memory = GatedMemory(d_model, n_heads, n_memory)
                context_dim = d_model
                if chain_mode == "self_attn":
                    self.role_embed = nn.Embedding(2, d_model)  # 0=partner, 1=teammate
                    self.fusion_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                elif chain_mode == "pairformer":
                    self.pairwise_block = PairwiseInterfaceBlock(d_model, n_heads, pair_hidden, dropout)
            # site feature = context + wt_site_proj + (mut_site_proj - wt_site_proj) + cos_sim(1) + delta_logp(1)
            self.site_encoder = nn.Sequential(
                nn.Linear(context_dim + d_model * 2 + 2, mlp_hidden),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
            )
            self.independent_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))
            self.pair_mlp = nn.Sequential(
                nn.Linear(mlp_hidden * 2 + 2, pair_hidden),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(pair_hidden, 1),
            )
            self.mlp_hidden = mlp_hidden
            if use_epinet:
                self.epinet = Epinet(mlp_hidden, epinet_index_dim, epinet_hidden, epinet_prior_scale)
        else:
            assert not use_epinet, "epinet is only wired up for core='static'"
            self.field = RecurrentInteractionField(d_model, n_heads, n_memory, n_rounds)
            self.site_init_proj = nn.Linear(d_model + aa_embed_dim * 2 + 1, d_model)
            self.readout = nn.Sequential(
                nn.Linear(d_model, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )
            if recurrent_use_pairwise:
                self.pair_mlp = nn.Sequential(
                    nn.Linear(d_model * 2 + 2, pair_hidden),
                    nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(pair_hidden, 1),
                )

        self.background_scale = nn.Parameter(torch.tensor(1.0))

    # ---------------------------------------------------------------- static
    def _build_context_tokens(self, partner_embedding: torch.Tensor, teammate_embedding: torch.Tensor) -> torch.Tensor:
        """Used when chain_mode in {"concat", "self_attn"}: combine partner
        + teammate token sequences into one, for writing into a single
        shared memory (as opposed to "dual_memory"'s two separate memories)."""
        p = self.proj(partner_embedding)
        t = self.proj(teammate_embedding)
        if self.chain_mode == "self_attn":
            if p.shape[0] > 0:
                p = p + self.role_embed.weight[0]
            if t.shape[0] > 0:
                t = t + self.role_embed.weight[1]
            tokens = torch.cat([p, t], dim=0)
            if tokens.shape[0] > 0:
                attn_out, _ = self.fusion_attn(tokens.unsqueeze(0), tokens.unsqueeze(0), tokens.unsqueeze(0))
                tokens = tokens + attn_out.squeeze(0)
            return tokens
        return torch.cat([p, t], dim=0)

    def _site_feature_static(self, self_embedding, mut_embedding, memory, seq_index, delta_logp):
        """Counterfactual site feature: compares the frozen backbone's own
        wildtype-sequence vs. mutant-sequence embedding at the mutated
        position, rather than a symbolic amino-acid-identity lookup."""
        device = self_embedding.device
        L = self_embedding.shape[0]
        lo, hi = max(0, seq_index - self.window), min(L, seq_index + self.window + 1)
        query = self.proj(self_embedding[lo:hi])
        if self.chain_mode == "dual_memory":
            interface_memory, chain_memory = memory
            context = torch.cat([
                self.interface_memory.read(query, interface_memory),
                self.chain_memory.read(query, chain_memory),
            ], dim=-1)
        elif self.chain_mode == "none":
            context = None
        elif self.chain_mode == "pairformer":
            context = self.pairwise_block(query, memory)
        else:
            context = self.combined_memory.read(query, memory)

        wt_raw = self_embedding[seq_index]      # [plm_hidden], frozen, un-projected
        mut_raw = mut_embedding[seq_index]      # [plm_hidden], frozen, un-projected
        cos_sim = F.cosine_similarity(wt_raw.unsqueeze(0), mut_raw.unsqueeze(0)).squeeze(0)

        wt_site_proj = self.proj(wt_raw)        # [d_model]
        mut_site_proj = self.proj(mut_raw)      # [d_model]
        diff = mut_site_proj - wt_site_proj     # [d_model]

        delta_t = torch.tensor([delta_logp], dtype=torch.float32, device=device)
        cos_t = cos_sim.reshape(1)
        parts = [wt_site_proj, diff, cos_t, delta_t] if context is None else \
            [context, wt_site_proj, diff, cos_t, delta_t]
        feat = torch.cat(parts, dim=-1)
        return self.site_encoder(feat)

    def _forward_static(self, sites: list[dict], epinet_z: torch.Tensor | None = None) -> torch.Tensor:
        device = sites[0]["self_embedding"].device
        background = sum(s["delta_logp"] for s in sites if s["delta_logp"] is not None)
        total = torch.as_tensor(background, dtype=torch.float32, device=device) * self.background_scale

        memory_cache = {}
        h_list, seq_indices, chain_keys = [], [], []
        for s in sites:
            if s["delta_logp"] is None:
                continue
            key = (id(s["partner_embedding"]), id(s["teammate_embedding"]))
            if key not in memory_cache:
                if self.chain_mode == "dual_memory":
                    memory_cache[key] = (
                        self.interface_memory.write(self.proj(s["partner_embedding"])),
                        self.chain_memory.write(self.proj(s["teammate_embedding"])),
                    )
                elif self.chain_mode == "none":
                    memory_cache[key] = None
                else:
                    tokens = self._build_context_tokens(s["partner_embedding"], s["teammate_embedding"])
                    memory_cache[key] = self.combined_memory.write(tokens)

            h = self._site_feature_static(
                s["self_embedding"], s["mut_embedding"], memory_cache[key],
                s["seq_index"], s["delta_logp"],
            )
            h_list.append(h)
            seq_indices.append(s["seq_index"])
            chain_keys.append(id(s["self_embedding"]))

        for h in h_list:
            total = total + self.independent_head(h).squeeze(-1)

        k = len(h_list)
        for i in range(k):
            for j in range(i + 1, k):
                same_chain = 1.0 if chain_keys[i] == chain_keys[j] else 0.0
                dist_feat = 1.0 / (1.0 + abs(seq_indices[i] - seq_indices[j])) if same_chain else 0.0
                pair_extra = torch.tensor([same_chain, dist_feat], dtype=torch.float32, device=device)
                pair_in = torch.cat([h_list[i], h_list[j], pair_extra], dim=-1)
                total = total + self.pair_mlp(pair_in).squeeze(-1)

        if self.use_epinet:
            pooled = torch.stack(h_list, dim=0).mean(dim=0) if h_list else \
                torch.zeros(self.mlp_hidden, device=device)
            z = epinet_z if epinet_z is not None else torch.randn(self.epinet.index_dim, device=device)
            total = total + self.epinet(pooled, z)
        return total

    # ------------------------------------------------------------- recurrent
    def _forward_recurrent(self, sites: list[dict]) -> torch.Tensor:
        device = sites[0]["self_embedding"].device
        valid = [s for s in sites if s["delta_logp"] is not None]
        background = sum(s["delta_logp"] for s in valid)
        total = torch.as_tensor(background, dtype=torch.float32, device=device) * self.background_scale
        if not valid:
            return total

        # shared environment: every other chain any site in this example can see
        env_tensors, seen = [], set()
        for s in valid:
            for emb in (s["partner_embedding"], s["teammate_embedding"]):
                if emb.shape[0] > 0 and id(emb) not in seen:
                    env_tensors.append(self.proj(emb))
                    seen.add(id(emb))
        env_tokens = torch.cat(env_tensors, dim=0) if env_tensors else torch.zeros(0, self.d_model, device=device)
        field = self.field.init_field(env_tokens)

        site_states, seq_indices, chain_keys = [], [], []
        for s in valid:
            self_embedding = s["self_embedding"]
            L = self_embedding.shape[0]
            idx = s["seq_index"]
            lo, hi = max(0, idx - self.window), min(L, idx + self.window + 1)
            window_feat = self.proj(self_embedding[lo:hi]).mean(dim=0)
            wt_embed = self.aa_embed(torch.tensor(AA_TO_IDX[s["wt_aa"]], device=device))
            mut_embed = self.aa_embed(torch.tensor(AA_TO_IDX[s["mut_aa"]], device=device))
            delta_t = torch.tensor([s["delta_logp"]], dtype=torch.float32, device=device)
            init_feat = torch.cat([window_feat, wt_embed, mut_embed, delta_t], dim=-1)
            site_states.append(self.site_init_proj(init_feat))
            seq_indices.append(idx)
            chain_keys.append(id(self_embedding))

        site_states = torch.stack(site_states, dim=0)  # [K, d]
        refined = self.field.run(field, site_states)   # [K, d]

        for h in refined:
            total = total + self.readout(h).squeeze(-1)

        if self.recurrent_use_pairwise:
            k = refined.shape[0]
            for i in range(k):
                for j in range(i + 1, k):
                    same_chain = 1.0 if chain_keys[i] == chain_keys[j] else 0.0
                    dist_feat = 1.0 / (1.0 + abs(seq_indices[i] - seq_indices[j])) if same_chain else 0.0
                    pair_extra = torch.tensor([same_chain, dist_feat], dtype=torch.float32, device=device)
                    pair_in = torch.cat([refined[i], refined[j], pair_extra], dim=-1)
                    total = total + self.pair_mlp(pair_in).squeeze(-1)
        return total

    def forward(self, sites: list[dict], epinet_z: torch.Tensor | None = None) -> torch.Tensor:
        """`sites`: one Example's mutation sites, each a dict with keys
        self_embedding, partner_embedding, teammate_embedding (tensors),
        seq_index, wt_aa, mut_aa, delta_logp. Returns a scalar predicted ddG.
        `epinet_z` (static core, use_epinet only): fixes the epistemic index
        instead of sampling a fresh one -- used by `predict_with_uncertainty`
        to draw several correlated-nothing samples per example."""
        if self.core == "static":
            return self._forward_static(sites, epinet_z=epinet_z)
        return self._forward_recurrent(sites)

    def predict_with_uncertainty(self, sites: list[dict], n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
        """Draws `n_samples` fresh epistemic indices and returns (mean, std)
        of the resulting predictions -- a calibrated posterior over ddG
        rather than a single point estimate. Requires `use_epinet=True`."""
        assert self.use_epinet, "predict_with_uncertainty requires use_epinet=True"
        preds = torch.stack([self.forward(sites) for _ in range(n_samples)])
        return preds.mean(), preds.std()
