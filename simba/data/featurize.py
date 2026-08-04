"""Bridges Example objects (plain sequences/positions) to the tensors
SimbaModel.forward expects, backed by the on-disk PLM score cache plus an
in-memory torch-tensor memo (so repeated appearances of the same wildtype
sequence across many mutation rows only get converted to a GPU tensor once
per training run)."""

from __future__ import annotations

import torch

from simba.data.dataset import Example
from simba.features.cache import MaskedMarginalCache, SequenceScoreCache
from simba.features.zeroshot import site_delta_logp, site_delta_logp_masked


class FeatureProvider:
    def __init__(
        self,
        cache: SequenceScoreCache,
        device: str,
        masked_cache: MaskedMarginalCache | None = None,
    ):
        self.cache = cache
        self.device = device
        self.masked_cache = masked_cache
        self._embed_memo: dict[str, torch.Tensor] = {}

    def _embedding_tensor(self, sequence: str) -> torch.Tensor:
        if sequence not in self._embed_memo:
            score = self.cache.get(sequence)
            self._embed_memo[sequence] = torch.from_numpy(score.embedding).float().to(self.device)
        return self._embed_memo[sequence]

    def _delta_logp(self, sequence: str, seq_index: int, wt_aa: str, mut_aa: str) -> float | None:
        if self.masked_cache is not None:
            vec = self.masked_cache.get(sequence, seq_index)
            if vec is not None:
                return site_delta_logp_masked(vec, wt_aa, mut_aa)
            # fall through to wildtype-marginal if this exact position was
            # never precomputed (e.g. ad-hoc inference outside the dataset)
        return site_delta_logp(self.cache.get(sequence), seq_index, wt_aa, mut_aa)

    def _mutant_sequence(self, site) -> str:
        return site.self_sequence[:site.seq_index] + site.mut_aa + site.self_sequence[site.seq_index + 1:]

    def featurize(self, example: Example) -> list[dict]:
        sites = []
        for site in example.sites:
            delta_logp = self._delta_logp(site.self_sequence, site.seq_index, site.wt_aa, site.mut_aa)
            self_embedding = self._embedding_tensor(site.self_sequence)
            mut_embedding = self._embedding_tensor(self._mutant_sequence(site))
            partner_embedding = (
                self._embedding_tensor(site.partner_sequence)
                if site.partner_sequence else torch.zeros(0, self_embedding.shape[-1], device=self.device)
            )
            teammate_embedding = (
                self._embedding_tensor(site.teammate_sequence)
                if site.teammate_sequence else torch.zeros(0, self_embedding.shape[-1], device=self.device)
            )
            sites.append({
                "self_embedding": self_embedding,
                "mut_embedding": mut_embedding,
                "partner_embedding": partner_embedding,
                "teammate_embedding": teammate_embedding,
                "seq_index": site.seq_index,
                "wt_aa": site.wt_aa,
                "mut_aa": site.mut_aa,
                "delta_logp": delta_logp,
            })
        return sites
