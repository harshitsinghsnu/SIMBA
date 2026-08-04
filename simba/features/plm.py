"""Frozen ESM2 wrapper: one forward pass per sequence gives both the
per-residue embedding (for the learned selection head) and the wildtype
-marginal per-position amino-acid log-probabilities (for the free zero-shot
background term). No fine-tuning -- this is the "lightweight" half of SIMBA's
design: all the trainable parameters live downstream of this frozen backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from simba import env

env.setup()

from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: E402

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class SequenceScore:
    embedding: np.ndarray   # [L, hidden], float16
    logprobs: np.ndarray    # [L, 20], float32, columns ordered as STANDARD_AAS


class ESMScorer:
    def __init__(self, model_name: str = "facebook/esm2_t30_150M_UR50D", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        vocab = self.tokenizer.get_vocab()
        self._aa_vocab_idx = np.array([vocab[aa] for aa in STANDARD_AAS], dtype=np.int64)

    @torch.no_grad()
    def score(self, sequence: str) -> SequenceScore:
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)

        # strip <cls>/<eos> special tokens flanking the sequence
        logits = out.logits[0, 1:-1, :]                      # [L, vocab]
        log_probs_full = torch.log_softmax(logits, dim=-1)
        log_probs_aa = log_probs_full[:, self._aa_vocab_idx]  # [L, 20]

        hidden = out.hidden_states[-1][0, 1:-1, :]            # [L, hidden]

        assert log_probs_aa.shape[0] == len(sequence), (
            f"token/sequence length mismatch: {log_probs_aa.shape[0]} vs {len(sequence)}"
        )

        return SequenceScore(
            embedding=hidden.to(torch.float16).cpu().numpy(),
            logprobs=log_probs_aa.to(torch.float32).cpu().numpy(),
        )

    @torch.no_grad()
    def masked_marginal_logprobs(
        self, sequence: str, positions: list[int], batch_size: int = 32
    ) -> np.ndarray:
        """Masked-marginal scoring (Meier et al. 2021): for each requested
        0-indexed sequence position, mask that single token and re-score --
        a stronger zero-shot mutation-effect proxy than the single-pass
        wildtype-marginal (which never actually hides the wildtype identity
        from the model). Positions for the same sequence are batched together
        since they share every other token.
        """
        if not positions:
            return np.zeros((0, len(STANDARD_AAS)), dtype=np.float32)

        base = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        base_ids = base["input_ids"][0]  # [1+L+1]
        mask_id = self.tokenizer.mask_token_id

        results = np.empty((len(positions), len(STANDARD_AAS)), dtype=np.float32)
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            input_ids = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
            token_positions = [p + 1 for p in chunk]  # +1 for leading <cls>
            for row, tok_pos in enumerate(token_positions):
                input_ids[row, tok_pos] = mask_id
            attention_mask = base["attention_mask"].repeat(len(chunk), 1)

            out = self.model(input_ids=input_ids.to(self.device),
                              attention_mask=attention_mask.to(self.device))
            log_probs_full = torch.log_softmax(out.logits, dim=-1)  # [b, T, vocab]
            for row, tok_pos in enumerate(token_positions):
                row_logp = log_probs_full[row, tok_pos, self._aa_vocab_idx]
                results[start + row] = row_logp.to(torch.float32).cpu().numpy()

        return results
