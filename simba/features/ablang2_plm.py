"""AbLang2 (Olsen et al., antibody-specific PLM) wrapper with the same
`.score(sequence) -> SequenceScore` / `.model_name` interface as `ESMScorer`,
so it drops straight into `SequenceScoreCache` unchanged. Used for the PLM
-backbone ablation on the combined pool's cold_target split (MODEL_REPORT.md
S3.20) -- this pool is general PPI, not just antibody-antigen, so most
sequences here are genuinely out-of-domain for AbLang2. That's an honest,
intentional part of the ablation (does an antibody-specific PLM transfer to
non-antibody proteins), not an oversight.

Single-chain extraction (`[seq, ""]`, take rows `1:1+len(seq)`) reuses the
exact convention already verified in `eval_ablang2_abdesign.py`: passing the
other chain as an empty string reliably isolates one chain's own per-residue
rows, with output length always `len(seq)+3` (BOS+seq+EOS, no separator/other
-chain contribution).

No logprobs equivalent exists for AbLang2 in this pipeline (not needed --
`AbsoluteAffinityModel` training here only ever reads `.embedding`), so
`logprobs` is a zero placeholder purely to satisfy `SequenceScore`'s shape.
"""

from __future__ import annotations

import numpy as np

import ablang2

from simba.features.plm import SequenceScore

HIDDEN = 480


class AbLang2Scorer:
    model_name = "ablang2-paired"

    def __init__(self, device: str | None = None):
        # ablang2 manages its own device placement internally; "cpu" here is
        # just the handle string it's constructed with, matching
        # eval_ablang2_abdesign.py's usage.
        self.model = ablang2.pretrained(model_to_use="ablang2-paired", device="cpu")

    def score(self, sequence: str) -> SequenceScore:
        if not sequence:
            return SequenceScore(embedding=np.zeros((0, HIDDEN), dtype=np.float16),
                                  logprobs=np.zeros((0, 20), dtype=np.float32))
        out = self.model([[sequence, ""]], mode="rescoding")[0]  # [len(seq)+3, 480]
        chain_emb = out[1:1 + len(sequence)]
        assert chain_emb.shape[0] == len(sequence), (
            f"AbLang2 token/sequence length mismatch: {chain_emb.shape[0]} vs {len(sequence)}"
        )
        return SequenceScore(
            embedding=chain_emb.astype(np.float16),
            logprobs=np.zeros((len(sequence), 20), dtype=np.float32),
        )
