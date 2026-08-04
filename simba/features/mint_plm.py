"""MINT (Ullanat et al., "Learning the language of protein-protein
interactions", PPI-pretrained on 96M STRING pairs) wrapper for the PLM
-backbone ablation (MODEL_REPORT.md S3.20).

Architecturally different from ESM2/AbLang2 here: those two encode each
chain independently (no cross-chain context at the PLM level -- all
cross-chain reasoning happens downstream, in SIMBA's own pairformer). MINT
instead *jointly* encodes the concatenated self+partner sequence with a
shared `chain_ids` tensor marking which chain each token belongs to, so its
per-residue embeddings are already cross-chain-aware before they ever reach
SIMBA's pairformer. This is MINT's actual design (not a workaround) --
stripping that away by encoding chains separately would defeat the point of
testing this specific backbone, so `embed_pair` always runs both chains
through one joint forward pass, then slices out each chain's own residue
rows by position by exact index (see `_split_by_chain`) matching the
tokenization convention from `mint/helpers/extract.py`'s `CollateFn`: each
chain is independently encoded as `<cls>SEQ<eos>` and only *then*
concatenated, so a chain's own residue rows sit at a fixed offset with no
cross-chain special tokens in between.

Because embedding depends on *both* sequences together, this can't reuse
`SequenceScoreCache` (keyed on one sequence) -- `MintPairCache` below is
the pair-keyed equivalent, storing both chains' embeddings from one cached
forward pass.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

MINT_REPO = Path(__file__).resolve().parent.parent.parent / "third_party" / "mint"
if str(MINT_REPO) not in sys.path:
    sys.path.insert(0, str(MINT_REPO))

from mint.data import Alphabet  # noqa: E402
from mint.helpers.extract import load_config, MINTWrapper  # noqa: E402

CKPT_PATH = MINT_REPO / "mint.ckpt"
CFG_PATH = MINT_REPO / "data" / "esm2_t33_650M_UR50D.json"
HIDDEN = 1280
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "embeddings_cache" / "mint"


class MINTPairScorer:
    model_name = "mint"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        cfg = load_config(str(CFG_PATH))
        self.wrapper = MINTWrapper(cfg, str(CKPT_PATH), sep_chains=False, device=self.device)
        self.wrapper.eval()
        for p in self.wrapper.parameters():
            p.requires_grad_(False)

    def _encode_chain(self, sequence: str) -> list[int]:
        return self.alphabet.encode("<cls>" + sequence.replace("J", "L") + "<eos>")

    @torch.no_grad()
    def embed_pair(self, self_seq: str, partner_seq: str) -> tuple[np.ndarray, np.ndarray]:
        self_ids = self._encode_chain(self_seq)
        partner_ids = self._encode_chain(partner_seq)
        tokens = torch.tensor(self_ids + partner_ids, dtype=torch.int64, device=self.device).unsqueeze(0)
        chain_ids = torch.cat([
            torch.zeros(len(self_ids), dtype=torch.int32),
            torch.ones(len(partner_ids), dtype=torch.int32),
        ]).unsqueeze(0).to(self.device)

        chain_out = self.wrapper.model(tokens, chain_ids, repr_layers=[33])["representations"][33][0]
        # chain_out: [total_len, 1280]. Exact-offset slice (batch size 1, no
        # padding) rather than boolean masking -- equally correct here and
        # avoids any doubt about mask/order interaction.
        self_emb = chain_out[1:1 + len(self_seq)]
        partner_start = len(self_ids) + 1
        partner_emb = chain_out[partner_start:partner_start + len(partner_seq)]

        assert self_emb.shape[0] == len(self_seq), f"{self_emb.shape[0]} vs {len(self_seq)}"
        assert partner_emb.shape[0] == len(partner_seq), f"{partner_emb.shape[0]} vs {len(partner_seq)}"

        return (self_emb.to(torch.float16).cpu().numpy(),
                partner_emb.to(torch.float16).cpu().numpy())


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _pair_hash(self_seq: str, partner_seq: str) -> str:
    return hashlib.sha256((self_seq + "||" + partner_seq).encode("utf-8")).hexdigest()[:24]


class MintPairCache:
    def __init__(self, scorer: MINTPairScorer, cache_dir: Path = CACHE_DIR):
        self.scorer = scorer
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, self_seq: str, partner_seq: str) -> Path:
        return self.dir / f"{_pair_hash(self_seq, partner_seq)}.npz"

    def get(self, self_seq: str, partner_seq: str) -> tuple[np.ndarray, np.ndarray]:
        path = self._path(self_seq, partner_seq)
        if path.exists():
            data = np.load(path)
            return data["self_emb"], data["partner_emb"]
        self_emb, partner_emb = self.scorer.embed_pair(self_seq, partner_seq)
        np.savez_compressed(path, self_emb=self_emb, partner_emb=partner_emb)
        return self_emb, partner_emb
