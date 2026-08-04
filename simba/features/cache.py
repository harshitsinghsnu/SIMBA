"""On-disk cache for per-sequence PLM scores, keyed by (model_name, sequence
hash). Sequences repeat heavily across mutation rows sharing a complex, so
caching keeps embedding extraction a one-time cost per unique sequence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from simba.features.plm import ESMScorer, SequenceScore

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "embeddings_cache"


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _seq_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:24]


class SequenceScoreCache:
    def __init__(self, scorer: ESMScorer, cache_dir: Path = CACHE_DIR):
        self.scorer = scorer
        self.dir = cache_dir / _sanitize(scorer.model_name)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sequence: str) -> Path:
        return self.dir / f"{_seq_hash(sequence)}.npz"

    def get(self, sequence: str) -> SequenceScore:
        path = self._path(sequence)
        if path.exists():
            data = np.load(path)
            return SequenceScore(embedding=data["embedding"], logprobs=data["logprobs"])
        score = self.scorer.score(sequence)
        np.savez_compressed(path, embedding=score.embedding, logprobs=score.logprobs)
        return score

    def get_many(self, sequences: list[str], show_progress: bool = True) -> dict[str, SequenceScore]:
        unique = sorted(set(sequences))
        iterator = unique
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(unique, desc=f"scoring ({self.scorer.model_name})")
        return {seq: self.get(seq) for seq in iterator}


class MaskedMarginalCache:
    """Sparse cache of masked-marginal log-probs, keyed by (sequence, position).

    Unlike SequenceScoreCache (one full-sequence pass, always covers every
    position), masked-marginal scoring costs one forward pass per position of
    interest, so this only ever computes the specific positions actually
    referenced by a mutation somewhere in the data -- typically a handful per
    sequence, not the whole length.
    """

    def __init__(self, scorer: ESMScorer, cache_dir: Path = CACHE_DIR):
        self.scorer = scorer
        self.dir = cache_dir / (_sanitize(scorer.model_name) + "_masked")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._memo: dict[str, dict[int, np.ndarray]] = {}

    def _path(self, sequence: str) -> Path:
        return self.dir / f"{_seq_hash(sequence)}.npz"

    def _load_from_disk(self, sequence: str) -> dict[int, np.ndarray]:
        path = self._path(sequence)
        if not path.exists():
            return {}
        data = np.load(path)
        return dict(zip(data["positions"].tolist(), data["logprobs"]))

    def _save(self, sequence: str, table: dict[int, np.ndarray]) -> None:
        positions = np.array(sorted(table.keys()), dtype=np.int64)
        logprobs = np.stack([table[p] for p in positions]) if len(positions) else \
            np.zeros((0, len(table and next(iter(table.values())))), dtype=np.float32)
        np.savez_compressed(self._path(sequence), positions=positions, logprobs=logprobs)

    def ensure(self, sequence: str, positions: set[int]) -> None:
        """Make sure every position in `positions` is scored and cached."""
        table = self._memo.get(sequence) or self._load_from_disk(sequence)
        missing = sorted(p for p in positions if p not in table)
        if missing:
            new_logprobs = self.scorer.masked_marginal_logprobs(sequence, missing)
            for p, lp in zip(missing, new_logprobs):
                table[p] = lp
            self._save(sequence, table)
        self._memo[sequence] = table

    def get(self, sequence: str, position: int) -> np.ndarray | None:
        table = self._memo.get(sequence)
        if table is None:
            table = self._load_from_disk(sequence)
            self._memo[sequence] = table
        return table.get(position)
