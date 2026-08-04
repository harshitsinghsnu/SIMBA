"""Precompute + cache PLM scores (embedding + wildtype-marginal log-probs) for
every unique wildtype sequence referenced across all processed datasets.

Only wildtype sequences are ever scored: SIMBA's background term and
selection head both read off a single forward pass over the wildtype chain
(wildtype-marginal scoring, standard practice for zero-shot PLM ddG methods),
so a mutant-sequence forward pass is never needed. Sequence-level content
deduplication (not just per-dataset key dedup) means the same antibody/antigen
chain reused across many mutation rows is only ever scored once.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# chain-key prefixes that hold a *mutant* sequence rather than wildtype -- skip these
_MUTANT_PREFIX = "mut_"


def collect_wildtype_sequences() -> set[str]:
    sequences: set[str] = set()
    for path in glob.glob(str(PROC_DIR / "*_sequences.json")):
        raw = json.loads(Path(path).read_text())
        for complex_id, chains in raw.items():
            for chain_key, value in chains.items():
                if chain_key.startswith(_MUTANT_PREFIX):
                    continue
                seq = value["sequence"] if isinstance(value, dict) else value
                if isinstance(seq, str) and seq:
                    sequences.add(seq)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from simba.features.cache import SequenceScoreCache
    from simba.features.plm import ESMScorer

    sequences = sorted(collect_wildtype_sequences())
    print(f"unique wildtype sequences to score: {len(sequences)}")

    scorer = ESMScorer(model_name=args.model, device=args.device)
    print(f"model: {args.model} on {scorer.device}")

    cache = SequenceScoreCache(scorer)
    cache.get_many(sequences)
    print(f"cache directory: {cache.dir}")


if __name__ == "__main__":
    main()
