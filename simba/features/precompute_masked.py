"""Precompute masked-marginal log-probs for every (sequence, position) pair
actually referenced by a mutation site anywhere across our loaders -- the
sparse counterpart to `precompute.py`'s full-sequence wildtype-marginal pass.
"""

from __future__ import annotations

import argparse

from simba.data.dataset import load_all_mutations, load_protattba, load_rde_skempi


def collect_needed_positions() -> dict[str, set[int]]:
    needed: dict[str, set[int]] = {}
    loaders = [load_all_mutations, load_rde_skempi] + [
        lambda n=n: load_protattba(n) for n in ("ab645", "ab1101", "s1131")
    ]
    for loader in loaders:
        for ex in loader():
            for site in ex.sites:
                needed.setdefault(site.self_sequence, set()).add(site.seq_index)
    return needed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from simba.features.cache import MaskedMarginalCache
    from simba.features.plm import ESMScorer

    needed = collect_needed_positions()
    total_positions = sum(len(v) for v in needed.values())
    print(f"unique sequences: {len(needed)}, total (sequence, position) pairs: {total_positions}")

    scorer = ESMScorer(model_name=args.model, device=args.device)
    cache = MaskedMarginalCache(scorer)

    from tqdm import tqdm
    for seq, positions in tqdm(needed.items(), desc="masked-marginal scoring"):
        cache.ensure(seq, positions)


if __name__ == "__main__":
    main()
