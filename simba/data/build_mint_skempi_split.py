"""Builds a second cvfold assignment for the same SKEMPI2 rows/complexes as
`build_rde_skempi.py`, using split_seed=2023 instead of 2022 -- MINT's own
`SkempiDataset` class (their `downstream/GeneralPPI/SKEMPI_v2/prepare_data
.ipynb`) uses this exact convention (3-fold, complex-level,
`random.Random(split_seed).shuffle()`, split_seed=2023 default).

This is *not* a byte-identical reproduction of MINT's actual reported
numbers: the cell in their notebook that actually builds the `split_0/1/2`
columns used for their headline results uses an unseeded `np.random.shuffle`,
so even MINT's own authors can't regenerate their exact test rows from the
published code. `SkempiDataset` (seed=2023) is the closest fully-specified,
reproducible approximation of their protocol, documented as such rather than
presented as identical.

Reuses `rde_skempi_mutations.csv` (same rows, same complexes, same ddG
values -- only the fold assignment differs) and
`rde_skempi_sequences.json` unchanged.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SPLIT_SEED = 2023  # MINT's SkempiDataset default (vs. RDE-PPI's own 2022)
NUM_CVFOLDS = 3


def main() -> None:
    df = pd.read_csv(PROC_DIR / "rde_skempi_mutations.csv")

    complex_to_row_indices: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        complex_to_row_indices.setdefault(row["complex"], []).append(idx)

    complex_list = sorted(complex_to_row_indices.keys())
    random.Random(SPLIT_SEED).shuffle(complex_list)
    split_size = math.ceil(len(complex_list) / NUM_CVFOLDS)
    complex_splits = [
        complex_list[i * split_size:(i + 1) * split_size] for i in range(NUM_CVFOLDS)
    ]

    cvfold = pd.Series(-1, index=df.index)
    for fold_idx, fold_complexes in enumerate(complex_splits):
        for cplx in fold_complexes:
            cvfold.loc[complex_to_row_indices[cplx]] = fold_idx
    df["cvfold"] = cvfold

    df.to_csv(PROC_DIR / "mint_skempi_mutations.csv", index=False)
    print(f"kept rows: {len(df)}")
    print(f"unique complexes: {len(complex_list)}")
    print(f"fold sizes (complexes): {[len(f) for f in complex_splits]}")
    print("\ncvfold row counts:\n", df["cvfold"].value_counts())


if __name__ == "__main__":
    main()
