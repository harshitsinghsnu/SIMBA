from pathlib import Path

import pandas as pd
import pytest

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RDE_CSV = PROC_DIR / "rde_skempi_mutations.csv"

pytestmark = pytest.mark.skipif(
    not RDE_CSV.exists(),
    reason="RDE-PPI split not built; run `python -m simba.data.build_rde_skempi` first",
)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(RDE_CSV)


def test_fully_resolved(df):
    assert len(df) > 6000
    assert df["resolved"].all()


def test_1kbh_blocked(df):
    assert "1KBH" not in set(df["pdbcode"])


def test_three_folds_no_leakage(df):
    assert set(df["cvfold"]) == {0, 1, 2}
    per_complex_folds = df.groupby("complex")["cvfold"].nunique()
    assert (per_complex_folds == 1).all()


def test_fold_sizes_roughly_balanced(df):
    counts = df["complex"].drop_duplicates().map(
        df.drop_duplicates("complex").set_index("complex")["cvfold"]
    )
    # complex-level counts per fold should be within a few of each other
    # (ceil(n/3) contiguous split of a shuffled list)
    fold_complex_counts = df.drop_duplicates("complex")["cvfold"].value_counts()
    assert fold_complex_counts.max() - fold_complex_counts.min() <= 5
