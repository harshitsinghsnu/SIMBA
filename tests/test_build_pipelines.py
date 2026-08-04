from pathlib import Path

import pandas as pd
import pytest

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SKEMPI_CSV = PROC_DIR / "skempi2_mutations.csv"
ABBIND_CSV = PROC_DIR / "abbind_mutations.csv"

pytestmark = pytest.mark.skipif(
    not SKEMPI_CSV.exists() or not ABBIND_CSV.exists(),
    reason="processed data not built; run `python -m simba.data.build_skempi` "
    "and `python -m simba.data.build_abbind` first",
)


@pytest.mark.parametrize("path,min_resolved_frac,min_rows", [
    (SKEMPI_CSV, 0.99, 6000),
    (ABBIND_CSV, 0.99, 1000),
])
def test_resolution_rate(path, min_resolved_frac, min_rows):
    df = pd.read_csv(path)
    assert len(df) >= min_rows
    resolved_frac = df["resolved"].mean()
    assert resolved_frac >= min_resolved_frac, (
        f"{path.name}: only {resolved_frac:.1%} of mutations resolved "
        f"(expected >= {min_resolved_frac:.0%})"
    )


@pytest.mark.parametrize("path", [SKEMPI_CSV, ABBIND_CSV])
def test_ddg_is_finite(path):
    df = pd.read_csv(path)
    assert df["ddG"].notna().all()
    assert df["ddG"].abs().max() < 50  # sanity bound in kcal/mol, catches unit bugs
