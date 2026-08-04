from pathlib import Path

import pandas as pd
import pytest

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
S4169_CSV = PROC_DIR / "geoppi_s4169_mutations.csv"
M1707_CSV = PROC_DIR / "geoppi_m1707_mutations.csv"

pytestmark = pytest.mark.skipif(
    not S4169_CSV.exists() or not M1707_CSV.exists(),
    reason="GeoPPI benchmarks not built; run `python -m simba.data.build_geoppi` first",
)


def test_s4169_row_count_and_partial_resolution():
    df = pd.read_csv(S4169_CSV)
    assert len(df) == 4169
    # Older SKEMPI-1.0-derived numbering convention doesn't cleanly match
    # SKEMPI2's redistributed PDB/.mapping files for every complex; resolved
    # subset is still large and honestly flagged rather than guessed.
    assert df["resolved"].mean() >= 0.55


def test_m1707_row_count_and_high_resolution():
    df = pd.read_csv(M1707_CSV)
    assert len(df) == 1707
    assert df["resolved"].mean() >= 0.98


def test_m1707_has_mutabind2_reference_predictions():
    df = pd.read_csv(M1707_CSV)
    assert df["mutabind2_prediction"].notna().mean() > 0.9
