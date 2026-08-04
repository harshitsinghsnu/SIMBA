from pathlib import Path

import pandas as pd
import pytest

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ABDESIGN_CSV = PROC_DIR / "abdesign_mutations.csv"

pytestmark = pytest.mark.skipif(
    not ABDESIGN_CSV.exists(),
    reason="AbDesign not built; run `python -m simba.data.build_abdesign` first "
    "(requires manually downloaded data/raw/abdesign/datasets_mut.csv)",
)


def test_all_rows_resolved():
    df = pd.read_csv(ABDESIGN_CSV)
    assert len(df) == 1303
    assert df["resolved"].all()


def test_label_types_match_expected_counts():
    df = pd.read_csv(ABDESIGN_CSV)
    counts = df.groupby(["source", "label_type"]).size().to_dict()
    assert counts[("AbDesign", "elisa_ratio")] == 658
    assert counts[("SKEMPIv2", "ddg")] == 377
    assert counts[("AB-Bind", "ddg")] == 268


def test_ddg_and_elisa_ratio_are_mutually_exclusive():
    df = pd.read_csv(ABDESIGN_CSV)
    ddg_rows = df[df["label_type"] == "ddg"]
    elisa_rows = df[df["label_type"] == "elisa_ratio"]
    assert ddg_rows["ddG"].notna().all()
    assert ddg_rows["elisa_ratio"].isna().all()
    assert elisa_rows["elisa_ratio"].notna().all()
    assert elisa_rows["ddG"].isna().all()


def test_unique_complexes():
    df = pd.read_csv(ABDESIGN_CSV)
    assert df["complex_id"].nunique() == 47
