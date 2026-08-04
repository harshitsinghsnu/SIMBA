import json
from pathlib import Path

import pandas as pd
import pytest

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ALL_MUTATIONS = PROC_DIR / "all_mutations.csv"
CLUSTERS_JSON = PROC_DIR / "clusters.json"

pytestmark = pytest.mark.skipif(
    not ALL_MUTATIONS.exists() or not CLUSTERS_JSON.exists(),
    reason="splits not built; run `python -m simba.data.build_splits` first",
)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ALL_MUTATIONS)


@pytest.fixture(scope="module")
def clusters():
    return json.loads(CLUSTERS_JSON.read_text())


def test_no_complex_id_spans_multiple_clusters(df):
    per_complex = df.groupby("complex_id")["cluster_id"].nunique()
    offenders = per_complex[per_complex > 1]
    assert offenders.empty, f"complex_ids in >1 cluster: {offenders.to_dict()}"


def test_cluster_split_has_no_leakage(df):
    """No cluster_id should appear in more than one of train/val/test under
    the leave-complex-out protocol -- this is the core anti-leakage guarantee
    the whole cluster-split protocol exists for."""
    per_cluster_splits = df.groupby("cluster_id")["split_cluster"].nunique()
    offenders = per_cluster_splits[per_cluster_splits > 1]
    assert offenders.empty, f"clusters spanning multiple splits: {offenders.to_dict()}"


def test_cluster_split_covers_all_rows(df):
    assert df["split_cluster"].isin(["train", "val", "test"]).all()


def test_random_split_covers_all_rows(df):
    assert df["split_random"].isin(["train", "val", "test"]).all()


def test_cross_dataset_overlap_flag_matches_cluster_composition(df, clusters):
    flagged = set(df.loc[df["cross_dataset_overlap"], "cluster_id"].astype(str))
    computed_overlap = {
        cid for cid, info in clusters.items() if len(info["datasets"]) > 1
    }
    assert flagged == computed_overlap


def test_cross_dataset_transfer_eval_set_is_leak_free(df):
    """Simulates the cross-dataset transfer protocol: train on SKEMPI2 minus
    overlap clusters, test on AB-Bind minus overlap clusters. No cluster_id
    should appear in both resulting sets."""
    train = df[(df["dataset"] == "skempi2") & (~df["cross_dataset_overlap"])]
    test = df[(df["dataset"] == "abbind") & (~df["cross_dataset_overlap"])]
    assert set(train["cluster_id"]) & set(test["cluster_id"]) == set()
    assert len(train) > 0
    assert len(test) > 0


@pytest.mark.skipif(
    "abdesign" not in pd.read_csv(ALL_MUTATIONS, usecols=["dataset"])["dataset"].values,
    reason="AbDesign not built into all_mutations.csv",
)
def test_abdesign_holdout_is_leak_free(df):
    """The AbDesign zero-shot holdout protocol (replicating the AbDesign
    paper's own evaluation): train_pool must share zero clusters with
    eval_abdesign."""
    assert set(df["split_abdesign_holdout"]) <= {"train_pool", "excluded", "eval_abdesign"}
    train_pool = df[df["split_abdesign_holdout"] == "train_pool"]
    eval_set = df[df["split_abdesign_holdout"] == "eval_abdesign"]
    assert set(train_pool["cluster_id"]) & set(eval_set["cluster_id"]) == set()
    assert len(train_pool) > 0
    assert len(eval_set) == (df["dataset"] == "abdesign").sum()
    # every eval_abdesign row must indeed be an abdesign row, and vice versa
    assert (eval_set["dataset"] == "abdesign").all()
    assert (df.loc[df["dataset"] == "abdesign", "split_abdesign_holdout"] == "eval_abdesign").all()


@pytest.mark.skipif(
    "abdesign" not in pd.read_csv(ALL_MUTATIONS, usecols=["dataset"])["dataset"].values,
    reason="AbDesign not built into all_mutations.csv",
)
def test_overlaps_abdesign_flag_excludes_train_pool_correctly(df):
    non_abdesign = df[df["dataset"] != "abdesign"]
    should_be_excluded = non_abdesign[non_abdesign["overlaps_abdesign"]]
    assert (should_be_excluded["split_abdesign_holdout"] == "excluded").all()
    should_be_train = non_abdesign[~non_abdesign["overlaps_abdesign"]]
    assert (should_be_train["split_abdesign_holdout"] == "train_pool").all()
