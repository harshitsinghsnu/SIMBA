"""Cluster complexes across datasets and assign split protocols:

  1. random             -- stratified random 80/10/10 over individual mutation rows
  2. cluster             -- leave-complex-out: whole sequence clusters assigned to
                             train/val/test so no cluster spans multiple splits
  3. cross_dataset_overlap -- flags rows whose cluster is shared across >=2 of
                             {skempi2, abbind, abdesign}, for any zero-shot
                             transfer evaluation that needs to exclude
                             contaminated clusters
  4. abdesign_holdout    -- replicates the AbDesign paper's own zero-shot
                             evaluation protocol: train only on SKEMPI2/AB-Bind
                             rows whose cluster shares nothing with AbDesign's
                             47 complexes ("train_pool"), evaluate untouched on
                             AbDesign ("eval_abdesign"); anything else is
                             "excluded" to keep the comparison leak-free.

Output: data/processed/all_mutations.csv (skempi2 + abbind + abdesign rows,
unified schema, with cluster_id / split_random / split_cluster /
cross_dataset_overlap / overlaps_abdesign / split_abdesign_holdout columns)
and data/processed/clusters.json (cluster_id -> member complex_ids and which
dataset(s) they come from).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from simba.data.clustering import complex_representative_kmers, greedy_cluster

PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SEED = 0
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def load_chain_seqs(seq_json_path: Path) -> dict[str, dict[str, str]]:
    """Load a {complex_id: {chain: sequence}} cache. Tolerates either a plain
    chain->sequence string mapping (AbDesign) or the richer
    chain->{"sequence": ..., "resnums": ...} mapping (SKEMPI2/AB-Bind)."""
    raw = json.loads(seq_json_path.read_text())
    return {
        complex_id: {
            chain: (info if isinstance(info, str) else info["sequence"])
            for chain, info in chains.items()
        }
        for complex_id, chains in raw.items()
    }


def stratified_random_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    rng = random.Random(seed)
    split = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby("dataset"):
        idx = list(group.index)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * SPLIT_FRACS["train"]))
        n_val = int(round(n * SPLIT_FRACS["val"]))
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:n_train + n_val]] = "val"
        split.loc[idx[n_train + n_val:]] = "test"
    return split


def balanced_cluster_split(
    cluster_row_counts: dict[int, int], seed: int = SEED
) -> dict[int, str]:
    """Greedy balanced multiway partition of clusters into train/val/test by
    row count, so no cluster is split across sets."""
    rng = random.Random(seed)
    clusters = list(cluster_row_counts.items())
    rng.shuffle(clusters)
    clusters.sort(key=lambda kv: kv[1], reverse=True)

    total = sum(cluster_row_counts.values())
    targets = {name: frac * total for name, frac in SPLIT_FRACS.items()}
    filled = {name: 0 for name in SPLIT_FRACS}
    assignment: dict[int, str] = {}

    for cluster_id, count in clusters:
        # assign to whichever split is furthest below its target (as a fraction)
        deficits = {
            name: targets[name] - filled[name] for name in SPLIT_FRACS
        }
        best = max(deficits, key=deficits.get)
        assignment[cluster_id] = best
        filled[best] += count
    return assignment


def main() -> None:
    skempi_df = pd.read_csv(PROC_DIR / "skempi2_mutations.csv")
    abbind_df = pd.read_csv(PROC_DIR / "abbind_mutations.csv")
    frames = [skempi_df, abbind_df]

    abdesign_path = PROC_DIR / "abdesign_mutations.csv"
    have_abdesign = abdesign_path.exists()
    if have_abdesign:
        frames.append(pd.read_csv(abdesign_path))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["resolved"]].reset_index(drop=True)

    seq_sources = [("skempi2", PROC_DIR / "skempi2_sequences.json"),
                   ("abbind", PROC_DIR / "abbind_sequences.json")]
    if have_abdesign:
        seq_sources.append(("abdesign", PROC_DIR / "abdesign_sequences.json"))

    complex_datasets: dict[str, set[str]] = {}
    merged_chains: dict[str, dict[str, str]] = {}
    for source_name, path in seq_sources:
        for complex_id, chains in load_chain_seqs(path).items():
            merged_chains.setdefault(complex_id, {}).update(chains)
            complex_datasets.setdefault(complex_id, set()).add(source_name)

    complex_kmers = {
        cid: complex_representative_kmers(chains) for cid, chains in merged_chains.items()
    }
    cluster_of = greedy_cluster(complex_kmers)

    clusters: dict[int, dict] = {}
    for cid, cluster_id in cluster_of.items():
        entry = clusters.setdefault(cluster_id, {"members": [], "datasets": set()})
        entry["members"].append(cid)
        entry["datasets"] |= complex_datasets[cid]
    cross_overlap_clusters = {
        cid for cid, info in clusters.items() if len(info["datasets"]) > 1
    }

    abdesign_overlap_clusters = {
        cid for cid, info in clusters.items()
        if "abdesign" in info["datasets"] and len(info["datasets"]) > 1
    }

    df["cluster_id"] = df["complex_id"].map(cluster_of)
    df["cross_dataset_overlap"] = df["cluster_id"].isin(cross_overlap_clusters)
    df["overlaps_abdesign"] = df["cluster_id"].isin(abdesign_overlap_clusters)

    df["split_random"] = stratified_random_split(df)

    row_counts = df.groupby("cluster_id").size().to_dict()
    cluster_split_assignment = balanced_cluster_split(row_counts)
    df["split_cluster"] = df["cluster_id"].map(cluster_split_assignment)

    if have_abdesign:
        # Zero-shot holdout protocol matching the AbDesign paper's own evaluation:
        # train only on SKEMPI2/AB-Bind complexes that share no cluster with any
        # of AbDesign's 47 complexes, then evaluate on AbDesign untouched.
        is_abdesign = df["dataset"] == "abdesign"
        train_eligible = (~is_abdesign) & (~df["overlaps_abdesign"])
        split_abdesign_holdout = pd.Series("excluded", index=df.index)
        split_abdesign_holdout[train_eligible] = "train_pool"
        split_abdesign_holdout[is_abdesign] = "eval_abdesign"
        df["split_abdesign_holdout"] = split_abdesign_holdout

    df.to_csv(PROC_DIR / "all_mutations.csv", index=False)
    with open(PROC_DIR / "clusters.json", "w") as f:
        json.dump(
            {
                str(cid): {"members": info["members"], "datasets": sorted(info["datasets"])}
                for cid, info in clusters.items()
            },
            f,
            indent=2,
        )

    print(f"total resolved rows:        {len(df)}")
    print(f"unique complexes:           {len(complex_kmers)}")
    print(f"clusters:                   {len(clusters)}")
    print(f"cross-dataset-overlap clusters: {len(cross_overlap_clusters)} "
          f"({df['cross_dataset_overlap'].sum()} rows)")
    print("\nsplit_random counts:\n", df["split_random"].value_counts())
    print("\nsplit_cluster counts:\n", df["split_cluster"].value_counts())
    print("\nsplit_cluster counts by dataset:\n",
          df.groupby(["dataset", "split_cluster"]).size())
    if have_abdesign:
        print(f"\nabdesign-overlap clusters:   {len(abdesign_overlap_clusters)} "
              f"({df['overlaps_abdesign'].sum()} rows excluded from train_pool)")
        print("\nsplit_abdesign_holdout counts:\n", df["split_abdesign_holdout"].value_counts())


if __name__ == "__main__":
    main()
