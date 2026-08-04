"""Absolute binding-affinity (Kd -> dG) data, for comparison against
PPI-Graphomer (Xie et al., BMC Bioinformatics 2025, 10.1186/s12859-025-06123-2),
which predicts absolute dG for a wildtype complex -- a different task from every
other benchmark in this project (all of which predict ddG, the *change* caused
by a mutation). PPI-Graphomer trains on PDBbind's protein-protein subset, which
requires a manual license/registration step; its external test sets (Kastritis
et al. 2011 benchmark; the PPI-Affinity paper's set) also aren't released as
plain downloadable files by that paper. Two license-free substitutes used
instead, same benchmark lineage:

  TEST:  the "Corrected Benchmark 1.0" affinity table (haddocking/binding
         -affinity-benchmark on GitHub, Apache-2.0, sequences embedded inline)
         -- 46 complexes, itself a corrected/reduced version of the same
         Kastritis-lineage protein-protein affinity benchmark PPI-Graphomer's
         "Test Set 1" is derived from. Not byte-identical to their filtered
         75-sample set, but the same source benchmark, kept fully held-out.

  TRAIN: SKEMPI2's own per-complex *wildtype* Kd (`Affinity_wt (M)`), already
         downloaded/resolved by build_skempi.py -- every SKEMPI2 mutation row
         references one wildtype affinity measurement per complex, so
         deduplicating by complex gives a free absolute-affinity training set
         with no new download. Complexes overlapping the 46-complex test set
         are dropped from training to keep the test genuinely held-out.

dG = R*T*ln(Kd), R = 1.987204e-3 kcal/(mol*K), T = each row's own experimental
temperature (Celsius -> Kelvin; SKEMPI2 rows lacking one default to 298.15K).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "affinity_benchmark"
SKEMPI_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "skempi2" / "skempi_v2.csv"
PPI_GRAPHOMER_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "ppi_graphomer"
BM5_PDB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "bm5_clean" / "PDBs"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

R_KCAL = 1.987204e-3  # kcal / (mol * K)


def kd_to_dg_celsius(kd: float, temperature_c: float) -> float:
    return R_KCAL * (temperature_c + 273.15) * np.log(kd)


def kd_to_dg_kelvin(kd: float, temperature_k: float) -> float:
    return R_KCAL * temperature_k * np.log(kd)


@dataclass
class AffinityExample:
    """No mutation site here -- the complex itself is the unit, so this is a
    much simpler record than dataset.py's Example/MutationSite (those are
    mutation-centric and don't apply to an absolute-affinity task)."""
    complex_id: str
    self_sequence: str
    partner_sequence: str
    dG: float
    meta: dict = field(default_factory=dict)


def _sequences_for_row(row: pd.Series, chain_ids: str) -> str:
    parts = []
    for ch in chain_ids:
        col = f"chain_{ch}"
        val = row.get(col)
        if pd.notna(val) and val:
            parts.append(str(val))
    return "".join(parts)


def build_test_set() -> list[AffinityExample]:
    df = pd.read_csv(RAW_DIR / "protein_protein_affinity_benchmark.csv")
    examples = []
    for _, row in df.iterrows():
        self_seq = _sequences_for_row(row, str(row["chain_id_1"]))
        partner_seq = _sequences_for_row(row, str(row["chain_id_2"]))
        if not self_seq or not partner_seq:
            continue
        dg = kd_to_dg_celsius(float(row["kd"]), float(row["temperature"]))
        examples.append(AffinityExample(
            complex_id=str(row["pdb_id"]).upper(), self_sequence=self_seq,
            partner_sequence=partner_seq, dG=dg,
            meta={"kd": float(row["kd"]), "source": "affinity_benchmark_1.0"},
        ))
    return examples


def _skempi2_wt_examples(only_ids: set[str] | None = None) -> dict[str, AffinityExample]:
    """Every SKEMPI2 complex with a valid wildtype affinity, keyed by
    upper-cased complex_id. `only_ids`, if given, restricts to that set
    (used to pull out specific PDB IDs matching an external test list)."""
    only = {c.upper() for c in only_ids} if only_ids is not None else None

    raw = pd.read_csv(SKEMPI_RAW, sep=";")
    raw["pdb"] = raw["#Pdb"].str.split("_").str[0].str.upper()
    raw["Affinity_wt_parsed"] = pd.to_numeric(raw["Affinity_wt_parsed"], errors="coerce")
    wt = raw.dropna(subset=["Affinity_wt_parsed"]).drop_duplicates(subset="pdb")[
        ["pdb", "Affinity_wt_parsed", "Temperature"]
    ]
    wt_by_pdb = wt.set_index("pdb").to_dict(orient="index")

    mutations_df = pd.read_csv(PROC_DIR / "skempi2_mutations.csv")
    complexes = mutations_df.drop_duplicates(subset="complex_id")[
        ["complex_id", "group_a_chains", "group_b_chains"]
    ]

    with open(PROC_DIR / "skempi2_sequences.json") as f:
        sequences = json.load(f)

    examples: dict[str, AffinityExample] = {}
    for _, row in complexes.iterrows():
        complex_id = row["complex_id"]
        if only is not None and complex_id.upper() not in only:
            continue
        wt_row = wt_by_pdb.get(complex_id.upper())
        if wt_row is None:
            continue
        temp_raw = wt_row.get("Temperature")
        try:
            # SKEMPI2's Temperature column is already in Kelvin (e.g. "298",
            # sometimes "298(assumed)"); default to 298.15K (25C) when missing,
            # matching the rest of this project's convention (build_rde_skempi.py).
            digits = "".join(ch for ch in str(temp_raw) if ch.isdigit() or ch == ".")
            temp_k = float(digits) if pd.notna(temp_raw) and digits else 298.15
        except ValueError:
            temp_k = 298.15

        raw_chains = sequences.get(complex_id, {})
        complex_seqs = {
            c: (info if isinstance(info, str) else info["sequence"])
            for c, info in raw_chains.items()
        }
        group_a = str(row["group_a_chains"]).split(",")
        group_b = str(row["group_b_chains"]).split(",")
        self_seq = "".join(complex_seqs.get(c, "") for c in group_a)
        partner_seq = "".join(complex_seqs.get(c, "") for c in group_b)
        if not self_seq or not partner_seq:
            continue

        dg = kd_to_dg_kelvin(float(wt_row["Affinity_wt_parsed"]), temp_k)
        examples[complex_id.upper()] = AffinityExample(
            complex_id=complex_id, self_sequence=self_seq, partner_sequence=partner_seq,
            dG=dg, meta={"kd": float(wt_row["Affinity_wt_parsed"]), "source": "skempi2_wt"},
        )
    return examples


def build_skempi2_wt_train_set(exclude_complex_ids: set[str] | None = None) -> list[AffinityExample]:
    exclude = {c.upper() for c in (exclude_complex_ids or set())}
    all_examples = _skempi2_wt_examples()
    return [ex for cid, ex in all_examples.items() if cid not in exclude]


def build_prott_affinity_test1_reproduction() -> list[AffinityExample]:
    """Partial reproduction of ProtT-Affinity's (Lou, arXiv:2511.16113, Nov
    2025) "Test set 1" -- 79 PDB complexes from the same Kastritis et al. 2011
    lineage, reused by ProAffinity-GNN (Zhou et al. 2024, JCIM) and several
    other recent baselines they benchmark against in one table. Their repo
    (github.com/fufuhonghong/ProtT-Affinity) ships only the PDB ID list, not
    sequences/affinities (still PDBbind-gated) -- but 50 of their 79 IDs are
    independently covered by data this project already has license-free
    (25 via the "Corrected Benchmark 1.0" CSV, the rest via SKEMPI2's own
    wildtype affinities), so this reconstructs those 50 rather than skipping
    the comparison. Prefers the benchmark-CSV version when an ID is in both
    (it's curated specifically for this exact benchmark's chain grouping).
    The remaining 29 IDs need PDBbind and are left out -- a real, if partial,
    reproduction rather than a claim of exact replication.
    """
    with open(RAW_DIR / "prott_affinity_test_pdbid.txt") as f:
        lines = [line.strip() for line in f if line.strip()]
    sep_idx = lines.index("test2:")
    test1_ids = {pid.upper() for pid in lines[1:sep_idx - 1]}

    from_benchmark = {ex.complex_id: ex for ex in build_test_set() if ex.complex_id in test1_ids}
    remaining = test1_ids - set(from_benchmark)
    from_skempi = _skempi2_wt_examples(only_ids=remaining)

    examples = list(from_benchmark.values()) + list(from_skempi.values())
    return examples


# The 42 of ProtT-Affinity/PPI-Graphomer's 79-ID "Test Set 1" not covered by
# the Corrected Benchmark 1.0 CSV, but present as HADDOCK-ready bound-matched
# receptor/ligand structures in github.com/haddocking/BM5-clean (Vreven et
# al.'s Docking Benchmark 5 lineage -- public, not PDBbind-gated). Downloaded
# once into data/raw/bm5_clean/PDBs/{ID}_r.pdb / {ID}_l.pdb (receptor/ligand,
# bound-matched numbering). Fixed list, not recomputed at runtime, since it
# reflects a one-time reconciliation against the exact 79-ID test1 list.
BM5_COVERED_IDS = {
    "1ACB", "1ATN", "1BUH", "1BVK", "1DQJ", "1E4K", "1EER", "1EFN", "1EWY",
    "1FC2", "1FFW", "1FSK", "1GLA", "1GXD", "1I2M", "1JTG", "1JWH", "1K5D",
    "1KAC", "1KKL", "1KTZ", "1LFD", "1MLC", "1MQ8", "1PPE", "1PVH", "1R0R",
    "1US7", "1VFB", "1WEJ", "1XQS", "1XU1", "1ZHI", "2A9K", "2ABZ", "2FJU",
    "2HQS", "2HRK", "2PCC", "2VIS", "3CPH", "4CPA",
}

# Other-method columns in the eLife supplementary table (Moal & Fernandez-
# Recio's CCharPPI-adjacent benchmark table, elife-07454-supp4-v4.csv) --
# real, published, per-complex predictions on these exact 79 rows, not just
# an aggregate reported number. `irsmd` and `ICs/NIS-based` are excluded (not
# affinity predictors -- structural/interface descriptors from the docking
# literature, included in the source table for a different purpose).
PPI_GRAPHOMER_BASELINE_COLUMNS = [
    "Kastritis2014", "Horton-Lewis", "Moal2011", "DFIRE", "PMF", "ZRANK",
    "ZRANK2", "ROSETTADOCK", "PYDOCK_TOT", "SIPPER", "AP_PISA", "FIREDOCK",
    "FIREDOCK_AB", "FIREDOCK_EI", "CP_PIE",
]


def _bm5_clean_example(pdb_id: str) -> AffinityExample | None:
    from simba.data.pdb_utils import parse_pdb_chains

    r_path = BM5_PDB_DIR / f"{pdb_id}_r.pdb"
    l_path = BM5_PDB_DIR / f"{pdb_id}_l.pdb"
    if not r_path.exists() or not l_path.exists():
        return None
    receptor_chains = parse_pdb_chains(r_path)
    ligand_chains = parse_pdb_chains(l_path)
    self_seq = "".join(cs.sequence for cs in receptor_chains.values())
    partner_seq = "".join(cs.sequence for cs in ligand_chains.values())
    if not self_seq or not partner_seq:
        return None
    return AffinityExample(
        complex_id=pdb_id, self_sequence=self_seq, partner_sequence=partner_seq,
        dG=float("nan"),  # filled in by the caller from the eLife table
        meta={"source": "bm5_clean"},
    )


def build_ppi_graphomer_test1_reproduction() -> list[AffinityExample]:
    """Fuller reproduction of PPI-Graphomer's (Xie et al., BMC Bioinformatics
    2025) "Test Set 1" (79 samples before their own filtering to 75) than
    `build_prott_affinity_test1_reproduction` above, using the exact same
    79-PDB-ID list (`prott_affinity_test_pdbid.txt`) but a different -- and
    now complete -- affinity/baseline source: `elife-07454-supp4-v4.csv`
    (downloaded from PPI-Graphomer's own public repo,
    github.com/xiebaoshu058/PPI-Graphomer/blob/main/data/), whose 79 PDB IDs
    are a byte-for-byte match to our test1 list. This is very likely the
    exact "[46]"-referenced table PPI-Graphomer's own paper describes as
    already summarizing "additional methods' performances on this dataset".

    Combines three sources per complex:
      - `Binding_affinity` and 15 other real per-complex baseline method
        scores (DFIRE, PMF, ZRANK, ZRANK2, ROSETTADOCK, PYDOCK_TOT, SIPPER,
        AP_PISA, FIREDOCK x3, CP_PIE, ...) from the eLife table -- for ALL 79
        IDs, resolving the affinity-label gap that limited the older
        ProtT-Affinity-based reproduction to 50/79.
      - Sequences: 24 IDs from the Corrected Benchmark 1.0 CSV (curated,
        already-verified chain grouping), 42 more from BM5-clean's public
        bound-matched receptor/ligand PDB structures (Vreven et al.'s Docking
        Benchmark 5 lineage -- not PDBbind-gated). Receptor -> self_sequence,
        ligand -> partner_sequence; this project's self/partner assignment is
        already arbitrary per training source file (see build_test_set's
        chain_id_1/chain_id_2 vs. SKEMPI2's group_a/group_b), so a fixed
        receptor=self convention introduces no new bias.
      - The remaining 13 IDs have neither a public structure source
        (Corrected Benchmark 1.0 nor BM5-clean) and are left unresolved.

    Net: 66/79 complexes recovered (up from 50/79), each carrying real
    per-complex predictions from 15 other methods for a genuinely row-matched
    comparison, not just PPI-Graphomer's own reported aggregate PCC.
    """
    with open(RAW_DIR / "prott_affinity_test_pdbid.txt") as f:
        lines = [line.strip() for line in f if line.strip()]
    sep_idx = lines.index("test2:")
    test1_ids = {pid.upper() for pid in lines[1:sep_idx - 1]}

    baseline_df = pd.read_csv(PPI_GRAPHOMER_RAW / "elife-07454-supp4-test1.csv")
    baseline_df["pdb_id"] = baseline_df["PDB"].str.replace(".pdb", "", regex=False).str.upper()
    baseline_by_id = baseline_df.set_index("pdb_id").to_dict(orient="index")

    from_benchmark = {ex.complex_id: ex for ex in build_test_set() if ex.complex_id in test1_ids}

    examples = []
    for pdb_id in sorted(test1_ids):
        row = baseline_by_id.get(pdb_id)
        if row is None:
            continue
        if pdb_id in from_benchmark:
            base = from_benchmark[pdb_id]
            self_seq, partner_seq = base.self_sequence, base.partner_sequence
            source = "affinity_benchmark_1.0"
        elif pdb_id in BM5_COVERED_IDS:
            ex = _bm5_clean_example(pdb_id)
            if ex is None:
                continue
            self_seq, partner_seq = ex.self_sequence, ex.partner_sequence
            source = "bm5_clean"
        else:
            continue  # one of the 13 IDs with no public structure source

        examples.append(AffinityExample(
            complex_id=pdb_id, self_sequence=self_seq, partner_sequence=partner_seq,
            dG=float(row["Binding_affinity"]),
            meta={"source": source, "baselines": {c: row[c] for c in PPI_GRAPHOMER_BASELINE_COLUMNS
                                                   if c in row and pd.notna(row[c])}},
        ))
    return examples


if __name__ == "__main__":
    test = build_test_set()
    print(f"[affinity benchmark test] {len(test)} complexes")
    test_ids = {e.complex_id for e in test}
    train = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
    print(f"[skempi2 wt-affinity train] {len(train)} complexes "
          f"(excluded {len(test_ids)} test-overlapping ids)")
    print("test dG range:", min(e.dG for e in test), max(e.dG for e in test))
    print("train dG range:", min(e.dG for e in train), max(e.dG for e in train))

    test1 = build_prott_affinity_test1_reproduction()
    print(f"\n[ProtT-Affinity test1 reproduction] {len(test1)}/79 complexes recovered")
    print("test1 dG range:", min(e.dG for e in test1), max(e.dG for e in test1))
