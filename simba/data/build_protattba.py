"""Ingest ProtAttBA's own released benchmark CSVs + cross-validation folds and
predictions (github.com/code4luck/ProtAttBA), our closest sequence-only peer.

Why this matters: ProtAttBA ships the *exact* wildtype/mutant sequences it
trained on (no PDB parsing needed here either), the exact 10-fold (AB645,
S1131) / 5-fold (AB1101) row assignment it used, and its own raw predictions
per held-out row. That lets us evaluate SIMBA on literally the same rows in
the same folds and diff against ProtAttBA's real predictions -- the strongest
possible apples-to-apples comparison against a sequence-only peer, no
re-running of their code required.

Resolving each mutation site's exact sequence position: the raw mutation
notation (e.g. "D:A488G") uses the *original PDB* chain letter and numbering,
which doesn't correspond to ProtAttBA's generic per-row sequence columns
(antibody_heavy_seq / antibody_light_seq / antigen_a_seq / antigen_b_seq) or
their numbering. Since we have both the wildtype and mutant sequence for
every role directly, we sidestep that entirely: diff each role's wildtype vs
mutant sequence to find the exact substituted position(s), then match each
parsed mutation site to a diff position by residue identity (wt_aa/mut_aa
must agree) rather than trusting the notation's chain letter or number.

Output per benchmark (AB645, AB1101, S1131):
  data/processed/protattba_<name>_mutations.csv  -- unified schema + `fold`
    + `protattba_prediction` columns
  data/processed/protattba_<name>_sequences.json -- {complex_id: {role: wt_seq}}
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from simba.data.mutations import parse_concatenated_abbind_mutation_string

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "protattba" / "cross_validation"
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Per-benchmark schema: ProtAttBA's three CSVs don't share column names.
BENCHMARK_SCHEMAS = {
    "AB645": {
        "mutation_col": "Mutation",
        "chain_seq_cols": {
            "L": ("antibody_light_seq", "antibody_light_seq_mut"),
            "H": ("antibody_heavy_seq", "antibody_heavy_seq_mut"),
            "A": ("antigen_a_seq", "antigen_a_seq_mut"),
            "B": ("antigen_b_seq", "antigen_b_seq_mut"),
        },
        "group_a_chains": "L,H", "group_b_chains": "A,B",
    },
    "AB1101": {
        "mutation_col": "Mutation",
        "chain_seq_cols": {
            "L": ("antibody_light_seq", "antibody_light_seq_mut"),
            "H": ("antibody_heavy_seq", "antibody_heavy_seq_mut"),
            "A": ("antigen_a_seq", "antigen_a_seq_mut"),
            "B": ("antigen_b_seq", "antigen_b_seq_mut"),
        },
        "group_a_chains": "L,H", "group_b_chains": "A,B",
    },
    "S1131": {
        "mutation_col": "mutation",
        "chain_seq_cols": {"A": ("a", "a_mut"), "B": ("b", "b_mut")},
        "group_a_chains": "A", "group_b_chains": "B",
    },
}


def _role_diff_positions(row, chain_seq_cols: dict) -> dict[str, list[int]]:
    """For each role whose wt/mut sequence differ (same length), the list of
    0-indexed positions where they differ."""
    role_diffs = {}
    for role, (wt_col, mut_col) in chain_seq_cols.items():
        wt_seq, mut_seq = row.get(wt_col), row.get(mut_col)
        if not (isinstance(wt_seq, str) and isinstance(mut_seq, str)):
            continue
        if wt_seq == mut_seq or len(wt_seq) != len(mut_seq):
            continue
        role_diffs[role] = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
    return role_diffs


def _assign_sites_to_roles(sites, row, chain_seq_cols: dict):
    """Match each parsed mutation site to (role, seq_index) by diffing that
    role's wildtype/mutant sequence and requiring residue-identity agreement.
    Returns None if any site can't be unambiguously resolved this way."""
    role_diffs = _role_diff_positions(row, chain_seq_cols)

    groups = OrderedDict()
    for s in sites:
        groups.setdefault(s.chain, []).append(s)

    assignment: dict[int, tuple[str, int]] = {}
    used_roles = set()
    for _letter, group_sites in groups.items():
        chosen_role, chosen_mapping = None, None
        for role, diffs in role_diffs.items():
            if role in used_roles or len(diffs) != len(group_sites):
                continue
            wt_seq = row[chain_seq_cols[role][0]]
            mut_seq = row[chain_seq_cols[role][1]]
            remaining = list(diffs)
            mapping = []
            ok = True
            for s in group_sites:
                match = next((p for p in remaining if wt_seq[p] == s.wt_aa and mut_seq[p] == s.mut_aa), None)
                if match is None:
                    ok = False
                    break
                mapping.append(match)
                remaining.remove(match)
            if ok:
                chosen_role, chosen_mapping = role, mapping
                break
        if chosen_role is None:
            return None
        used_roles.add(chosen_role)
        for s, pos in zip(group_sites, chosen_mapping):
            assignment[id(s)] = (chosen_role, pos)

    return [assignment[id(s)] for s in sites]


def build_one(name: str) -> None:
    schema = BENCHMARK_SCHEMAS[name]
    data = pd.read_csv(RAW_DIR / "data" / "csv" / f"{name}.csv")
    results = pd.read_csv(RAW_DIR / "results" / f"{name}_results.csv")
    pred_by_index = results.set_index("test_indices")[["fold", "prediction"]]

    seq_cache: dict[str, dict[str, str]] = {}
    rows = []
    n_unresolved = 0

    for row_idx, row in data.iterrows():
        complex_id = f"{name}_{row_idx}"  # each row can have distinct per-chain
                                           # mutant sequences even for the same
                                           # PDB, so key sequences per-row
        chains: dict[str, str] = {}
        for role, (wt_col, _mut_col) in schema["chain_seq_cols"].items():
            wt_seq = row.get(wt_col)
            if isinstance(wt_seq, str) and wt_seq:
                chains[role] = wt_seq
        seq_cache[complex_id] = chains

        try:
            sites = parse_concatenated_abbind_mutation_string(str(row[schema["mutation_col"]]))
        except ValueError:
            sites = []

        role_assignments = _assign_sites_to_roles(sites, row, schema["chain_seq_cols"]) if sites else None
        resolved = role_assignments is not None
        if not resolved:
            n_unresolved += 1
            mutation_records = [
                {"chain": s.chain, "wt_aa": s.wt_aa, "resnum": s.resnum,
                 "icode": s.icode, "mut_aa": s.mut_aa, "seq_index": None}
                for s in sites
            ]
        else:
            mutation_records = [
                {"chain": role, "wt_aa": s.wt_aa, "resnum": s.resnum,
                 "icode": s.icode, "mut_aa": s.mut_aa, "seq_index": pos}
                for s, (role, pos) in zip(sites, role_assignments)
            ]

        pred_row = pred_by_index.loc[row_idx] if row_idx in pred_by_index.index else None

        rows.append(
            {
                "dataset": "protattba",
                "benchmark": name,
                "complex_id": complex_id,
                "pdb_id": row.get("PDB"),
                "group_a_chains": schema["group_a_chains"],
                "group_b_chains": schema["group_b_chains"],
                "mutations": json.dumps(mutation_records),
                "n_mutations": len(sites),
                "ddG": float(row["ddG"]) if pd.notna(row.get("ddG")) else None,
                "resolved": resolved and len(sites) > 0,
                "fold": int(pred_row["fold"]) if pred_row is not None else None,
                "protattba_prediction": float(pred_row["prediction"]) if pred_row is not None else None,
            }
        )

    out_df = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(PROC_DIR / f"protattba_{name.lower()}_mutations.csv", index=False)
    with open(PROC_DIR / f"protattba_{name.lower()}_sequences.json", "w") as f:
        json.dump(seq_cache, f)

    n_resolved = int(out_df["resolved"].sum())
    print(f"[{name}] rows={len(out_df)} resolved={n_resolved} "
          f"({100 * n_resolved / max(len(out_df), 1):.1f}%) folds={out_df['fold'].nunique()}")


def main() -> None:
    for name in BENCHMARK_SCHEMAS:
        build_one(name)


if __name__ == "__main__":
    main()
