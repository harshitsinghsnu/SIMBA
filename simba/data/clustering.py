"""Sequence-identity-style clustering of complexes, without external tools.

No mmseqs2/cd-hit is available in this environment, so complexes are grouped
by greedy single-linkage clustering on k-mer Jaccard similarity of their full
(all-chain) sequence content. This is coarser than a true alignment-based
clustering but is self-contained, fast at this dataset scale (a few hundred
complexes), and sufficient to catch near-duplicate/homologous complexes
(e.g. the same antibody or antigen reused across PDB entries) so they don't
leak across train/val/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

K = 5
JACCARD_THRESHOLD = 0.5


def kmers(seq: str, k: int = K) -> frozenset[str]:
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def complex_representative_kmers(chain_seqs: dict[str, str], k: int = K) -> frozenset[str]:
    """Union of k-mers over all chains of a complex (order-independent)."""
    all_kmers: set[str] = set()
    for seq in chain_seqs.values():
        all_kmers |= kmers(seq, k)
    return frozenset(all_kmers)


@dataclass
class Cluster:
    members: list[str] = field(default_factory=list)
    kmer_union: frozenset[str] = frozenset()


def greedy_cluster(
    complex_kmers: dict[str, frozenset[str]], threshold: float = JACCARD_THRESHOLD
) -> dict[str, int]:
    """Greedy single-linkage clustering. Returns complex_id -> cluster_id."""
    clusters: list[Cluster] = []
    assignment: dict[str, int] = {}

    for complex_id, kset in complex_kmers.items():
        best_idx, best_sim = None, 0.0
        for idx, cluster in enumerate(clusters):
            sim = jaccard(kset, cluster.kmer_union)
            if sim > best_sim:
                best_sim, best_idx = sim, idx
        if best_idx is not None and best_sim >= threshold:
            clusters[best_idx].members.append(complex_id)
            clusters[best_idx].kmer_union = clusters[best_idx].kmer_union | kset
            assignment[complex_id] = best_idx
        else:
            clusters.append(Cluster(members=[complex_id], kmer_union=kset))
            assignment[complex_id] = len(clusters) - 1

    return assignment
