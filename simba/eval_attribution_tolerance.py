"""Re-scores every already-computed Integrated-Gradients case study
(1BRS, 1CSE, and the 5 named complexes) with a relaxed "near-miss" overlap
criterion: an attributed residue counts as a hit if it is *itself* within
the true-interface distance cutoff, OR if its immediate sequence neighbor
(index-1 or index+1) is -- since ESM2's embeddings are contextual, a
model's attention can land one residue off a true contact and still be
capturing real signal, especially where local sequence-adjacent residues
are also structurally close (loops, short turns). No new IG computation is
needed -- this reads the attribution/distance arrays already saved to
`figures/ig_*_attribution.json` by the earlier interpretability scripts.

Reports exact-match precision (as originally computed) side by side with
tolerance-relaxed precision, for every case and side, so the effect of the
relaxation is visible rather than just asserted.
"""

from __future__ import annotations

import json

CASES = [
    ("1brs", "Barnase-Barstar (1BRS, held out)", "topk"),
    ("1cse", "Eglin c-Subtilisin (1CSE, in training)", "threshold"),
    ("ras_raf", "Ras-Raf (4G0N, in training)", "threshold"),
    ("trypsin_bpti", "Trypsin-BPTI (3BTD, in training)", "threshold"),
    ("colicin_im9", "Colicin E9-Im9 (1EMV, in training)", "threshold"),
    ("fab2d1_1918ha", "Fab 2D1-1918 H1N1 HA (3LZF, in training)", "threshold"),
    ("hyhel10_lysozyme", "HyHEL-10-Lysozyme (3HFM, in training)", "threshold"),
]


def tolerant_precision(selected_idx, dist, cutoff, tolerance=1):
    n = len(dist)
    true_iface = set(i for i, d in enumerate(dist) if d <= cutoff)
    if not selected_idx:
        return None, 0
    hits = 0
    for i in selected_idx:
        window = range(max(0, i - tolerance), min(n, i + tolerance + 1))
        if any(w in true_iface for w in window):
            hits += 1
    return hits / len(selected_idx), len(selected_idx)


def selected_indices(d, side, mode):
    if mode == "topk":
        attr = d[f"{side}_attribution"]
        k = d[f"{side}_precision_k"]
        order = sorted(range(len(attr)), key=lambda i: -attr[i])
        return order[:k]
    else:
        attr_norm = d[f"{side}_attribution_norm"]
        threshold = d["attribution_threshold"]
        return [i for i, v in enumerate(attr_norm) if v > threshold]


def main():
    results = []
    for case_name, label, mode in CASES:
        with open(f"figures/ig_{case_name}_attribution.json") as f:
            d = json.load(f)
        cutoff = d["interface_cutoff_angstrom"]

        row = {"case": case_name, "label": label}
        for side, dist_key in [("self", "self_distance_to_partner"), ("partner", "partner_distance_to_self")]:
            idx = selected_indices(d, side, mode)
            dist = d[dist_key]
            exact_key = f"{side}_precision_at_k" if mode == "topk" else f"{side}_precision_above_threshold"
            exact_prec = d.get(exact_key)
            tol_prec, n_sel = tolerant_precision(idx, dist, cutoff, tolerance=1)
            n_iface = sum(1 for x in dist if x <= cutoff)
            random_baseline = n_iface / len(dist)
            # tolerant random baseline: chance a random residue's +-1 window touches the interface
            tol_random = sum(
                1 for i in range(len(dist))
                if any(dist[w] <= cutoff for w in range(max(0, i - 1), min(len(dist), i + 2)))
            ) / len(dist)
            row[side] = {
                "n_selected": n_sel, "exact_precision": exact_prec, "tolerant_precision": tol_prec,
                "random_baseline": random_baseline, "tolerant_random_baseline": tol_random,
            }
        results.append(row)

        print(f"\n{label}")
        for side in ("self", "partner"):
            r = row[side]
            exact = r["exact_precision"]
            tol = r["tolerant_precision"]
            print(f"  {side}: n={r['n_selected']}  exact={exact}  "
                  f"+/-1={tol}  (random exact={r['random_baseline']:.3f}, random +/-1={r['tolerant_random_baseline']:.3f})")

    with open("figures/attribution_tolerance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved figures/attribution_tolerance_results.json")


if __name__ == "__main__":
    main()
