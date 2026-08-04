"""Maps the 1BRS Integrated Gradients attribution (`interpretability_1brs.py`)
directly onto the real 3D structure via the pymol2 Python API, and renders
comparison images: attribution-colored cartoon (what the model weighted
most heavily) side by side with the real, distance-based interface
definition (what is structurally true), so the two can be inspected
visually rather than only via the summary correlation number.
"""

from __future__ import annotations

import json

import pymol2

IN_JSON = "figures/ig_1brs_attribution.json"
PDB_PATH = "data/raw/skempi2/PDBs/1BRS.pdb"
SELF_CHAIN, PARTNER_CHAIN = "A", "D"


def resi_selector(resnums) -> str:
    tokens = []
    for resnum, icode in resnums:
        icode = (icode or "").strip()
        tokens.append(f"{resnum}{icode}")
    return "+".join(tokens)


def set_bfactors(cmd, chain: str, resnums, values) -> None:
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    for (resnum, icode), v in zip(resnums, values):
        icode = (icode or "").strip()
        resi = f"{resnum}{icode}"
        norm = (v - lo) / span
        cmd.alter(f"chain {chain} and resi {resi}", f"b={norm}")


def main():
    with open(IN_JSON) as f:
        d = json.load(f)

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(PDB_PATH, "cplx")
        cmd.hide("everything")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.set("cartoon_transparency", 0.0)
        cmd.show("cartoon")
        cmd.color("grey70", "cplx")

        set_bfactors(cmd, SELF_CHAIN, d["self_resnums"], d["self_attribution"])
        set_bfactors(cmd, PARTNER_CHAIN, d["partner_resnums"], d["partner_attribution"])

        # --- Panel 1: attribution-colored cartoon ---
        # Explicit minimum/maximum: some atoms in these chains (waters,
        # alternate conformers) are never touched by set_bfactors' per
        # -residue alter() above and keep their original crystallographic
        # B-factor (up to ~70), which would otherwise dominate spectrum's
        # auto-ranging and compress every real 0-1 attribution value into a
        # near-identical sliver of the color scale.
        cmd.spectrum("b", "blue_white_red", f"polymer and (chain {SELF_CHAIN} or chain {PARTNER_CHAIN})",
                      minimum=0.0, maximum=1.0)
        self_k = max(1, int(round(0.15 * len(d["self_sequence"]))))
        partner_k = max(1, int(round(0.15 * len(d["partner_sequence"]))))
        top_self = sorted(range(len(d["self_attribution"])),
                           key=lambda i: -d["self_attribution"][i])[:self_k]
        top_partner = sorted(range(len(d["partner_attribution"])),
                              key=lambda i: -d["partner_attribution"][i])[:partner_k]
        top_self_resi = resi_selector([d["self_resnums"][i] for i in top_self])
        top_partner_resi = resi_selector([d["partner_resnums"][i] for i in top_partner])
        cmd.show("sticks", f"(chain {SELF_CHAIN} and resi {top_self_resi}) and not name C+N+O")
        cmd.show("sticks", f"(chain {PARTNER_CHAIN} and resi {top_partner_resi}) and not name C+N+O")
        cmd.orient("cplx")
        cmd.ray(1400, 1050)
        cmd.png("figures/ig_1brs_structure_attribution.png", dpi=200)

        # --- Panel 2: ground-truth interface (distance <= cutoff) ---
        cmd.hide("sticks")
        cmd.color("grey70", "cplx")
        cutoff = d["interface_cutoff_angstrom"]
        self_iface = [d["self_resnums"][i] for i, dist in enumerate(d["self_distance_to_partner"]) if dist <= cutoff]
        partner_iface = [d["partner_resnums"][i] for i, dist in enumerate(d["partner_distance_to_self"]) if dist <= cutoff]
        self_iface_resi = resi_selector(self_iface)
        partner_iface_resi = resi_selector(partner_iface)
        cmd.color("forest", f"chain {SELF_CHAIN} and resi {self_iface_resi}")
        cmd.color("forest", f"chain {PARTNER_CHAIN} and resi {partner_iface_resi}")
        cmd.orient("cplx")
        cmd.ray(1400, 1050)
        cmd.png("figures/ig_1brs_structure_groundtruth.png", dpi=200)

        print("saved figures/ig_1brs_structure_attribution.png")
        print("saved figures/ig_1brs_structure_groundtruth.png")


if __name__ == "__main__":
    main()
