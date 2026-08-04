"""Maps the 1CSE Integrated Gradients attribution onto the real structure,
same convention as `render_1brs_pymol.py`, except residues are selected for
the "sticks" highlight by the fixed attribution > 0.4 threshold (as
requested) rather than top-15% ranking, so the two conventions can be
compared directly against the ig_1brs_structure_attribution.png panel.
"""

from __future__ import annotations

import json

import pymol2

IN_JSON = "figures/ig_1cse_attribution.json"
PDB_PATH = "data/raw/skempi2/PDBs/1CSE.pdb"
SELF_CHAIN, PARTNER_CHAIN = "I", "E"  # eglin c (self, in training), subtilisin (partner)


def resi_selector(resnums) -> str:
    return "+".join(f"{resnum}{(icode or '').strip()}" for resnum, icode in resnums)


def set_bfactors(cmd, chain: str, resnums, norm_values) -> None:
    for (resnum, icode), v in zip(resnums, norm_values):
        icode = (icode or "").strip()
        cmd.alter(f"chain {chain} and resi {resnum}{icode}", f"b={v}")


def main():
    with open(IN_JSON) as f:
        d = json.load(f)
    threshold = d["attribution_threshold"]

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(PDB_PATH, "cplx")
        cmd.hide("everything")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.show("cartoon")
        cmd.color("grey70", "cplx")

        set_bfactors(cmd, SELF_CHAIN, d["self_resnums"], d["self_attribution_norm"])
        set_bfactors(cmd, PARTNER_CHAIN, d["partner_resnums"], d["partner_attribution_norm"])

        # --- Panel 1: attribution-colored cartoon, sticks where attribution > threshold ---
        cmd.spectrum("b", "blue_white_red", f"polymer and (chain {SELF_CHAIN} or chain {PARTNER_CHAIN})",
                      minimum=0.0, maximum=1.0)
        self_above = [d["self_resnums"][i] for i, v in enumerate(d["self_attribution_norm"]) if v > threshold]
        partner_above = [d["partner_resnums"][i] for i, v in enumerate(d["partner_attribution_norm"]) if v > threshold]
        if self_above:
            cmd.show("sticks", f"(chain {SELF_CHAIN} and resi {resi_selector(self_above)}) and not name C+N+O")
        if partner_above:
            cmd.show("sticks", f"(chain {PARTNER_CHAIN} and resi {resi_selector(partner_above)}) and not name C+N+O")
        cmd.orient("cplx")
        cmd.ray(1400, 1050)
        cmd.png("figures/ig_1cse_structure_attribution.png", dpi=200)

        # --- Panel 2: ground-truth interface (distance <= cutoff) ---
        cmd.hide("sticks")
        cmd.color("grey70", "cplx")
        cutoff = d["interface_cutoff_angstrom"]
        self_iface = [d["self_resnums"][i] for i, dist in enumerate(d["self_distance_to_partner"]) if dist <= cutoff]
        partner_iface = [d["partner_resnums"][i] for i, dist in enumerate(d["partner_distance_to_self"]) if dist <= cutoff]
        cmd.color("forest", f"chain {SELF_CHAIN} and resi {resi_selector(self_iface)}")
        cmd.color("forest", f"chain {PARTNER_CHAIN} and resi {resi_selector(partner_iface)}")
        cmd.orient("cplx")
        cmd.ray(1400, 1050)
        cmd.png("figures/ig_1cse_structure_groundtruth.png", dpi=200)

        # --- Panel 3: overlap of (attribution > threshold) with true interface ---
        cmd.hide("sticks")
        cmd.color("grey80", "cplx")
        self_above_set = {tuple(r) for r in self_above}
        partner_above_set = {tuple(r) for r in partner_above}
        self_iface_set = {tuple(r) for r in self_iface}
        partner_iface_set = {tuple(r) for r in partner_iface}
        self_overlap = list(self_above_set & self_iface_set)
        partner_overlap = list(partner_above_set & partner_iface_set)
        self_attr_only = list(self_above_set - self_iface_set)
        partner_attr_only = list(partner_above_set - partner_iface_set)
        if self_attr_only:
            cmd.color("marine", f"chain {SELF_CHAIN} and resi {resi_selector(self_attr_only)}")
        if partner_attr_only:
            cmd.color("marine", f"chain {PARTNER_CHAIN} and resi {resi_selector(partner_attr_only)}")
        if self_overlap:
            cmd.color("magenta", f"chain {SELF_CHAIN} and resi {resi_selector(self_overlap)}")
            cmd.show("sticks", f"(chain {SELF_CHAIN} and resi {resi_selector(self_overlap)}) and not name C+N+O")
        if partner_overlap:
            cmd.color("magenta", f"chain {PARTNER_CHAIN} and resi {resi_selector(partner_overlap)}")
            cmd.show("sticks", f"(chain {PARTNER_CHAIN} and resi {resi_selector(partner_overlap)}) and not name C+N+O")
        cmd.orient("cplx")
        cmd.ray(1400, 1050)
        cmd.png("figures/ig_1cse_structure_overlap.png", dpi=200)

        print("saved figures/ig_1cse_structure_attribution.png")
        print("saved figures/ig_1cse_structure_groundtruth.png")
        print("saved figures/ig_1cse_structure_overlap.png")
        print(f"self overlap (attr>{threshold} AND true interface): {sorted(self_overlap)}")
        print(f"partner overlap (attr>{threshold} AND true interface): {sorted(partner_overlap)}")


if __name__ == "__main__":
    main()
