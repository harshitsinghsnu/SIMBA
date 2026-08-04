"""PyMOL structure renders for the 5 named-complex interpretability case
studies: attribution-colored cartoon, ground-truth interface, and overlap
-highlighted (attr>threshold AND true interface), same convention as
`render_1cse_pymol.py`, generalized to multi-chain self/partner groups.
"""

from __future__ import annotations

import json

import pymol2

CASES = ["ras_raf", "trypsin_bpti", "colicin_im9", "fab2d1_1918ha", "hyhel10_lysozyme"]


def resi_selector(keys_for_chain) -> str:
    return "+".join(f"{resnum}{(icode or '').strip()}" for resnum, icode in keys_for_chain)


def group_by_chain(keys):
    by_chain: dict[str, list] = {}
    for chain, resnum, icode in keys:
        by_chain.setdefault(chain, []).append((resnum, icode))
    return by_chain


def set_bfactors(cmd, keys, norm_values) -> None:
    for (chain, resnum, icode), v in zip(keys, norm_values):
        icode = (icode or "").strip()
        cmd.alter(f"chain {chain} and resi {resnum}{icode}", f"b={v}")


def color_by_chain(cmd, color, keys) -> str:
    """Colors residues (grouped per chain) and returns a combined selection string."""
    by_chain = group_by_chain(keys)
    sels = []
    for chain, resis in by_chain.items():
        sel = f"chain {chain} and resi {resi_selector(resis)}"
        cmd.color(color, sel)
        sels.append(sel)
    return " or ".join(f"({s})" for s in sels) if sels else "none"


def main():
    for case_name in CASES:
        with open(f"figures/ig_{case_name}_attribution.json") as f:
            d = json.load(f)
        pdb_path = f"data/raw/skempi2/PDBs/{d['complex_id']}.pdb"
        threshold = d["attribution_threshold"]
        cutoff = d["interface_cutoff_angstrom"]
        self_keys = [tuple(k) for k in d["self_keys"]]
        partner_keys = [tuple(k) for k in d["partner_keys"]]

        with pymol2.PyMOL() as p:
            cmd = p.cmd
            cmd.load(pdb_path, "cplx")
            cmd.hide("everything")
            cmd.bg_color("white")
            cmd.set("ray_opaque_background", 1)
            cmd.show("cartoon")
            cmd.color("grey70", "cplx")

            set_bfactors(cmd, self_keys, d["self_attribution_norm"])
            set_bfactors(cmd, partner_keys, d["partner_attribution_norm"])
            all_chains = "+".join(sorted(set(k[0] for k in self_keys + partner_keys)))
            cmd.spectrum("b", "blue_white_red", f"polymer and chain {all_chains}", minimum=0.0, maximum=1.0)
            cmd.orient("cplx")
            cmd.ray(1200, 900)
            cmd.png(f"figures/ig_{case_name}_structure_attribution.png", dpi=180)

            # overlap: attr>threshold AND true interface
            cmd.color("grey80", "cplx")
            self_above = [k for k, v in zip(self_keys, d["self_attribution_norm"]) if v > threshold]
            partner_above = [k for k, v in zip(partner_keys, d["partner_attribution_norm"]) if v > threshold]
            self_iface = [k for k, dist in zip(self_keys, d["self_distance_to_partner"]) if dist <= cutoff]
            partner_iface = [k for k, dist in zip(partner_keys, d["partner_distance_to_self"]) if dist <= cutoff]
            self_overlap = list(set(self_above) & set(self_iface))
            partner_overlap = list(set(partner_above) & set(partner_iface))
            self_attr_only = list(set(self_above) - set(self_iface))
            partner_attr_only = list(set(partner_above) - set(partner_iface))

            if self_attr_only:
                color_by_chain(cmd, "marine", self_attr_only)
            if partner_attr_only:
                color_by_chain(cmd, "marine", partner_attr_only)
            if self_overlap:
                sel = color_by_chain(cmd, "magenta", self_overlap)
                cmd.show("sticks", f"({sel}) and not name C+N+O")
            if partner_overlap:
                sel = color_by_chain(cmd, "magenta", partner_overlap)
                cmd.show("sticks", f"({sel}) and not name C+N+O")
            cmd.orient("cplx")
            cmd.ray(1200, 900)
            cmd.png(f"figures/ig_{case_name}_structure_overlap.png", dpi=180)

            print(f"[{case_name}] saved attribution + overlap renders "
                  f"(self overlap={len(self_overlap)}, partner overlap={len(partner_overlap)})")


if __name__ == "__main__":
    main()
