"""Structure figure comparing SIMBA's top-attributed barstar residues
against Lee & Tidor's (2001) literature-established electrostatic hotspot
residues, colored to show overlap (or its absence) directly on the 1BRS
structure. See `compare_barstar_literature_hotspots.py` for the residue
-numbering reconciliation and overlap computation this consumes.
"""

from __future__ import annotations

import json

import pymol2

PDB_PATH = "data/raw/skempi2/PDBs/1BRS.pdb"
PARTNER_CHAIN = "D"
COMPARISON_JSON = "figures/barstar_literature_comparison.json"
OUT_PNG = "figures/barstar_literature_overlap.png"

COLOR_PAPER_ONLY = "orange"
COLOR_SIMBA_ONLY = "marine"
COLOR_OVERLAP = "magenta"
COLOR_BACKGROUND = "grey80"


def resi_selector(resnums) -> str:
    return "+".join(str(r) for r in resnums)


def main():
    with open(COMPARISON_JSON) as f:
        c = json.load(f)

    paper_only = sorted(set(c["paper_hotspots"]) - set(c["simba_top15pct"]))
    simba_only = sorted(set(c["simba_top15pct"]) - set(c["paper_hotspots"]))
    overlap = sorted(set(c["paper_hotspots"]) & set(c["simba_top15pct"]))
    print(f"paper-only: {paper_only}")
    print(f"SIMBA-only (top15%): {simba_only}")
    print(f"overlap: {overlap}")

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.load(PDB_PATH, "cplx")
        cmd.hide("everything")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        cmd.show("cartoon")
        cmd.color(COLOR_BACKGROUND, "cplx")
        cmd.color("grey90", "chain A")  # barnase: de-emphasized, this figure is about barstar

        if paper_only:
            sel = f"chain {PARTNER_CHAIN} and resi {resi_selector(paper_only)}"
            cmd.color(COLOR_PAPER_ONLY, sel)
            cmd.show("sticks", f"({sel}) and not name C+N+O")
        if simba_only:
            sel = f"chain {PARTNER_CHAIN} and resi {resi_selector(simba_only)}"
            cmd.color(COLOR_SIMBA_ONLY, sel)
            cmd.show("sticks", f"({sel}) and not name C+N+O")
        if overlap:
            sel = f"chain {PARTNER_CHAIN} and resi {resi_selector(overlap)}"
            cmd.color(COLOR_OVERLAP, sel)
            cmd.show("sticks", f"({sel}) and not name C+N+O")

        cmd.orient(f"chain {PARTNER_CHAIN}")
        cmd.zoom(f"chain {PARTNER_CHAIN}", buffer=5)
        cmd.ray(1400, 1050)
        cmd.png(OUT_PNG, dpi=200)
        print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
