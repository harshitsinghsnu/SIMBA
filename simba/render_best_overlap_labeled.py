"""Labeled, zoomed-in PyMOL2 figures for the two best-performing
interpretability case studies (see MODEL_REPORT.md S3.18): 1CSE (eglin
c-subtilisin) and 3BTD (trypsin-BPTI). Every residue shown as a colored
stick -- on both sides, not just the "best" side -- is labeled with its
residue name + number.

Label placement: each labeled CA's real 3D position is read back from
PyMOL and projected onto the current view's screen plane. Every label is
placed at the *same* fixed radius from the cluster centroid (not from its
own residue), at an angle evenly spaced around the full circle (spacing
= 360 / n_labels, in the same order as each residue's true angle from the
centroid).

The radius (in Angstrom) needed to hit a target *pixel* gap between
adjacent labels is not a fixed constant -- it depends on the actual
on-screen scale after `cmd.zoom`, which depends on the buffer argument
*and* on the bounding-box size of the labeled selection itself (which
differs a lot between the two figures here). Assuming a fixed buffer ->
pixel-scale relationship silently under- or over-shoots depending on the
case (this was verified directly: the same buffer value produced visibly
different px/Angstrom scales for the two structures). So the scale is
measured empirically each time: two throwaway markers are placed a known
Angstrom distance apart, rendered, and their pixel separation measured
directly, then the radius is solved for and refined over a few
iterations (zoom depends on radius, and the achieved scale depends on
zoom, so this is a small fixed-point loop, not a one-shot formula).

The connector line for each label runs from the true residue position
out to that residue's point on the layout circle, so line length varies
(residues near the centroid get a longer line) but label positions
themselves are guaranteed non-overlapping by construction.

Leader lines: CGO-based lines/cylinders (and a PIL pixel-space overlay)
were tried first and abandoned -- CGO lines combined with a shown cartoon
representation reproducibly rendered as wildly wrong stray lines in this
PyMOL build (a genuine rendering bug, not a coordinate error; verified
extensively). The actual fix was to stop using CGO entirely: PyMOL has a
*native* label-connector feature (`label_placement_offset` + `set
label_connector, 1`) that draws the leader line as part of the label
itself rather than as a separate CGO object, and it renders correctly
with the cartoon shown. That is what is used below.

Color convention (same for both figures): blue = self-side overlap
residue, orange = partner-side overlap residue. "Overlap" means
attribution > threshold AND the residue is genuinely within the true
-interface distance cutoff (both conditions checked directly from the
saved `figures/ig_*_attribution.json`, not assumed).
"""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pymol2
from PIL import Image

CASES = [
    {"json": "figures/ig_1cse_attribution.json", "pdb": "data/raw/skempi2/PDBs/1CSE.pdb",
     "out": "figures/best_overlap_1cse_labeled.png", "self_label": "eglin c", "partner_label": "subtilisin"},
    {"json": "figures/ig_trypsin_bpti_attribution.json", "pdb": "data/raw/skempi2/PDBs/3BTD.pdb",
     "out": "figures/best_overlap_trypsin_bpti_labeled.png", "self_label": "BPTI", "partner_label": "trypsin"},
]

SELF_COLOR = "marine"
PARTNER_COLOR = "orange"
CONNECTOR_COLOR = "grey40"
ZOOM_MARGIN = 15.0
TARGET_LABEL_PX_SEP = 340.0  # desired on-screen pixel gap between adjacent labels, in FINAL image pixels
IMG_W, IMG_H = 1700, 1700  # square, so horizontal and vertical px/Angstrom scale are identical --
                            # a non-square viewport was measured to give a smaller vertical scale,
                            # which silently broke separation for label pairs that differ mostly in y
CALIB_W, CALIB_H = 500, 500  # low-res preview used only to measure px/Angstrom scale (cheap, but must be
                              # rescaled to IMG_W since a smaller preview has fewer px per Angstrom)


def measure_px_per_angstrom(cmd, centroid, cam_right, probe_a):
    """Empirically calibrate the actual on-screen scale at the *current*
    zoom, instead of assuming a fixed relationship between cmd.zoom's
    buffer argument and the resulting field of view. That assumption
    breaks down here because the FOV also depends on the bounding box of
    the labeled selection itself (which differs a lot between the two
    cases), so the same buffer value does not yield the same px/Angstrom
    scale for both figures. Places two throwaway markers a known distance
    apart, renders a small low-res preview, measures their pixel distance
    by color, and removes the markers again."""
    p1 = centroid + probe_a * cam_right
    p2 = centroid - probe_a * cam_right
    cmd.pseudoatom("_calib1", pos=list(p1))
    cmd.pseudoatom("_calib2", pos=list(p2))
    cmd.show("spheres", "_calib1 or _calib2")
    cmd.set("sphere_scale", 4, "_calib1 or _calib2")
    cmd.color("red", "_calib1")
    cmd.color("blue", "_calib2")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cmd.ray(CALIB_W, CALIB_H)
    cmd.png(tmp_path, dpi=150)
    img = np.array(Image.open(tmp_path).convert("RGB"), dtype=np.int16)
    cmd.delete("_calib1 or _calib2")

    red_mask = (img[:, :, 0] > 150) & (img[:, :, 1] < 100) & (img[:, :, 2] < 100)
    blue_mask = (img[:, :, 2] > 150) & (img[:, :, 0] < 100) & (img[:, :, 1] < 100)
    red_ys, red_xs = np.nonzero(red_mask)
    blue_ys, blue_xs = np.nonzero(blue_mask)
    red_center = np.array([red_xs.mean(), red_ys.mean()])
    blue_center = np.array([blue_xs.mean(), blue_ys.mean()])
    px_dist = float(np.linalg.norm(red_center - blue_center))
    # Rescale to what this distance would be in the FINAL (IMG_W-wide) render,
    # since the low-res preview has fewer pixels per Angstrom than the real output.
    px_dist_final_scale = px_dist * (IMG_W / CALIB_W)
    return px_dist_final_scale / (2.0 * probe_a)


def get_keys(d, side):
    if f"{side}_keys" in d:
        return [tuple(k) for k in d[f"{side}_keys"]]
    chain = d[f"{side}_chain"]
    return [(chain, r, ic) for r, ic in d[f"{side}_resnums"]]


def overlap_indices(d, side):
    threshold = d["attribution_threshold"]
    cutoff = d["interface_cutoff_angstrom"]
    attr_norm = d[f"{side}_attribution_norm"]
    dist_key = "self_distance_to_partner" if side == "self" else "partner_distance_to_self"
    dist = d[dist_key]
    above = set(i for i, v in enumerate(attr_norm) if v > threshold)
    iface = set(i for i, dd in enumerate(dist) if dd <= cutoff)
    return sorted(above & iface)


def resi_sel(chain, resnum, icode):
    icode = (icode or "").strip()
    return f"chain {chain} and resi {resnum}{icode}"


def evenly_spaced_angles(angles_raw):
    """Place n labels at exactly-even angles around a full circle (gap =
    360/n), in the same cyclic order as their true angles from the
    centroid. Returns adjusted_angles_deg in the original input order."""
    n = len(angles_raw)
    if n == 1:
        return [angles_raw[0]]
    gap_deg = 360.0 / n
    order = sorted(range(n), key=lambda i: angles_raw[i])
    base = angles_raw[order[0]]
    adjusted = [0.0] * n
    for rank, idx in enumerate(order):
        adjusted[idx] = (base + rank * gap_deg) % 360.0
    return adjusted


def main():
    for case in CASES:
        with open(case["json"]) as f:
            d = json.load(f)

        self_keys = get_keys(d, "self")
        partner_keys = get_keys(d, "partner")
        self_overlap_keys = [self_keys[i] for i in overlap_indices(d, "self")]
        partner_overlap_keys = [partner_keys[i] for i in overlap_indices(d, "partner")]
        all_keys = [(k, SELF_COLOR) for k in self_overlap_keys] + \
                   [(k, PARTNER_COLOR) for k in partner_overlap_keys]
        print(f"{case['out']}: self overlap={self_overlap_keys}, partner overlap={partner_overlap_keys} "
              f"({len(all_keys)} total labeled residues)")

        with pymol2.PyMOL() as p:
            cmd = p.cmd
            cmd.load(case["pdb"], "cplx")
            cmd.hide("everything")
            cmd.bg_color("white")
            cmd.set("ray_opaque_background", 1)
            cmd.set("orthoscopic", 1)
            cmd.set("label_size", 12)
            cmd.set("label_color", "black")
            cmd.set("label_outline_color", "white")
            cmd.set("label_font_id", 7)  # sans-serif bold
            cmd.set("label_connector", 1)
            cmd.set("label_connector_color", CONNECTOR_COLOR)
            cmd.set("label_connector_width", 1.5)
            cmd.show("cartoon")
            cmd.color("grey80", "cplx")
            cmd.set("cartoon_transparency", 0.35)

            for (chain, resnum, icode), color in all_keys:
                sel = resi_sel(chain, resnum, icode)
                cmd.color(color, sel)
                cmd.show("sticks", f"({sel}) and not name C+N+O")
                cmd.set("cartoon_transparency", 0.0, sel)

            all_sel = " or ".join(f"({resi_sel(*k)})" for k, _ in all_keys)
            cmd.orient(all_sel)

            n = len(all_keys)
            gap_rad = math.radians(360.0 / n) if n > 1 else 1.0
            radius = 30.0  # initial guess, Angstrom; refined below against the real render scale
            for _ in range(4):
                cmd.zoom(all_sel, buffer=radius + ZOOM_MARGIN)
                view = cmd.get_view()
                cam_right = np.array(view[0:3])
                coords_tmp = []
                for (chain, resnum, icode), _ in all_keys:
                    ca_sel = f"({resi_sel(chain, resnum, icode)}) and name CA"
                    coords_tmp.append(np.array(cmd.get_atom_coords(ca_sel)))
                centroid_tmp = np.mean(coords_tmp, axis=0)
                probe_a = max(radius * 0.4, 8.0)
                px_per_a = measure_px_per_angstrom(cmd, centroid_tmp, cam_right, probe_a)
                radius = TARGET_LABEL_PX_SEP / (px_per_a * gap_rad)
                radius = min(radius, 300.0)

            cmd.zoom(all_sel, buffer=radius + ZOOM_MARGIN)

            # Real 3D CA coordinates -> screen-plane (camera) coordinates,
            # so the layout circle matches what the rendered view actually
            # shows, not the underlying lab-frame coordinates. Read the view
            # *after* the final zoom so cam_right/cam_up exactly match the
            # camera state used for rendering.
            view = cmd.get_view()
            cam_right = np.array(view[0:3])
            cam_up = np.array(view[3:6])
            coords = []
            for (chain, resnum, icode), _ in all_keys:
                ca_sel = f"({resi_sel(chain, resnum, icode)}) and name CA"
                xyz = np.array(cmd.get_atom_coords(ca_sel))
                coords.append(xyz)
            centroid = np.mean(coords, axis=0)
            rel = [c - centroid for c in coords]
            anchors_2d = [(float(np.dot(r, cam_right)), float(np.dot(r, cam_up))) for r in rel]
            angles_raw = [math.degrees(math.atan2(y, x)) % 360.0 for x, y in anchors_2d]

            angles = evenly_spaced_angles(angles_raw)
            print(f"  calibrated radius={radius:.1f}A, px_per_A={px_per_a:.2f}")

            for (key, _), (ax, ay), angle in zip(all_keys, anchors_2d, angles):
                chain, resnum, icode = key
                ca_sel = f"({resi_sel(chain, resnum, icode)}) and name CA"
                rad = math.radians(angle)
                fx, fy = radius * math.cos(rad), radius * math.sin(rad)
                offset_cam = cam_right * (fx - ax) + cam_up * (fy - ay)
                cmd.label(ca_sel, "'%s%s' % (resn, resi)")
                cmd.set("label_placement_offset", tuple(offset_cam), ca_sel)

            cmd.ray(IMG_W, IMG_H)
            cmd.png(case["out"], dpi=250)

        print(f"saved {case['out']} ({len(self_overlap_keys)} self [{SELF_COLOR}] + "
              f"{len(partner_overlap_keys)} partner [{PARTNER_COLOR}] labeled residues)")


if __name__ == "__main__":
    main()
