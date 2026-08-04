"""Integrated-Gradients interpretability, generalized across 5 named,
well-known complexes confirmed present in the checkpoint's training data
(verified directly against real RCSB entry titles, not assumed from PDB ID
alone -- see chat record / MODEL_REPORT.md for the verification):

  Ras-Raf              4G0N  (H-Ras-GppNHp bound to the RBD of Raf Kinase)
  Trypsin-BPTI         3BTD  (bovine beta-trypsin + a P1-variant of BPTI;
                              the user asked about 3BTK specifically, which
                              is not in the pool, but 3BTD/E/G/M/Q/W --
                              the same crystallographic series -- are)
  Colicin E9-Im9       1EMV  (colicin E9 DNase domain + cognate immunity
                              protein Im9)
  Fab 2D1-1918 H1N1 HA 3LZF  (Fab 2D1 in complex with the 1918 pandemic
                              influenza hemagglutinin)
  HyHEL-10-lysozyme    3HFM  (HyHEL-10 Fab + hen egg-white lysozyme)

(MDM2-p53 and CR6261-HA-stem, the other two complexes asked about, are
confirmed absent from this project's data -- no case study for those.)

Every wildtype row for these 5 complexes landed in the checkpoint's
*training* split (verified by replaying the same seeded shuffle
`train_final_checkpoint.py` used), so -- like 1CSE -- these test whether
attribution quality is better on complexes the trainable head has actually
seen a gradient signal from, as opposed to 1BRS (held out entirely).

Self/partner chain assignment is taken from this project's own training
data directly (`build_combined_affinity.py`'s wildtype row for each
complex), not assumed from group_a/group_b convention -- for two of the
five (3LZF, 3HFM) "self" or "partner" is a concatenation of two PDB chains
(an antibody's heavy+light, or a cleaved HA's two fragments), verified by
exact sequence match before use.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr, spearmanr

from simba import env

env.setup()

from simba.data.pdb_utils import parse_pdb_ca_coords, parse_pdb_chains  # noqa: E402
from simba.features.cache import SequenceScoreCache  # noqa: E402
from simba.features.plm import ESMScorer  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402

CKPT_PATH = "checkpoints/simba_pairformer_combined.pt"
META_PATH = "checkpoints/simba_pairformer_combined.json"
INTERFACE_CUTOFF_A = 8.0
IG_STEPS = 64
ATTRIBUTION_THRESHOLD = 0.4

CASES = [
    {"name": "ras_raf", "label": "Ras-Raf", "complex_id": "4G0N",
     "self_chains": ["B"], "partner_chains": ["A"]},
    {"name": "trypsin_bpti", "label": "Trypsin-BPTI", "complex_id": "3BTD",
     "self_chains": ["I"], "partner_chains": ["E"]},
    {"name": "colicin_im9", "label": "Colicin E9-Im9", "complex_id": "1EMV",
     "self_chains": ["A"], "partner_chains": ["B"]},
    {"name": "fab2d1_1918ha", "label": "Fab 2D1-1918 H1N1 HA", "complex_id": "3LZF",
     "self_chains": ["A", "B"], "partner_chains": ["H", "L"]},
    {"name": "hyhel10_lysozyme", "label": "HyHEL-10-Lysozyme", "complex_id": "3HFM",
     "self_chains": ["Y"], "partner_chains": ["H", "L"]},
]


def compute_attribution(model, self_embedding, partner_embedding, attribute_self: bool):
    def broadcast_to(t, batch_size):
        return t.expand(batch_size, *t.shape[1:]) if t.shape[0] != batch_size else t

    if attribute_self:
        def forward_fn(x, fixed):
            fixed = broadcast_to(fixed, x.shape[0])
            return torch.stack([model(x[i], fixed[i]) for i in range(x.shape[0])])
        target, fixed = self_embedding, partner_embedding
    else:
        def forward_fn(x, fixed):
            fixed = broadcast_to(fixed, x.shape[0])
            return torch.stack([model(fixed[i], x[i]) for i in range(x.shape[0])])
        target, fixed = partner_embedding, self_embedding

    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        inputs=target.unsqueeze(0), baselines=torch.zeros_like(target).unsqueeze(0),
        additional_forward_args=(fixed.unsqueeze(0),), n_steps=IG_STEPS,
        return_convergence_delta=True,
    )
    per_residue = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    return per_residue, float(delta.abs().item())


def build_group(chains_dict, chain_ids):
    """Concatenate sequence + (chain, resnum, icode) keys across one or more
    PDB chains, preserving chain identity per residue (needed for multi
    -chain self/partner groups, e.g. an antibody's heavy+light)."""
    seq = ""
    keys = []
    for cid in chain_ids:
        cs = chains_dict[cid]
        seq += cs.sequence
        keys.extend((cid, r.resnum, r.icode) for r in cs.residues)
    return seq, keys


def min_distance(coords, point, target_keys):
    px, py, pz = point
    best = None
    for (chain, resnum, icode) in target_keys:
        x, y, z = coords[chain][(resnum, icode)]
        d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
        if best is None or d < best:
            best = d
    return best


def run_case(model, cache, case):
    pdb_path = f"data/raw/skempi2/PDBs/{case['complex_id']}.pdb"
    chains = parse_pdb_chains(pdb_path)
    coords = parse_pdb_ca_coords(pdb_path)

    self_seq, self_keys = build_group(chains, case["self_chains"])
    partner_seq, partner_keys = build_group(chains, case["partner_chains"])
    print(f"[{case['name']}] self({'+'.join(case['self_chains'])})={len(self_seq)} aa, "
          f"partner({'+'.join(case['partner_chains'])})={len(partner_seq)} aa", flush=True)

    device = next(model.parameters()).device
    self_embedding = torch.from_numpy(cache.get(self_seq).embedding).float().to(device)
    partner_embedding = torch.from_numpy(cache.get(partner_seq).embedding).float().to(device)

    with torch.no_grad():
        pred_dg = model(self_embedding, partner_embedding).item()
    print(f"[{case['name']}] predicted dG = {pred_dg:.3f} kcal/mol (wildtype row was in TRAIN)", flush=True)

    self_attr, self_delta = compute_attribution(model, self_embedding, partner_embedding, True)
    partner_attr, partner_delta = compute_attribution(model, self_embedding, partner_embedding, False)
    print(f"[{case['name']}] IG convergence delta: self={self_delta:.4f} partner={partner_delta:.4f}", flush=True)

    self_dist = np.array([min_distance(coords, coords[k[0]][(k[1], k[2])], partner_keys) for k in self_keys])
    partner_dist = np.array([min_distance(coords, coords[k[0]][(k[1], k[2])], self_keys) for k in partner_keys])

    def normalize(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / ((hi - lo) or 1.0)

    self_attr_norm, partner_attr_norm = normalize(self_attr), normalize(partner_attr)
    self_pearson, _ = pearsonr(self_dist, self_attr)
    self_spearman, _ = spearmanr(self_dist, self_attr)
    partner_pearson, _ = pearsonr(partner_dist, partner_attr)
    partner_spearman, _ = spearmanr(partner_dist, partner_attr)
    print(f"[{case['name']}] self: corr(dist,attr) Pearson={self_pearson:.3f} Spearman={self_spearman:.3f}", flush=True)
    print(f"[{case['name']}] partner: corr(dist,attr) Pearson={partner_pearson:.3f} Spearman={partner_spearman:.3f}", flush=True)

    def threshold_stats(dist, attr_norm, threshold, cutoff=INTERFACE_CUTOFF_A):
        above = np.where(attr_norm > threshold)[0]
        true_iface = set(np.where(dist <= cutoff)[0].tolist())
        n_iface = len(true_iface)
        if len(above) == 0:
            return None, 0, n_iface
        hits = sum(1 for i in above if i in true_iface)
        return hits / len(above), len(above), n_iface

    self_prec, self_n_above, self_n_iface = threshold_stats(self_dist, self_attr_norm, ATTRIBUTION_THRESHOLD)
    partner_prec, partner_n_above, partner_n_iface = threshold_stats(partner_dist, partner_attr_norm, ATTRIBUTION_THRESHOLD)
    self_baseline = self_n_iface / len(self_seq)
    partner_baseline = partner_n_iface / len(partner_seq)
    print(f"[{case['name']}] self: {self_n_above}/{len(self_seq)} residues attr>{ATTRIBUTION_THRESHOLD}, "
          f"precision={self_prec} (random={self_baseline:.3f})", flush=True)
    print(f"[{case['name']}] partner: {partner_n_above}/{len(partner_seq)} residues attr>{ATTRIBUTION_THRESHOLD}, "
          f"precision={partner_prec} (random={partner_baseline:.3f})", flush=True)

    result = {
        **case, "predicted_dG": pred_dg,
        "self_sequence": self_seq, "partner_sequence": partner_seq,
        "self_keys": self_keys, "partner_keys": partner_keys,
        "self_attribution": self_attr.tolist(), "partner_attribution": partner_attr.tolist(),
        "self_attribution_norm": self_attr_norm.tolist(), "partner_attribution_norm": partner_attr_norm.tolist(),
        "self_distance_to_partner": self_dist.tolist(), "partner_distance_to_self": partner_dist.tolist(),
        "self_corr_pearson": self_pearson, "self_corr_spearman": self_spearman,
        "partner_corr_pearson": partner_pearson, "partner_corr_spearman": partner_spearman,
        "attribution_threshold": ATTRIBUTION_THRESHOLD, "interface_cutoff_angstrom": INTERFACE_CUTOFF_A,
        "self_precision_above_threshold": self_prec, "self_n_above_threshold": int(self_n_above), "self_baseline": self_baseline,
        "partner_precision_above_threshold": partner_prec, "partner_n_above_threshold": int(partner_n_above), "partner_baseline": partner_baseline,
        "trained_on": True,
    }
    out_path = f"figures/ig_{case['name']}_attribution.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{case['name']}] saved {out_path}\n", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(META_PATH) as f:
        meta = json.load(f)
    model = AbsoluteAffinityModel(
        plm_hidden_size=meta["plm_hidden_size"], d_model=meta["d_model"],
        n_heads=meta["n_heads"], n_memory=meta["n_memory"], pair_hidden=meta["pair_hidden"],
        mlp_hidden=meta["mlp_hidden"], use_pairformer=meta["use_pairformer"],
        n_pairformer_layers=meta["n_pairformer_layers"],
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    scorer = ESMScorer(model_name=meta["backbone"], device=device)
    cache = SequenceScoreCache(scorer)

    for case in CASES:
        run_case(model, cache, case)


if __name__ == "__main__":
    main()
