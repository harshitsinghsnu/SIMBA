"""Interpretability case study #2: same Integrated-Gradients pipeline as
`interpretability_1brs.py`, but on 1CSE (subtilisin Carlsberg / chain E,
274 aa -- eglin c / chain I, 63 aa), a complex that *was* part of the
checkpoint's training data (its wildtype row landed in the training split,
not the held-out validation slice -- confirmed directly by re-running the
same seeded shuffle `train_final_checkpoint.py` used). This directly tests
whether attribution quality improves on a complex the trainable head has
actually seen a gradient signal from, versus 1BRS (held out entirely).

Self/partner assignment matches this project's own training convention
exactly (verified, not assumed): self_sequence == chain I (eglin c),
partner_sequence == chain E (subtilisin) -- the reverse of 1CSE's
group_a/group_b convention used elsewhere (`skempi2_mutations.csv` lists
group_a=E, group_b=I), because `build_skempi2_expanded.py`'s wildtype/mutant
sequence construction assigns "self" to whichever side the mutations in
this complex's rows fall on (eglin c carries the mutated positions in
SKEMPI2's eglin-subtilisin rows), not to group_a by definition.

In addition to the continuous per-residue attribution score used in the
1BRS study, this script also reports a fixed absolute threshold (attribution
> 0.4 on the same 0-1 min-max-normalized scale) as a second, complementary
"important residue" definition alongside the top-K/top-15% ranking already
used for 1BRS -- both are reported so the choice of threshold convention
doesn't quietly change the conclusion.
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

PDB_PATH = "data/raw/skempi2/PDBs/1CSE.pdb"
SELF_CHAIN, PARTNER_CHAIN = "I", "E"  # eglin c (self, in training), subtilisin (partner) -- verified above
CKPT_PATH = "checkpoints/simba_pairformer_combined.pt"
META_PATH = "checkpoints/simba_pairformer_combined.json"
INTERFACE_CUTOFF_A = 8.0
IG_STEPS = 64
ATTRIBUTION_THRESHOLD = 0.4
OUT_JSON = "figures/ig_1cse_attribution.json"


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
        inputs=target.unsqueeze(0),
        baselines=torch.zeros_like(target).unsqueeze(0),
        additional_forward_args=(fixed.unsqueeze(0),),
        n_steps=IG_STEPS,
        return_convergence_delta=True,
    )
    per_residue = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    return per_residue, float(delta.abs().item())


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

    chains = parse_pdb_chains(PDB_PATH)
    self_cs, partner_cs = chains[SELF_CHAIN], chains[PARTNER_CHAIN]
    print(f"[1cse] self(eglin c)={len(self_cs.sequence)} aa, "
          f"partner(subtilisin)={len(partner_cs.sequence)} aa", flush=True)

    scorer = ESMScorer(model_name=meta["backbone"], device=device)
    cache = SequenceScoreCache(scorer)
    self_embedding = torch.from_numpy(cache.get(self_cs.sequence).embedding).float().to(device)
    partner_embedding = torch.from_numpy(cache.get(partner_cs.sequence).embedding).float().to(device)

    with torch.no_grad():
        pred_dg = model(self_embedding, partner_embedding).item()
    print(f"[1cse] predicted dG = {pred_dg:.3f} kcal/mol (this complex WAS in the training split)", flush=True)

    self_attr, self_delta = compute_attribution(model, self_embedding, partner_embedding, attribute_self=True)
    partner_attr, partner_delta = compute_attribution(model, self_embedding, partner_embedding, attribute_self=False)
    print(f"[1cse] IG convergence delta: self={self_delta:.4f} partner={partner_delta:.4f}", flush=True)

    coords = parse_pdb_ca_coords(PDB_PATH)

    def distances(cs, target_chain):
        out = []
        for r in cs.residues:
            px, py, pz = coords[cs.chain][(r.resnum, r.icode)]
            d = None
            for (x, y, z) in coords[target_chain].values():
                dd = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
                if d is None or dd < d:
                    d = dd
            out.append(d)
        return np.asarray(out)

    self_dist = distances(self_cs, PARTNER_CHAIN)
    partner_dist = distances(partner_cs, SELF_CHAIN)

    def normalize(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / ((hi - lo) or 1.0)

    self_attr_norm = normalize(self_attr)
    partner_attr_norm = normalize(partner_attr)

    self_pearson, _ = pearsonr(self_dist, self_attr)
    self_spearman, _ = spearmanr(self_dist, self_attr)
    partner_pearson, _ = pearsonr(partner_dist, partner_attr)
    partner_spearman, _ = spearmanr(partner_dist, partner_attr)
    print(f"[1cse] eglin c (self): corr(distance, attribution) Pearson={self_pearson:.3f} Spearman={self_spearman:.3f}", flush=True)
    print(f"[1cse] subtilisin (partner): corr(distance, attribution) Pearson={partner_pearson:.3f} Spearman={partner_spearman:.3f}", flush=True)

    def threshold_precision(dist, attr_norm, threshold, cutoff=INTERFACE_CUTOFF_A):
        above = np.where(attr_norm > threshold)[0]
        true_interface = set(np.where(dist <= cutoff)[0].tolist())
        if len(above) == 0:
            return None, 0, len(true_interface)
        hits = sum(1 for i in above if i in true_interface)
        return hits / len(above), len(above), len(true_interface)

    self_prec, self_n_above, self_n_iface = threshold_precision(self_dist, self_attr_norm, ATTRIBUTION_THRESHOLD)
    partner_prec, partner_n_above, partner_n_iface = threshold_precision(partner_dist, partner_attr_norm, ATTRIBUTION_THRESHOLD)
    self_baseline = self_n_iface / len(self_cs.sequence)
    partner_baseline = partner_n_iface / len(partner_cs.sequence)
    print(f"[1cse] eglin c: {self_n_above}/{len(self_cs.sequence)} residues with attribution > {ATTRIBUTION_THRESHOLD}, "
          f"precision={self_prec} (random baseline {self_baseline:.3f})", flush=True)
    print(f"[1cse] subtilisin: {partner_n_above}/{len(partner_cs.sequence)} residues with attribution > {ATTRIBUTION_THRESHOLD}, "
          f"precision={partner_prec} (random baseline {partner_baseline:.3f})", flush=True)

    result = {
        "predicted_dG": pred_dg, "self_chain": SELF_CHAIN, "partner_chain": PARTNER_CHAIN,
        "self_sequence": self_cs.sequence, "partner_sequence": partner_cs.sequence,
        "self_resnums": [[r.resnum, r.icode] for r in self_cs.residues],
        "partner_resnums": [[r.resnum, r.icode] for r in partner_cs.residues],
        "self_attribution": self_attr.tolist(), "partner_attribution": partner_attr.tolist(),
        "self_attribution_norm": self_attr_norm.tolist(), "partner_attribution_norm": partner_attr_norm.tolist(),
        "self_distance_to_partner": self_dist.tolist(), "partner_distance_to_self": partner_dist.tolist(),
        "self_corr_pearson": self_pearson, "self_corr_spearman": self_spearman,
        "partner_corr_pearson": partner_pearson, "partner_corr_spearman": partner_spearman,
        "attribution_threshold": ATTRIBUTION_THRESHOLD,
        "self_precision_above_threshold": self_prec, "self_n_above_threshold": int(self_n_above),
        "self_baseline": self_baseline,
        "partner_precision_above_threshold": partner_prec, "partner_n_above_threshold": int(partner_n_above),
        "partner_baseline": partner_baseline,
        "interface_cutoff_angstrom": INTERFACE_CUTOFF_A,
        "trained_on": True,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[1cse] saved to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
