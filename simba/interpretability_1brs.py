"""Interpretability case study: Integrated Gradients on the trained
pairformer checkpoint (`train_final_checkpoint.py`), applied to 1BRS
(barnase-barstar), a classic, extensively alanine-scanned hotspot system
(Schreiber & Fersht) held out entirely from that checkpoint's training set
-- so this reflects genuine generalization, not memorization.

Attribution scope: IG is applied to the *frozen ESM2 embedding* input of
the trainable scoring head (projection + GatedMemory + PairwiseInterfaceBlock
+ MLP head), not back through the 650M-parameter frozen backbone itself --
the question this answers is "which residues does SIMBA's trainable
machinery weight most heavily", which only requires differentiating through
the trainable path. A zero-vector baseline (no residue information) is used,
standard practice for IG on continuous embeddings.

Ground truth: real CA-CA distance from each barnase (self, chain A) residue
to the nearest barstar (partner, chain D) residue, computed directly from
the PDB structure -- a post-hoc diagnostic only, matching this project's
established convention (`simba/data/pdb_utils.py`, reused from the §3.11
biological error analysis), never fed into the model. Residues within a
cutoff of the partner chain are the operational "true interface" set.
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

PDB_PATH = "data/raw/skempi2/PDBs/1BRS.pdb"
SELF_CHAIN, PARTNER_CHAIN = "A", "D"  # barnase (self), barstar (partner) -- project convention
CKPT_PATH = "checkpoints/simba_pairformer_combined.pt"
META_PATH = "checkpoints/simba_pairformer_combined.json"
INTERFACE_CUTOFF_A = 8.0  # CA-CA angstroms, same style cutoff as §3.11's distance analysis
IG_STEPS = 64
OUT_JSON = "figures/ig_1brs_attribution.json"


def compute_attribution(model, self_embedding, partner_embedding, attribute_self: bool):
    """Returns a per-residue attribution score (L2 norm over the embedding
    dim) for whichever side (`self` or `partner`) is attributed, holding the
    other side fixed as a non-attributed forward argument.

    The model only ever runs on one unbatched example at a time (variable
    sequence length across examples rules out batching everywhere else in
    this project too), but captum's IntegratedGradients internally expands
    its `inputs` tensor into `n_steps` interpolated copies along a leading
    batch dimension, and repeats `additional_forward_args` to match that
    same batch size -- so `forward_fn` must loop over that dimension itself
    rather than assume a fixed batch size of 1.
    """
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
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    chains = parse_pdb_chains(PDB_PATH)
    self_cs, partner_cs = chains[SELF_CHAIN], chains[PARTNER_CHAIN]
    print(f"[1brs] self(barnase)={len(self_cs.sequence)} aa, "
          f"partner(barstar)={len(partner_cs.sequence)} aa", flush=True)

    scorer = ESMScorer(model_name=meta["backbone"], device=device)
    cache = SequenceScoreCache(scorer)
    self_embedding = torch.from_numpy(cache.get(self_cs.sequence).embedding).float().to(device)
    partner_embedding = torch.from_numpy(cache.get(partner_cs.sequence).embedding).float().to(device)

    with torch.no_grad():
        pred_dg = model(self_embedding, partner_embedding).item()
    print(f"[1brs] predicted dG = {pred_dg:.3f} kcal/mol (held out of training)", flush=True)

    self_attr, self_delta = compute_attribution(model, self_embedding, partner_embedding, attribute_self=True)
    partner_attr, partner_delta = compute_attribution(model, self_embedding, partner_embedding, attribute_self=False)
    print(f"[1brs] IG convergence delta: self={self_delta:.4f} partner={partner_delta:.4f}", flush=True)

    coords = parse_pdb_ca_coords(PDB_PATH)
    self_dist = []
    for r in self_cs.residues:
        d = None
        px, py, pz = coords[SELF_CHAIN][(r.resnum, r.icode)]
        for (x, y, z) in coords[PARTNER_CHAIN].values():
            dd = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
            if d is None or dd < d:
                d = dd
        self_dist.append(d)
    self_dist = np.asarray(self_dist)

    partner_dist = []
    for r in partner_cs.residues:
        px, py, pz = coords[PARTNER_CHAIN][(r.resnum, r.icode)]
        d = None
        for (x, y, z) in coords[SELF_CHAIN].values():
            dd = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** 0.5
            if d is None or dd < d:
                d = dd
        partner_dist.append(d)
    partner_dist = np.asarray(partner_dist)

    self_pearson, _ = pearsonr(self_dist, self_attr)
    self_spearman, _ = spearmanr(self_dist, self_attr)
    partner_pearson, _ = pearsonr(partner_dist, partner_attr)
    partner_spearman, _ = spearmanr(partner_dist, partner_attr)
    print(f"[1brs] barnase (self): corr(distance, attribution) Pearson={self_pearson:.3f} Spearman={self_spearman:.3f}", flush=True)
    print(f"[1brs] barstar (partner): corr(distance, attribution) Pearson={partner_pearson:.3f} Spearman={partner_spearman:.3f}", flush=True)

    def precision_at_k(dist, attr, k, cutoff=INTERFACE_CUTOFF_A):
        top_k_idx = np.argsort(-attr)[:k]
        true_interface = set(np.where(dist <= cutoff)[0].tolist())
        hits = sum(1 for i in top_k_idx if i in true_interface)
        return hits / k, len(true_interface)

    self_k = max(1, int(round(0.15 * len(self_cs.sequence))))
    partner_k = max(1, int(round(0.15 * len(partner_cs.sequence))))
    self_prec, self_n_iface = precision_at_k(self_dist, self_attr, self_k)
    partner_prec, partner_n_iface = precision_at_k(partner_dist, partner_attr, partner_k)
    self_baseline = self_n_iface / len(self_cs.sequence)
    partner_baseline = partner_n_iface / len(partner_cs.sequence)
    print(f"[1brs] barnase: precision@top{self_k} (<= {INTERFACE_CUTOFF_A}A) = {self_prec:.3f} "
          f"(random baseline {self_baseline:.3f}, {self_n_iface}/{len(self_cs.sequence)} true interface residues)", flush=True)
    print(f"[1brs] barstar: precision@top{partner_k} (<= {INTERFACE_CUTOFF_A}A) = {partner_prec:.3f} "
          f"(random baseline {partner_baseline:.3f}, {partner_n_iface}/{len(partner_cs.sequence)} true interface residues)", flush=True)

    result = {
        "predicted_dG": pred_dg,
        "self_chain": SELF_CHAIN, "partner_chain": PARTNER_CHAIN,
        "self_sequence": self_cs.sequence, "partner_sequence": partner_cs.sequence,
        "self_resnums": [[r.resnum, r.icode] for r in self_cs.residues],
        "partner_resnums": [[r.resnum, r.icode] for r in partner_cs.residues],
        "self_attribution": self_attr.tolist(), "partner_attribution": partner_attr.tolist(),
        "self_distance_to_partner": self_dist.tolist(), "partner_distance_to_self": partner_dist.tolist(),
        "self_corr_pearson": self_pearson, "self_corr_spearman": self_spearman,
        "partner_corr_pearson": partner_pearson, "partner_corr_spearman": partner_spearman,
        "self_precision_at_k": self_prec, "self_precision_k": self_k, "self_precision_baseline": self_baseline,
        "partner_precision_at_k": partner_prec, "partner_precision_k": partner_k, "partner_precision_baseline": partner_baseline,
        "interface_cutoff_angstrom": INTERFACE_CUTOFF_A,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[1brs] saved to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
