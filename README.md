# SIMBA

**A Sequence-Only, Frozen-Backbone Cross-Attention Model for Protein-Protein Binding Affinity and Mutation-Effect Prediction**

SIMBA scores absolute binding free energy (ΔG) and mutation-induced binding-energy change (ΔΔG)
with a single shared architecture: two frozen ESM-2 encoders, a fixed-size interface memory the
partner chain writes into via cross-attention, and a Pairformer-lite block that refines the
self-chain / memory representation with row and column attention.

## Repository layout

```
SIMBA/
  simba/                 the package (import as `simba.<module>`)
    models/               AbsoluteAffinityModel (SIMBA) and the two ablation baselines
    features/             frozen-backbone scoring (ESM-2, MINT, AbLang2) + on-disk embedding cache
    data/                 dataset builders (SKEMPI2, AbDesign, RDE-PPI, ATLAS, MINT-protocol split, ...)
    eval_*.py             one script per experiment, each with its own argparse CLI (see below)
    make_*.py             figure-generation scripts
    train.py, train_final_checkpoint.py
  tests/                  unit tests
  scripts/
    download_data.py       fetches the public raw datasets
    run_main_results.ps1   the headline 3-architecture x 3-split ablation
    run_all_results.ps1    every other experiment
    make_figures.ps1        regenerates figures from result JSONs
  environment.yml / requirements.txt
```

## 1. Setup

### Environment

```powershell
conda env create -f environment.yml
conda activate simba
```

or with an existing Python 3.12 + CUDA 11.8 environment:

```powershell
pip install -r requirements.txt
```

A CUDA GPU is required for training/embedding (all results were produced on a single consumer
GPU, 16 GB VRAM).

### MINT backbone (optional, needed for 3 of the experiments)

The MINT comparison, the frozen-backbone ablation, and the backbone x head factorial all use MINT
as an alternative frozen backbone. Its code and released checkpoint are a separate dependency:

```powershell
git clone https://github.com/VarunUllanat/mint third_party/mint
# then follow that repo's instructions to download its released checkpoint
```

Everything else, including the main 3-split ablation below, runs without this.

### Data

```powershell
python scripts/download_data.py
```

Fetches SKEMPI2 and AB-Bind automatically. Two datasets need a one-time manual step:

- **AbDesign**: register at https://naturalantibody.com/ab-design/, save the file at
  `data/raw/abdesign/abdesign_raw.csv`.
- **Kastritis-lineage benchmark** and **ATLAS TCR:pMHC**: both public; see
  `simba/data/build_affinity_benchmark.py` and `simba/data/build_atlas.py` for expected filenames
  and source URLs.

ESM-2/MINT/AbLang2 embeddings are computed once per unique sequence and cached to disk
(`data/processed/embeddings_cache/`) on first use of any script — this is not shipped (it grows to
tens of GB) but is fully deterministic to regenerate, since the backbone is frozen.

## 2. Running the main experiment

The paper's central result is a 3-architecture x 3-split-protocol ablation on the combined
general-PPI pool (7,247 examples, 541 complexes): SIMBA (Pairformer) vs. two baselines
(Concat+MLP, Cross-attn+MLP), each trained under random-complex, sequence-similarity, and
cold-target 5-fold cross-validation.

This is driven by `simba/eval_skempi2_5fold.py`, a standard argparse CLI:

```powershell
python -m simba.eval_skempi2_5fold `
    --pool combined `           # skempi2 | combined
    --split cold_target `       # complex | seq_similarity | cold_target
    --model pairformer `        # concat_mlp | cross_attn_mlp | pairformer
    --backbone facebook/esm2_t33_650M_UR50D `
    --epochs 40 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 `
    --val-frac 0.15 --seed 0 `
    --out results.json --predictions-out results.predictions.json
```

| Flag | Values | Meaning |
|---|---|---|
| `--pool` | `skempi2`, `combined` | `skempi2`: expanded SKEMPI2 only (7,046 ex. / 340 cplx.). `combined`: + Kastritis-lineage benchmark46 + ATLAS TCR:pMHC (7,247 ex. / 541 cplx.) |
| `--split` | `complex`, `seq_similarity`, `cold_target` | Fold-construction protocol, increasing strictness (see paper Section 4.3) |
| `--model` | `concat_mlp`, `cross_attn_mlp`, `pairformer` | Architecture under test; `pairformer` is SIMBA |
| `--backbone` | any ESM-2 HF checkpoint name | Frozen encoder |
| `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--val-frac`, `--seed` | numeric | Standard training hyperparameters |
| `--out` | path | Per-fold summary metrics (JSON) |
| `--predictions-out` | path | Raw per-example predictions, for later bootstrap CIs / figures without retraining |

To reproduce the full ablation matrix (all 3 models x all 3 splits, plus the SKEMPI2-only
complex-split check) in one go:

```powershell
.\scripts\run_main_results.ps1
```

Every other experiment in the paper (Kastritis-lineage vs. PPI-Graphomer, AbDesign, the MINT
comparison, the frozen-backbone ablation, zero-/few-shot generalization, chain-order
symmetrization, gating, the direct-ΔΔG-head control, the backbone x head factorial, and the
interpretability analysis) has its own argparse-documented script under `simba/eval_*.py` /
`simba/interpretability_*.py`; run all of them via:

```powershell
.\scripts\run_all_results.ps1
```

Both scripts are self-documenting — each block prints which experiment it's running and what
flags it's using.

## 3. Figures

```powershell
.\scripts\make_figures.ps1
```

Regenerates every figure from the result JSONs produced above. Pure plotting, no GPU needed.

## Tests

```powershell
python -m pytest tests/
```

## Determinism

All split/exclusion rules are seeded and documented in the paper (Section 4.3, "Reproducibility
summary"). Multi-seed results use seeds `[0, 1, 2]` throughout. GPU training is not bitwise-
deterministic across different GPUs/cuDNN versions, so exact digits may drift in the last decimal
place; reported comparisons and their statistical significance (bootstrap CIs, paired permutation
tests) are robust to this.

## Citation

```bibtex
@inproceedings{simba2026,
  title     = {SIMBA: A Sequence-Only, Frozen-Backbone Cross-Attention Model
               for Protein-Protein Binding Affinity and Mutation-Effect Prediction},
  author    = {Author names omitted},
  year      = {2026}
}
```
