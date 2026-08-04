# Reproduces every remaining number/figure reported in the paper, beyond
# the main 3-split architecture ablation (run_main_results.ps1 first).
# Each block is independent -- comment out any you don't need. Expected
# numbers are noted inline; compare against Table S2 and Figures 3-6.
#
# Usage (from the repo root, in PowerShell, after run_main_results.ps1):
#   .\scripts\run_all_results.ps1

$ErrorActionPreference = "Continue"
$PYTHON = "python"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTORCH_CUDA_ALLOC_CONF = ""
$env:PYTHONUNBUFFERED = "1"

# --- Absolute affinity: Kastritis-lineage benchmark46 vs. PPI-Graphomer ---
# Expected: Pearson 0.788 +/- 0.008 over 3 seeds (Table S2, row 1)
Write-Host "=== Kastritis-lineage (benchmark46), 3 seeds ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_affinity_extended --mode multiseed --test-sets benchmark46 --seeds 0 1 2 --out affinity_extended_results.json

# --- Pairformer depth selection (Section 3.4) ---
# Expected: Pearson 0.755 (1 layer) / 0.683 (2 layers) / 0.636 (3 layers)
Write-Host "=== Pairformer depth ablation, prott_test1 ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_affinity --test-set prott_test1 --n-pairformer-layers 1 --out depth1_results.json
& $PYTHON -m simba.eval_affinity --test-set prott_test1 --n-pairformer-layers 2 --out depth2_results.json
& $PYTHON -m simba.eval_affinity --test-set prott_test1 --n-pairformer-layers 3 --out depth3_results.json

# --- Mutation-effect: AbDesign zero-shot holdout + MINT-protocol split ---
# Expected AbDesign: Pearson/Spearman 0.352+/-0.014 / 0.371+/-0.027 (Table S2, row 2)
# Expected MINT-protocol SIMBA: overall Pearson/Spearman 0.528/0.435 (Table S2, row 3)
Write-Host "=== AbDesign + MINT-protocol split (SIMBA's own numbers) ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_ddg_affinity --benchmarks abdesign mint_split --out ddg_multiseed_results.json

# --- MINT's own reported method, reproduced on the identical rows/folds ---
# Expected: overall Pearson/Spearman 0.375/0.282
# Requires third_party/mint (see README) and MINT's released checkpoint.
Write-Host "=== MINT's own method (requires third_party/mint) ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_mint_own_method_skempi

# --- Fisher intra-target correlation (Figure 3D) ---
# Expected: overall (pooled) Pearson 0.681, Fisher-z-corrected intra-target mean 0.269
# Requires skempi2_5fold_combined_pairformer.predictions.json from run_main_results.ps1.
# Writes to figures/fisher_intra_target_results.json.
Write-Host "=== Fisher intra-target correlation ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path figures | Out-Null
& $PYTHON -m simba.eval_fisher_intra_target

# --- Frozen-backbone ablation (Figure 4C) ---
# Expected: ESM-2 0.658+/-0.047, MINT 0.647+/-0.054, AbLang2 0.499+/-0.107
# Requires third_party/mint for the MINT backbone. Cold-target split is hardcoded.
Write-Host "=== Backbone ablation (ESM-2 / MINT / AbLang2), cold-target split ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_plm_ablation_cold --backbone esm2
& $PYTHON -m simba.eval_plm_ablation_cold --backbone mint
& $PYTHON -m simba.eval_plm_ablation_cold --backbone ablang2

# --- Zero-shot / few-shot generalization (Figure 4D, Section 7.2) ---
# Requires a trained checkpoint -- run train_final_checkpoint.py first if
# checkpoints/simba_pairformer_combined.pt does not already exist.
Write-Host "=== Zero-shot / few-shot generalization ===" -ForegroundColor Cyan
if (-not (Test-Path "checkpoints\simba_pairformer_combined.pt")) {
    Write-Host "  training final checkpoint first (needed for interpretability + generalization) ..." -ForegroundColor Yellow
    & $PYTHON -m simba.train_final_checkpoint
}
& $PYTHON -m simba.eval_novel_generalization

# --- Figure 5: additional robustness and design controls ---
# (A/B) chain-order symmetrization -- expected f(A,B)=0.788, f(B,A)=0.776, avg=0.808
Write-Host "=== Chain-order symmetrization (3 seeds) ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_symmetrization_test --seeds 0 1 2 --out symmetrization_test_results.json

# (C) gating ablation -- expected gated 0.651/0.661, ungated 0.649/0.651 (Pearson/Spearman)
Write-Host "=== Gating ablation, cold-target split ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_gate_ablation --split cold_target --out gate_ablation_results.json

# (D) direct joint ddG head vs. two-pass subtraction -- expected subtraction wins on
#     overall Pearson/Spearman (0.523/0.428 vs. 0.508/0.397, single seed)
Write-Host "=== Direct ddG head vs. two-pass subtraction, RDE-PPI ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_direct_ddg_head --out direct_ddg_head_results.json

# (E) backbone x head factorial -- expected ESM-2+Pairformer 0.528, ESM-2+Ridge 0.429,
#     MINT+Pairformer 0.488, MINT+Ridge 0.375. Requires third_party/mint.
Write-Host "=== Backbone x head factorial (MINT-protocol split) ===" -ForegroundColor Cyan
& $PYTHON -m simba.eval_mint_protocol_cross --config both --out mint_protocol_cross_results.json

# --- Figure 6: interpretability (Integrated Gradients on 1CSE, 3BTD) ---
Write-Host "=== Interpretability: 1CSE + case-study complexes (incl. 3BTD) ===" -ForegroundColor Cyan
& $PYTHON -m simba.interpretability_1cse
& $PYTHON -m simba.interpretability_case_studies

Write-Host "`nAll experiments complete. Run .\scripts\make_figures.ps1 to regenerate Figures 2-6." -ForegroundColor Green
