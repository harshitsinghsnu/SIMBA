# Reproduces SIMBA's headline result (paper Section 5.1, Figure 3A-C, Table S3):
# the 3-architecture x 3-split-protocol ablation on the combined general-PPI
# pool (7,247 examples, 541 complexes) -- this is the paper's central,
# most-referenced comparison and should be run first.
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\run_main_results.ps1
#
# Expected headline numbers (Pearson correlation, Table S3):
#   Complex (random):            Concat+MLP 0.6574  Cross-attn+MLP 0.6627  SIMBA 0.6993
#   Sequence-similarity cluster: Concat+MLP 0.5150  Cross-attn+MLP 0.5327  SIMBA 0.5539
#   Cold-target:                 Concat+MLP 0.6146  Cross-attn+MLP 0.6006  SIMBA 0.6582
#
# Each (model, split) cell is 5-fold cross-validation; results and raw
# out-of-fold predictions are written as
# skempi2_5fold_combined_<split>_<model>.json / .predictions.json in the
# repo root (matching the paper's own convention -- these are exactly the
# files make_panel1_figure.py reads to build Figure 3).

$ErrorActionPreference = "Continue"
$PYTHON = "python"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$env:PYTORCH_CUDA_ALLOC_CONF = ""
$env:PYTHONUNBUFFERED = "1"

$models = @("concat_mlp", "cross_attn_mlp", "pairformer")
$splits = @(
    @{Name="complex"; Suffix=""},
    @{Name="seq_similarity"; Suffix="seqsim_"},
    @{Name="cold_target"; Suffix="coldtarget_"}
)

foreach ($split in $splits) {
    foreach ($model in $models) {
        $out = "skempi2_5fold_combined_$($split.Suffix)$model.json"
        Write-Host "=== pool=combined split=$($split.Name) model=$model -> $out ===" -ForegroundColor Cyan
        & $PYTHON -m simba.eval_skempi2_5fold --pool combined --split $($split.Name) --model $model --out $out
    }
}

# SKEMPI2-only pool, complex split (Table S3's fourth row)
foreach ($model in $models) {
    $out = "skempi2_5fold_$model.json"
    Write-Host "=== pool=skempi2 split=complex model=$model -> $out ===" -ForegroundColor Cyan
    & $PYTHON -m simba.eval_skempi2_5fold --pool skempi2 --split complex --model $model --out $out
}

Write-Host "`nDone. Compare the printed Pearson/Spearman/RMSE against Table S3 in the paper." -ForegroundColor Green
Write-Host "Next: run .\scripts\run_all_results.ps1 for the remaining paper experiments,"
Write-Host "or .\scripts\make_figures.ps1 to regenerate Figures 2-6 from these results."
