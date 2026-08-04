# Regenerates every figure in the paper from already-computed result JSONs
# (run_main_results.ps1 + run_all_results.ps1 must have been run first).
# Pure plotting -- no GPU/training involved, runs in seconds.
#
# Usage (from the repo root, in PowerShell):
#   .\scripts\make_figures.ps1

$PYTHON = "python"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force -Path figures | Out-Null

Write-Host "Figure 1 (architecture diagram) ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_architecture_figure

Write-Host "Figure 2 (dataset characteristics) ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_dataset_analysis_figure

Write-Host "Figure 3 (architecture ablation + Fisher + confirmed wins) ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_panel1_figure

Write-Host "Figure 4 sub-panels (scatter/MINT/backbone/few-shot) then composite ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_mint_comparison_figure
& $PYTHON -m simba.make_plm_ablation_cold_figure
& $PYTHON -m simba.make_fewshot_scaling_figure
& $PYTHON -m simba.make_panel2_figure

Write-Host "Figure 5 (chain-order, gating, direct-head, backbone x head) ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_reviewer_experiments_figure

Write-Host "Figure 6 (interpretability, 1CSE + 3BTD) ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_panel3_figure

Write-Host "Supplementary: SKEMPI2-only pool ablation figure ..." -ForegroundColor Cyan
& $PYTHON -m simba.make_skempi2_only_figure

Write-Host "`nAll figures written to .\figures\" -ForegroundColor Green
