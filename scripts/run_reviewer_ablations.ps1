# Memory and Pairformer-depth ablations (Section "Ablation of the interface memory"
# and the depth discussion; Table "Memory and depth ablations").
# Cold-target split of the combined general-PPI pool, 3 seeds, identical folds.
# This is the slowest block in the repo (~11-12 h per depth run on a 16 GB GPU).
# Each run writes its JSON as soon as a configuration finishes and is skipped if
# its output already exists, so the script can be interrupted and re-run.
#
# Usage (from the repo root, in PowerShell, after scripts/download_data.py):
#   .\scripts\run_reviewer_ablations.ps1
#
# Expected pooled Pearson (mean over 3 seeds):
#   no memory 0.668 | memory only (no Pairformer-lite) 0.551
#   memory M=8/16/24/32: 0.658 / 0.656 / 0.654 / 0.663
#   M=24 with 1/2/3 Pairformer-lite layers: 0.654 / 0.656 / 0.659

$ErrorActionPreference = "Continue"
$PYTHON = "python"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTORCH_CUDA_ALLOC_CONF = ""
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path results | Out-Null

function Invoke-Ablation($out, $extra) {
    if (Test-Path $out) { Write-Host "skip $out (exists)" -ForegroundColor DarkGray; return }
    Write-Host "=== $out ===" -ForegroundColor Cyan
    & $PYTHON -m simba.eval_memory_size_sweep @extra --out $out
}

foreach ($seed in 0, 1, 2) {
    # memory vs. no memory vs. no Pairformer-lite
    Invoke-Ablation "results/ablation_no_memory_s$seed.json"   @("--variant", "no_memory",   "--memory-sizes", "0",  "--seed", $seed)
    Invoke-Ablation "results/ablation_memory_only_s$seed.json" @("--variant", "memory_only", "--memory-sizes", "24", "--seed", $seed)
    # memory-size sweep (M=48 exceeds 16 GB; swept up to 32)
    Invoke-Ablation "results/ablation_memory_size_s$seed.json" @("--variant", "memory", "--memory-sizes", "8", "16", "24", "32", "--seed", $seed)
    # Pairformer-lite depth (1 layer is the M=24 row of the sweep above)
    foreach ($layers in 2, 3) {
        Invoke-Ablation "results/ablation_depth${layers}_s$seed.json" @("--variant", "memory", "--layers", $layers, "--memory-sizes", "24", "--seed", $seed)
    }
}

Write-Host "`nAblation JSONs written to .\results\" -ForegroundColor Green
