# Overnight runner: runs all 5 reviewer-response experiments sequentially,
# in the same conda env used throughout this project, with a consistent,
# warning-free environment.
#
# Usage (from D:\SIMBA, in PowerShell):
#   .\scripts\run_overnight_experiments.ps1
#
# Each experiment:
#   - writes its own timestamped log under logs\overnight_<timestamp>\
#   - writes its own *_results.json (raw per-example predictions included,
#     so every result can be replotted/re-bootstrapped later without rerunning)
#   - runs independently: if one experiment errors out, the script logs it
#     and moves on to the next rather than aborting the whole night's run.
#
# At the end, prints a one-line PASS/FAIL summary for all 5, plus total wall time.

$ErrorActionPreference = "Continue"

$PYTHON = "D:/condaenvs/simba/python.exe"
$REPO_ROOT = "D:\SIMBA"
Set-Location $REPO_ROOT

# Consistent environment for every run:
#   - PYTORCH_CUDA_ALLOC_CONF cleared: "expandable_segments" is a Linux-only
#     CUDA allocator feature; leaving it set produces a UserWarning on every
#     run under Windows PyTorch builds and was already fixed at the top of
#     each script, but cleared here too so it's consistent regardless of
#     what's inherited from the calling shell.
#   - PYTHONUNBUFFERED so each script's `flush=True` prints land in the log
#     file immediately instead of being buffered until the process exits.
$env:PYTORCH_CUDA_ALLOC_CONF = ""
$env:PYTHONUNBUFFERED = "1"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\overnight_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "=== Overnight experiment run started $timestamp ===" -ForegroundColor Cyan
Write-Host "Logs: $logDir"
Write-Host ""

# Each entry: (name, log file, full command line as an array of args)
$experiments = @(
    @{
        Name = "symmetrization_test"
        Log  = "$logDir\01_symmetrization_test.log"
        Args = @("-m", "simba.eval_symmetrization_test", "--seeds", "0", "1", "2",
                 "--out", "symmetrization_test_results.json")
    },
    @{
        Name = "direct_ddg_head"
        Log  = "$logDir\02_direct_ddg_head.log"
        Args = @("-m", "simba.eval_direct_ddg_head",
                 "--out", "direct_ddg_head_results.json")
    },
    @{
        Name = "mint_protocol_cross"
        Log  = "$logDir\03_mint_protocol_cross.log"
        Args = @("-m", "simba.eval_mint_protocol_cross", "--config", "both",
                 "--out", "mint_protocol_cross_results.json")
    },
    @{
        Name = "gate_ablation"
        Log  = "$logDir\04_gate_ablation.log"
        Args = @("-m", "simba.eval_gate_ablation", "--split", "cold_target",
                 "--out", "gate_ablation_results.json")
    },
    @{
        Name = "memory_size_sweep"
        Log  = "$logDir\05_memory_size_sweep.log"
        Args = @("-m", "simba.eval_memory_size_sweep", "--memory-sizes", "8", "16", "24", "32", "48",
                 "--split", "cold_target", "--out", "memory_size_sweep_results.json")
    }
)

$results = @()
$overallStart = Get-Date

foreach ($exp in $experiments) {
    $name = $exp.Name
    Write-Host "--- [$name] starting (see $($exp.Log) for live output) ---" -ForegroundColor Yellow
    $start = Get-Date
    $success = $false
    try {
        # Routed through cmd's own redirection rather than PowerShell's
        # `2>&1` on a native exe: PowerShell 5.1 wraps every stderr line from
        # a native process as a NativeCommandError when redirected this way,
        # which would spam the console with these scripts' ordinary stderr
        # chatter (tqdm bars, HF load messages, PyTorch warnings) and can
        # falsely flip `$?` to false even on a clean exit. `$LASTEXITCODE`
        # after a cmd.exe call still reflects the real python.exe exit code.
        $argsJoined = ($exp.Args | ForEach-Object { '"' + $_ + '"' }) -join ' '
        $cmdLine = '"' + $PYTHON + '" ' + $argsJoined + ' > "' + $exp.Log + '" 2>&1'
        & cmd /c $cmdLine
        if ($LASTEXITCODE -eq 0) {
            $success = $true
        } else {
            Write-Host "[$name] exited with code $LASTEXITCODE -- see $($exp.Log)" -ForegroundColor Red
        }
    } catch {
        Write-Host "[$name] threw an exception: $_" -ForegroundColor Red
        Add-Content -Path $exp.Log -Value "`n`nPOWERSHELL EXCEPTION: $_"
    }
    $elapsed = (Get-Date) - $start
    $results += [PSCustomObject]@{
        Name = $name
        Success = $success
        ElapsedMinutes = [math]::Round($elapsed.TotalMinutes, 1)
        Log = $exp.Log
    }
    $status = if ($success) { "OK" } else { "FAILED" }
    Write-Host "--- [$name] $status ($([math]::Round($elapsed.TotalMinutes, 1)) min) ---" -ForegroundColor $(if ($success) { "Green" } else { "Red" })
    Write-Host ""
}

$overallElapsed = (Get-Date) - $overallStart

Write-Host "=== Overnight run summary ===" -ForegroundColor Cyan
foreach ($r in $results) {
    $status = if ($r.Success) { "PASS" } else { "FAIL" }
    $color = if ($r.Success) { "Green" } else { "Red" }
    Write-Host ("  [{0,-4}] {1,-25} {2,6} min   log: {3}" -f $status, $r.Name, $r.ElapsedMinutes, $r.Log) -ForegroundColor $color
}
Write-Host ""
Write-Host "Total wall time: $([math]::Round($overallElapsed.TotalHours, 2)) hours"
Write-Host "Result JSONs written to $REPO_ROOT (one per experiment, listed above)."

$summaryPath = "$logDir\00_SUMMARY.txt"
$results | Format-Table -AutoSize | Out-String | Set-Content -Path $summaryPath
Add-Content -Path $summaryPath -Value "`nTotal wall time: $([math]::Round($overallElapsed.TotalHours, 2)) hours"
Write-Host "Summary written to $summaryPath"
