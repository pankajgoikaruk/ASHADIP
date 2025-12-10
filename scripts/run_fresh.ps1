# NOTE: Legacy script kept for reference.
# For the full modern pipeline (training + analysis + profiling),
# please use scripts\run_full.ps1 instead.


# ASHADIP v0 end-to-end runner (steps 2→7)
param([string]$DataRoot="data\moth_sounds",[string]$CacheDir="data_cache",[string]$Config="configs\audio_moth.yaml")
$ErrorActionPreference="Stop"
$env:PYTHONPATH=(Get-Location).Path
Write-Host "== ASHADIP v0: fresh run ==" -ForegroundColor Cyan

Write-Host "`n[1/6] Prep segments..." -ForegroundColor Yellow
python -m scripts.prep_segments --root $DataRoot --cache $CacheDir --sr 16000 --segment_sec 1.0 --hop 0.5 --silence_dbfs -40 --bandpass 100 3000

Write-Host "`n[2/6] Extract features..." -ForegroundColor Yellow
python -m scripts.extract_features --cache $CacheDir --n_mels 64 --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn

Write-Host "`n[3/6] Train ExitNet..." -ForegroundColor Yellow
python -m training.train --config $Config

$runDir = Get-ChildItem -Directory "runs" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $runDir) { throw "No run directory found under 'runs'." }
$runPath = $runDir.FullName
Write-Host "Using run: $runPath" -ForegroundColor Green

Write-Host "`n[4/6] Calibrate temperatures..." -ForegroundColor Yellow
python -m training.calibrate --run_dir $runPath --segments_csv (Join-Path $CacheDir "segments.csv") --features_root (Join-Path $CacheDir "features")

Write-Host "`n[5/6] Select threshold (τ)..." -ForegroundColor Yellow
python -m training.thresholds_offline --run_dir $runPath --segments_csv (Join-Path $CacheDir "segments.csv") --features_root (Join-Path $CacheDir "features")

Write-Host "`n[6/6] Policy test & summarize..." -ForegroundColor Yellow
python -m scripts.policy_test --run_dir $runPath --segments_csv (Join-Path $CacheDir "segments.csv") --features_root (Join-Path $CacheDir "features")
python -m scripts.summarize_run --run_dir $runPath --segments_csv (Join-Path $CacheDir "segments.csv") --features_root (Join-Path $CacheDir "features")

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
