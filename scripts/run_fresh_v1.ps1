$code = @'
# ASHADIP v1 end-to-end runner (steps 2→7)
param(
  [string]$DataRoot="data\moth_sounds",
  [string]$CacheDir="data_cache_v1",
  [string]$Config="configs\audio_moth.yaml"
)
$ErrorActionPreference="Stop"
$env:PYTHONPATH=(Get-Location).Path
Write-Host "== ASHADIP v1: fresh run ==" -ForegroundColor Cyan

Write-Host "`n[1/6] Prep segments..." -ForegroundColor Yellow
python -m scripts.prep_segments --root $DataRoot --cache $CacheDir --sr 16000 --segment_sec 1.0 --hop 0.5 --silence_dbfs -40 --bandpass 100 3000

Write-Host "`n[2/6] Extract features..." -ForegroundColor Yellow
python -m scripts.extract_features --cache $CacheDir --n_mels 64 --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn

Write-Host "`n[3/6] Train ExitNet..." -ForegroundColor Yellow
python -m training.train --config $Config

# Prefer V1 runs directory if present
$runDir = (Get-ChildItem -Directory "runs_v1*" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if (-not $runDir) { $runDir = Get-ChildItem -Directory "runs*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 }
if (-not $runDir) { throw "No run directory found." }
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
'@
New-Item -ItemType Directory -Path scripts -ErrorAction SilentlyContinue | Out-Null
Set-Content -Path scripts\run_fresh_v1.ps1 -Value $code -Encoding UTF8
