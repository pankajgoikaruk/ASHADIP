param(
  [string]$DataRoot   = "data\moth_sounds",
  [string]$CacheDir   = "data_cache",
  [string]$Config     = "configs\audio_moth.yaml",
  [string]$RunsRoot   = "runs",
  [string]$Variant    = "V0",
  [string]$Device     = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec     = 0.5
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

# Start wall-clock timer for the whole pipeline
$pipelineStart = Get-Date

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host "  DataRoot   = $DataRoot"   -ForegroundColor DarkGray
Write-Host "  CacheDir   = $CacheDir"   -ForegroundColor DarkGray
Write-Host "  RunsRoot   = $RunsRoot"   -ForegroundColor DarkGray
Write-Host "  Variant    = $Variant"    -ForegroundColor DarkGray
Write-Host "  Device     = $Device"     -ForegroundColor DarkGray
Write-Host "  SegmentSec = $SegmentSec" -ForegroundColor DarkGray
Write-Host "  HopSec     = $HopSec"     -ForegroundColor DarkGray

# --------------------- 1) Prep segments ---------------------
Write-Host "`n[1/9] Prep segments..." -ForegroundColor Yellow
python -m scripts.prep_segments `
  --root $DataRoot `
  --cache $CacheDir `
  --sr 16000 `
  --segment_sec $SegmentSec `
  --hop $HopSec `
  --silence_dbfs -40 `
  --bandpass 100 3000

# --------------------- 2) Extract features ---------------------
Write-Host "`n[2/9] Extract features..." -ForegroundColor Yellow
python -m scripts.extract_features `
  --cache $CacheDir `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn

# --------------------- 3) Train ExitNet ---------------------
Write-Host "`n[3/9] Train ExitNet..." -ForegroundColor Yellow
python -m training.train --config $Config

# Find the latest run directory under RunsRoot
$runDir = Get-ChildItem -Directory $RunsRoot | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $runDir) { throw "No run directory found under '$RunsRoot'." }
$runPath = $runDir.FullName
$runId   = Split-Path $runPath -Leaf
Write-Host "Using run: $runPath" -ForegroundColor Green

# Common paths for later steps
$SegCsv   = Join-Path $CacheDir "segments.csv"
$FeatRoot = Join-Path $CacheDir "features"

# --------------------- 4) Calibrate temperatures ---------------------
Write-Host "`n[4/9] Calibrate temperatures..." -ForegroundColor Yellow
python -m training.calibrate `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot

# --------------------- 5) Select threshold (tau) ---------------------
Write-Host "`n[5/9] Select threshold (tau)..." -ForegroundColor Yellow
python -m training.thresholds_offline `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot

# --------------------- 6) Policy test ---------------------
Write-Host "`n[6/9] Policy test..." -ForegroundColor Yellow
python -m scripts.policy_test `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot

# --------------------- 7) Summarise run ---------------------
Write-Host "`n[7/9] Summarise run..." -ForegroundColor Yellow
python -m scripts.summarize_run `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot

# --------------------- 8) Analyse run (CM, ROC, learning curves) ---------------------
Write-Host "`n[8/9] Analyse run (CM, ROC, learning curves)..." -ForegroundColor Yellow
python -m scripts.analyse_run `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot

# --------------------- 9) Profile latency (on-device-style) ---------------------
Write-Host "`n[9/9] Profile latency (on-device-style)..." -ForegroundColor Yellow
python -m scripts.profile_latency `
  --run_dir      $runPath `
  --segments_csv $SegCsv `
  --features_root $FeatRoot `
  --variant $Variant `
  --device  $Device

# --------------------- Pipeline timing & logging ---------------------
$pipelineEnd   = Get-Date
$elapsed       = $pipelineEnd - $pipelineStart
$totalSeconds  = [Math]::Round($elapsed.TotalSeconds, 2)
$totalMinutes  = [Math]::Round($elapsed.TotalMinutes, 2)
$timestampIso  = Get-Date -Format o

Write-Host ""
Write-Host ("Total wall-clock time: {0} seconds (~{1} minutes)" -f $totalSeconds, $totalMinutes) -ForegroundColor Cyan

# Ensure analysis directory exists
$analysisDir  = "analysis"
New-Item -ItemType Directory -Path $analysisDir -ErrorAction SilentlyContinue | Out-Null

# CSV path for pipeline runtimes
$runtimeCsv = Join-Path $analysisDir "pipeline_runtime.csv"

# If CSV doesn't exist, write header
if (-not (Test-Path $runtimeCsv)) {
    "timestamp,variant,segment_sec,hop_sec,device,cache_dir,runs_root,run_id,total_seconds,total_minutes" | Out-File $runtimeCsv -Encoding UTF8
}

# Append one row for this run
$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
    $timestampIso, `
    $Variant, `
    $SegmentSec, `
    $HopSec, `
    $Device, `
    $CacheDir, `
    $RunsRoot, `
    $runId, `
    $totalSeconds, `
    $totalMinutes

Add-Content -Path $runtimeCsv -Value $csvLine

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
Write-Host "You can now regenerate LaTeX tables if needed (classification, variants, on-device)." -ForegroundColor DarkGray
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

# Append one row for this run
$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
    $timestampIso, `
    $Variant, `
    $SegmentSec, `
    $HopSec, `
    $Device, `
    $CacheDir, `
    $RunsRoot, `
    $runId, `
    $totalSeconds, `
    $totalMinutes

Add-Content -Path $runtimeCsv -Value $csvLine
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

# --------------------- 10) Generate reports & LaTeX tables ---------------------
Write-Host "`n[10/10] Generate reports & LaTeX tables..." -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File scripts\run_reports.ps1 `
  -RunDir $runPath `
  -Variant $Variant `
  -DeviceFilter $Device

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
Write-Host "All reports + LaTeX tables up to date (classification, variants, on-device)." -ForegroundColor DarkGray


