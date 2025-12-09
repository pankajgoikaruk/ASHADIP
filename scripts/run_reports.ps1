param(
  [Parameter(Mandatory = $true)]
  [string]$RunDir,                # e.g. "runs\20251208_175617"

  [string]$Variant      = "V0",   # e.g. "V0", "V1", ...
  [string]$DeviceFilter = "cpu",  # currently used only as a label in logs

  [string]$SegmentsCsv  = "data_cache\segments.csv",
  [string]$FeaturesRoot = "data_cache\features"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

Write-Host "== ASHADIP: generate reports & tables ==" -ForegroundColor Cyan
Write-Host "  RunDir       = $RunDir"
Write-Host "  Variant      = $Variant"
Write-Host "  DeviceFilter = $DeviceFilter"
Write-Host "  SegmentsCsv  = $SegmentsCsv"
Write-Host "  FeaturesRoot = $FeaturesRoot"
Write-Host ""

# Normalise RunDir to full path
$runPath = Resolve-Path $RunDir
$runPath = $runPath.Path

if (-not (Test-Path $runPath)) {
  throw "Run directory not found: $runPath"
}

# Ensure analysis folders exist
New-Item -ItemType Directory -Path "analysis" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "analysis\tables" -ErrorAction SilentlyContinue | Out-Null

# ---------------------------------------------------------------------------
# [1/5] Ensure report.json exists (test-set classification per exit)
# ---------------------------------------------------------------------------
Write-Host "[1/5] Evaluating test set (report.json)..." -ForegroundColor Yellow

python -m training.eval `
  --run_dir "$runPath" `
  --segments_csv "$SegmentsCsv" `
  --features_root "$FeaturesRoot"

Write-Host "  -> training.eval finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [2/5] Summarise run (summary.json + calibration plots)
#      We set KMP_DUPLICATE_LIB_OK=TRUE to avoid the OMP runtime error.
# ---------------------------------------------------------------------------
Write-Host "`n[2/5] Summarising run (summary.json, calibration plots)..." -ForegroundColor Yellow

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python -m scripts.summarize_run `
  --run_dir "$runPath" `
  --segments_csv "$SegmentsCsv" `
  --features_root "$FeaturesRoot"

Write-Host "  -> summarize_run finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [3/5] Global variants summary CSV (analysis/all_runs_summary.csv)
# ---------------------------------------------------------------------------
Write-Host "`n[3/5] Updating global variants summary (analysis/all_runs_summary.csv)..." -ForegroundColor Yellow

python -m scripts.compare_variants --root .

Write-Host "  -> compare_variants finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [4/5] Per-run classification LaTeX table for this variant/run
#       (analysis/tables/<Variant>_classification_table.tex)
# ---------------------------------------------------------------------------
Write-Host "`n[4/5] Generating classification LaTeX table for this run..." -ForegroundColor Yellow

$analysisJson = Join-Path $runPath "analysis_run.json"
if (-not (Test-Path $analysisJson)) {
  throw "analysis_run.json not found at $analysisJson. Did analyse_run.py run?"
}

$outClsTex = "analysis\tables\{0}_classification_table.tex" -f $Variant
$runLabel  = "$Variant baseline"

python -m scripts.analysis_to_latex `
  --analysis_json "$analysisJson" `
  --out_tex "$outClsTex" `
  --run_label "$runLabel"

Write-Host "  -> Wrote classification table: $outClsTex" -ForegroundColor Green

# ---------------------------------------------------------------------------
# [5/5] Cross-variant summary + on-device performance tables
#       (variants_avg_summary_table.tex, on_device_performance_table.tex)
# ---------------------------------------------------------------------------
Write-Host "`n[5/5] Generating cross-variant & on-device LaTeX tables..." -ForegroundColor Yellow

# Variants summary table (policy accuracy vs compute saving / exit mix)
$allRunsCsv = "analysis\all_runs_summary.csv"
if (Test-Path $allRunsCsv) {
  $variantsTex = "analysis\tables\variants_avg_summary_table.tex"
  python -m scripts.variants_to_latex `
    --summary_csv "$allRunsCsv" `
    --out_tex "$variantsTex"
  Write-Host "  -> Wrote variants summary table: $variantsTex" -ForegroundColor Green
}
else {
  Write-Host "  (skip) analysis\all_runs_summary.csv not found; run compare_variants.py first." -ForegroundColor DarkYellow
}

# On-device performance table (if profiling CSV exists)
$onDevCsv = "analysis\on_device_summary.csv"
if (Test-Path $onDevCsv) {
  $onDevTex = "analysis\tables\on_device_performance_table.tex"
  python -m scripts.ondevice_to_latex `
    --summary_csv "$onDevCsv" `
    --out_tex "$onDevTex"
  Write-Host "  -> Wrote on-device performance table: $onDevTex" -ForegroundColor Green
}
else {
  Write-Host "  (skip) analysis\on_device_summary.csv not found; run profile_latency.py / run_full.ps1 first." -ForegroundColor DarkYellow
}

Write-Host "`n== Done. Reports & LaTeX tables updated for $Variant / $runPath ==" -ForegroundColor Cyan
