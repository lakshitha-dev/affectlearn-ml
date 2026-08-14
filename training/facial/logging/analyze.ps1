<#
Analyze logged predictions from run_camera_log.ps1.

Usage:
  .\analyze.ps1
  .\analyze.ps1 -CsvFile "my_run.csv"
#>

param(
    [string]$CsvFile = "predictions.csv"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir

# Build full path
if (-not [System.IO.Path]::IsPathRooted($CsvFile)) {
    $CsvFile = Join-Path (Get-Location) $CsvFile
}

if (-not (Test-Path $CsvFile)) {
    Write-Host "ERROR: CSV file not found: $CsvFile" -ForegroundColor Red
    Write-Host "Try: .\analyze.ps1 -CsvFile `"predictions.csv`"" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== AffectLearn Prediction Analysis ===" -ForegroundColor Green
Write-Host "Analyzing: $CsvFile"
Write-Host ""

# Run the Python analysis script
$AnalysisScript = Join-Path $ParentDir "analyze_predictions.py"
Push-Location $ParentDir
python "$AnalysisScript" "$CsvFile"
Pop-Location
