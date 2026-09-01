<#
Run camera engagement demo with CSV logging.

Usage:
  .\run_camera_log.ps1
  .\run_camera_log.ps1 -ModelsDir "G:\My Drive\affectlearn-ml\models" -Output "my_run.csv"
#>

param(
    [string]$ModelsDir = "G:\My Drive\affectlearn-ml\models",
    [string]$Output = "predictions.csv",
    [int]$Camera = 0
)

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentDir = Split-Path -Parent $ScriptDir

# Check if models directory exists
if (-not (Test-Path $ModelsDir)) {
    Write-Host "ERROR: Models directory not found: $ModelsDir" -ForegroundColor Red
    Write-Host "Try: .\run_camera_log.ps1 -ModelsDir `"C:\path\to\models`"" -ForegroundColor Yellow
    exit 1
}

# Check if Python script exists
$PythonScript = Join-Path $ParentDir "predict_camera_log.py"
if (-not (Test-Path $PythonScript)) {
    Write-Host "ERROR: predict_camera_log.py not found at: $PythonScript" -ForegroundColor Red
    exit 1
}

Write-Host "=== AffectLearn Camera Logging Demo ===" -ForegroundColor Green
Write-Host "Models dir: $ModelsDir"
Write-Host "Output CSV: $Output"
Write-Host "Camera: $Camera"
Write-Host "Press 'q' in the video window to stop."
Write-Host ""

# Run the Python script
Push-Location $ParentDir
python predict_camera_log.py --models-dir "$ModelsDir" --output "$Output" --camera $Camera
Pop-Location

Write-Host ""
Write-Host "Done! Results saved to: $(Join-Path (Get-Location) $Output)" -ForegroundColor Green
