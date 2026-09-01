<#
Master launcher for AffectLearn camera logging workflow.

Interactive menu to:
  1. Run camera demo with logging
  2. Analyze saved predictions
  3. View recent logs
  4. Open models directory
#>

function Show-Menu {
    Clear-Host
    Write-Host "=== AffectLearn Camera Logging Workflow ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Run camera demo with logging"
    Write-Host "2. Analyze predictions (CSV)"
    Write-Host "3. View recent predictions"
    Write-Host "4. Open logging folder"
    Write-Host "5. Exit"
    Write-Host ""
}

function Run-CameraDemo {
    Write-Host "Enter models directory path (or press Enter for default):" -ForegroundColor Yellow
    Write-Host "Default: G:\My Drive\affectlearn-ml\models"
    $modelsDir = Read-Host

    if ([string]::IsNullOrWhiteSpace($modelsDir)) {
        $modelsDir = "G:\My Drive\affectlearn-ml\models"
    }

    Write-Host "Enter output CSV filename (or press Enter for 'predictions.csv'):" -ForegroundColor Yellow
    $output = Read-Host
    if ([string]::IsNullOrWhiteSpace($output)) {
        $output = "predictions.csv"
    }

    Write-Host ""
    Write-Host "Starting camera demo..." -ForegroundColor Green
    Write-Host "Press 'q' in the video window to stop."
    Write-Host ""

    & ".\run_camera_log.ps1" -ModelsDir "$modelsDir" -Output "$output"
}

function Analyze-Predictions {
    $csvFiles = Get-ChildItem "*.csv" -ErrorAction SilentlyContinue

    if ($csvFiles.Count -eq 0) {
        Write-Host "No CSV files found in current directory." -ForegroundColor Yellow
        Read-Host "Press Enter to continue"
        return
    }

    Write-Host "Available prediction files:" -ForegroundColor Green
    $csvFiles | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host ""

    $csvFile = Read-Host "Enter filename to analyze (or press Enter for most recent)"
    if ([string]::IsNullOrWhiteSpace($csvFile)) {
        $csvFile = $csvFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Name }
    }

    Write-Host ""
    & ".\analyze.ps1" -CsvFile "$csvFile"
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Show-Recent {
    $csvFiles = Get-ChildItem "*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5

    if ($csvFiles.Count -eq 0) {
        Write-Host "No CSV files found." -ForegroundColor Yellow
    } else {
        Write-Host "Recent prediction files:" -ForegroundColor Green
        $csvFiles | ForEach-Object {
            $lines = @(Get-Content $_.FullName).Count
            Write-Host "  $($_.Name) — $lines rows — $(Get-Date $_.LastWriteTime -Format 'yyyy-MM-dd HH:mm:ss')"
        }
    }
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Open-Folder {
    $folder = Get-Location
    Write-Host "Opening: $folder" -ForegroundColor Green
    explorer.exe $folder
    Start-Sleep -Seconds 1
}

# Main loop
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter choice (1-5)"

    switch ($choice) {
        "1" { Run-CameraDemo }
        "2" { Analyze-Predictions }
        "3" { Show-Recent }
        "4" { Open-Folder }
        "5" {
            Write-Host "Goodbye!" -ForegroundColor Green
            exit 0
        }
        default {
            Write-Host "Invalid choice. Try again." -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
}
