# AffectLearn Camera Logging — PowerShell Scripts

Windows PowerShell wrappers for running the facial engagement demo with CSV logging and analysis.

Located in: `affectlearn-ml/training/facial/logging/`

## Quick Start

```powershell
# Open PowerShell in this folder and run the interactive menu
.\menu.ps1
```

Or directly:

```powershell
# Run camera demo (logs to predictions.csv)
.\run_camera_log.ps1

# Analyze a CSV file
.\analyze.ps1 -CsvFile "predictions.csv"
```

## Scripts

### menu.ps1
Interactive menu launcher. Choose:
1. **Run camera demo** — opens camera, records predictions to CSV
2. **Analyze predictions** — shows stats from a CSV file
3. **View recent** — lists recent CSV files
4. **Open folder** — opens Windows Explorer
5. **Exit**

```powershell
.\menu.ps1
```

### run_camera_log.ps1
Launches the camera demo with logging. 

**Parameters:**
- `-ModelsDir` — path to folder with `cnn_lstm_*.pt` files (default: `G:\My Drive\affectlearn-ml\models`)
- `-Output` — CSV filename to save predictions (default: `predictions.csv`)
- `-Camera` — camera index if default is wrong (default: `0`)

**Examples:**
```powershell
# Default (Google Drive models, predictions.csv)
.\run_camera_log.ps1

# Custom output file
.\run_camera_log.ps1 -Output "test_run_1.csv"

# Custom camera and output
.\run_camera_log.ps1 -Camera 1 -Output "camera1_predictions.csv"
```

**During run:**
- Webcam window shows: face detection box (green) + 4-affect panel with color-coded levels
- Predictions written to CSV every ~half-second
- Press **'q'** in the video window to stop

### analyze.ps1
Parses a logged CSV and prints summary statistics.

**Parameters:**
- `-CsvFile` — path to predictions CSV (default: `predictions.csv`)

**Examples:**
```powershell
# Analyze default file
.\analyze.ps1

# Analyze specific file
.\analyze.ps1 -CsvFile "test_run_1.csv"
```

**Output:** Per-affect stats (confidence range, level distribution %).

## Workflow Example

```powershell
# 1. Start interactive menu
.\menu.ps1

# 2. Choose "1" → run camera demo
#    (Perform behaviors: attentive → look away → eyes closed → attentive)
#    (Press 'q' to stop; results → predictions.csv)

# 3. Choose "2" → analyze predictions
#    (View distribution of engagement/boredom/confusion/frustration)

# 4. Choose "5" → exit
```

## CSV Output

Each row: `timestamp, Engagement_level, Engagement_confidence, Boredom_level, Boredom_confidence, ...`

Example:
```
timestamp,Engagement_level,Engagement_confidence,Boredom_level,Boredom_confidence,Confusion_level,Confusion_confidence,Frustration_level,Frustration_confidence
2026-06-16T12:00:00.123456,High,0.7234,Low,0.5123,Very Low,0.6789,Very Low,0.8456
2026-06-16T12:00:00.456789,High,0.7456,Very Low,0.6234,Low,0.5432,Very Low,0.8234
```

## Troubleshooting

**"Models directory not found"**
- Ensure Google Drive is mapped to `G:\` or edit `run_camera_log.ps1` `-ModelsDir` default
- Check that the path contains `cnn_lstm_*.pt` files

**PowerShell execution policy error**
- Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Or bypass for one run: `powershell -ExecutionPolicy Bypass -File menu.ps1`

**Camera won't open**
- Try `-Camera 1` or `-Camera 2`
- Check no other app is using the webcam

**CSV file is empty**
- Normal on first run — takes 1–2 seconds to fill the 16-frame buffer before predictions start
- Run for at least 30 seconds to get meaningful data

## Tips

- **Batch mode:** Create a `.bat` file:
  ```batch
  @echo off
  cd /d "%~dp0"
  powershell -NoExit -File menu.ps1
  ```
  Then double-click it to launch.

- **Log to timestamped file:**
  ```powershell
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  .\run_camera_log.ps1 -Output "predictions_$timestamp.csv"
  ```

- **Automated workflow:** Create a script that runs demo, pauses, then auto-analyzes:
  ```powershell
  .\run_camera_log.ps1 -Output "auto_test.csv"
  Read-Host "Demo complete. Press Enter to analyze"
  .\analyze.ps1 -CsvFile "auto_test.csv"
  ```

---

**For detailed technical docs**, see `README_LOGGING.md` in the parent directory (`../`).
