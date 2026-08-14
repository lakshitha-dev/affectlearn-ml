# Camera Logging & Analysis Scripts

Two companion scripts for running the facial engagement model on live webcam with full prediction logging.

## predict_camera_log.py

Live webcam prediction with CSV logging. Runs the same as `predict_camera.py` but writes every prediction to a timestamped CSV file.

### Usage

```bash
# Basic: logs to predictions.csv
python predict_camera_log.py --models-dir "G:\My Drive\affectlearn-ml\models"

# Custom output file
python predict_camera_log.py --models-dir "G:\My Drive\affectlearn-ml\models" --output my_run.csv

# Custom camera (if default is wrong)
python predict_camera_log.py --models-dir "G:\My Drive\affectlearn-ml\models" --camera 1
```

### CSV Output Format

Each row contains:
- `timestamp` (ISO 8601)
- `Engagement_level`, `Engagement_confidence`
- `Boredom_level`, `Boredom_confidence`
- `Confusion_level`, `Confusion_confidence`
- `Frustration_level`, `Frustration_confidence`

Example:
```
timestamp,Engagement_level,Engagement_confidence,Boredom_level,Boredom_confidence,...
2026-06-16T11:50:33.123456,High,0.7234,Low,0.5123,...
2026-06-16T11:50:33.456789,High,0.7456,Very Low,0.6234,...
```

### Controls

- **Press 'q'** in the video window to stop and close the CSV file.
- The window displays the same green face box + color-coded affect panel as the original demo.
- Predictions update every ~12 frames (~half-second on 30fps camera).

---

## analyze_predictions.py

Parse and summarize a logged CSV file, showing distribution and confidence statistics.

### Usage

```bash
python analyze_predictions.py predictions.csv
```

### Example Output

```
=== Prediction Summary ===
Total predictions: 120
Affects tracked: Engagement, Boredom, Confusion, Frustration
Time range: 2026-06-16T11:50:30.123456 to 2026-06-16T11:52:15.654321

Engagement:
  Average confidence: 0.687
  Confidence range: 0.412 – 0.891
  Level distribution:
    Very Low      5  ( 4.2%)
    Low          15  (12.5%)
    High         70  (58.3%)
    Very High    30  (25.0%)

Boredom:
  Average confidence: 0.542
  Confidence range: 0.251 – 0.823
  Level distribution:
    Very Low     60  (50.0%)
    Low          40  (33.3%)
    High         15  (12.5%)
    Very High     5  ( 4.2%)

...
```

---

## Workflow Example: Test with Scripted Behaviors

1. **Run the logging demo:**
   ```bash
   python predict_camera_log.py --models-dir "G:\My Drive\affectlearn-ml\models" --output test_behaviors.csv
   ```

2. **Perform behaviors in sequence** (each ~15-30 seconds), note the time transitions:
   - **0–15s:** Attentive (eyes open, facing camera, engaged posture)
   - **15–30s:** Looking away (rotate head ~45°, look off-camera)
   - **30–45s:** Eyes closed or blink heavily
   - **45–60s:** Back to attentive

3. **Stop** by pressing 'q' in the video window.

4. **Analyze** the CSV:
   ```bash
   python analyze_predictions.py test_behaviors.csv
   ```

5. **Inspect patterns:**
   - Does `Engagement` drop during "Looking away" and "Eyes closed"?
   - Does `Confusion` or `Frustration` spike?
   - Check confidence levels — higher = more certain predictions.

**Note:** This is a **sanity check**, not validation. The model is trained on DAiSEE label distributions (which don't perfectly align with "eyes open = engaged"). See the main README for rigorous evaluation methodology.

---

## Tips

- **Best lighting:** Front-lit face (avoid backlighting or shadows).
- **Frame rate:** The rolling buffer expects ~30fps video. If your camera is slower, predictions take longer to stabilize.
- **Confidence interpretation:** 
  - >0.6 = fairly confident
  - 0.4–0.6 = uncertain, near-random
  - <0.4 = the model is guessing
- **CSV file grows:** Each prediction appends a row. A 5-minute run at 30fps with predictions every 12 frames is ~150 rows (~15KB).

---

## Troubleshooting

**Camera doesn't open:**
- Try `--camera 1` or `--camera 2`
- Check that no other app is using the webcam

**No model files found:**
- Verify the `--models-dir` path exists and contains `cnn_lstm_*.pt` files
- The default is `affectlearn-ml/models`, which only has behavioral models locally. Use the Google Drive path: `"G:\My Drive\affectlearn-ml\models"`

**CSV file empty:**
- The header row is written immediately. If the file exists but only has headers, the demo ran but didn't make any predictions (usually takes 1–2 seconds of video to fill the 16-frame buffer).

**Low confidence scores:**
- Normal. The DAiSEE 4-class engagement task is inherently difficult. ~50% accuracy is near the representational ceiling. Average confidence around 0.4–0.6 is expected.
