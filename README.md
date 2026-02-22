# Productivity Pi (Camera + LCD + Ollama)

A Raspberry Pi app that encourages focused desk work by combining:
- **Camera vision** (user presence + eye/gaze heuristics)
- **Phone distraction detection** (optional YOLO model)
- **LCD status display** (16x2 I2C)
- **Interactive local coach** using **Ollama** (for example `gemma:2b`)

## What it tracks
- `user_in_frame`: whether a face is detected.
- `eye_openness`: simple eye openness signal from eye bounding boxes.
- `gaze_centered`: rough estimate that user is looking toward the monitor/work area.
- `phone_detected`: YOLO class-based phone detection (if enabled and model available).
- `focus_score`: 0-100 score that rises when focused and drops when distracted.

## Hardware
- Raspberry Pi 4/5
- Pi camera (or USB webcam)
- 16x2 I2C LCD (common PCF8574 backpack)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
python3 -m productive_pi.main
```

Fullscreen on an external monitor:
```bash
python3 -m productive_pi.main --fullscreen
```

Headless mode (no preview window):
```bash
python3 -m productive_pi.main --headless
```

While running the preview, the video overlay shows:
- A top banner: `USER IN FRAME` or `USER NOT IN FRAME`
- Eye metrics including estimated `EyeYaw` and `EyePitch` angles (degrees)
- Focus/status and phone detection alerts

## Environment config
Defaults are in `/Users/anfalhosen/Desktop/Hackathon/productive_pi/config.py`, and each setting can be overridden with environment variables.

Common examples:
```bash
export CAMERA_INDEX=0
export LCD_ENABLED=1
export LCD_I2C_ADDR=0x27
export OLLAMA_ENABLED=1
export OLLAMA_MODEL=gemma:2b
export OLLAMA_URL=http://127.0.0.1:11434/api/generate
export YOLO_ENABLED=1
export YOLO_MODEL_PATH=models/yolov8n.pt
```

## Ollama setup
Install Ollama on the Pi and pull a small model:
```bash
ollama pull gemma:2b
```

The app sends a short context prompt every `COACH_INTERVAL_SEC` and prints a concise tip.

## Notes
- Vision currently uses OpenCV Haar cascades + pupil heuristics (works even when `mediapipe.solutions` is unavailable).
- If `ultralytics` or YOLO weights are missing, phone detection is automatically disabled.
- If LCD init fails, the app continues without LCD output.
- This scaffold is designed for extension (e.g., points, streaks, daily goals, web dashboard).
