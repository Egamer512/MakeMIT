# Productivity Pi (Camera + LCD + Ollama)

A Raspberry Pi app that encourages focused desk work by combining:
- **Camera vision** (user presence + eye/gaze heuristics)
- **LCD status display** (16x2 I2C)
- **Interactive local coach** using **Ollama** (for example `gemma:2b`)

## What it tracks
- `user_in_frame`: whether a face is detected.
- `eye_openness`: simple eye openness signal from eye bounding boxes.
- `gaze_centered`: rough estimate that user is looking toward the monitor/work area.
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

## Configure via file (.env)
```bash
cp .env.example .env
```
Then edit `.env` with your values (for ElevenLabs, set `ELEVENLABS_ENABLED=1` and your `ELEVENLABS_API_KEY`).

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
export VOICE_ENABLED=1
export VOICE_BACKEND=local
export LOCAL_VOICE_NAME=Samantha
export DISTRACTION_VOICE_COOLDOWN_SECONDS=20

# Optional later (disabled for now):
# export ELEVENLABS_ENABLED=1
# export ELEVENLABS_API_KEY=your_key_here
# export ELEVENLABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL
```

## Ollama setup
Install Ollama on the Pi and pull a small model:
```bash
ollama pull gemma:2b
```

The app sends a short context prompt every `COACH_INTERVAL_SEC` and prints a concise tip.

## Notes
- Vision currently uses OpenCV Haar cascades + pupil heuristics (works even when `mediapipe.solutions` is unavailable).
- Voice alert defaults to free local TTS (`say` on macOS, `espeak`/`spd-say` on Linux) with no API key needed.
- ElevenLabs is optional and only used when `ELEVENLABS_ENABLED=1`.
- Off-task voice escalation currently speaks at 10s and 30s of continuous off-task time.
- Off-task timer only resets after `OFF_TASK_RESET_SECONDS` of continuous on-task time (reduces flicker resets).
- Natural blinks are ignored using `BLINK_EYE_OPENNESS_THRESHOLD`.
- Gaze must be continuously off for `GAZE_OFF_GRACE_SECONDS` before off-task timing starts.
- ElevenLabs playback on macOS uses built-in `afplay`; no extra player install required on Mac.
- If ElevenLabs API fails, the app falls back to local TTS automatically.
- Set `VOICE_TEST_ON_START=1` in `.env` to hear a startup voice check immediately.
- Set `VOICE_DEBUG=1` to print exact voice backend/player/API diagnostics.
- Set `ELEVENLABS_FALLBACK_LOCAL=0` to disable fallback and expose pure ElevenLabs errors.
- If LCD init fails, the app continues without LCD output.
- This scaffold is designed for extension (e.g., points, streaks, daily goals, web dashboard).
