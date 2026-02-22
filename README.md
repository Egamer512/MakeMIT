# Productivity Pi

Camera-based focus monitor for Raspberry Pi 5 with:
- live camera overlay (`in frame`, gaze, eye angles)
- off-task timer + voice escalation (10s and 30s)
- optional 16x2 I2C LCD status
- optional GPIO LED alert on breadboard when off-task

## Current logic
Off-task is:
- user out of frame, or
- user in frame but gaze off for longer than grace window

Blinks are ignored so natural blinking does not increment off-task timer.

## Raspberry Pi 5 first-time setup
Run these on the Pi:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libatlas-base-dev i2c-tools
```

Enable interfaces:
```bash
sudo raspi-config
```
Then:
- `Interface Options` -> `I2C` -> `Enable`
- `Interface Options` -> `Camera` -> `Enable` (if using Pi Camera module)

Reboot:
```bash
sudo reboot
```

## Project setup on Pi
```bash
cd /path/to/Hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure with `.env`
Create or edit `/Users/anfalhosen/Desktop/Hackathon/.env`.

Recommended Pi baseline:
```bash
# Camera
CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480

# Ollama disabled for now
OLLAMA_ENABLED=0

# Voice (ElevenLabs)
VOICE_ENABLED=1
VOICE_BACKEND=elevenlabs
ELEVENLABS_ENABLED=1
ELEVENLABS_API_KEY=your_real_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_turbo_v2_5
ELEVENLABS_FALLBACK_LOCAL=0
VOICE_TEST_ON_START=1
VOICE_DEBUG=1

# Focus timing
OFF_TASK_RESET_SECONDS=1.5
BLINK_EYE_OPENNESS_THRESHOLD=0.11
GAZE_OFF_GRACE_SECONDS=0.8

# LCD (set 0 if not connected)
LCD_ENABLED=1
LCD_I2C_ADDR=0x27
LCD_COLS=16
LCD_ROWS=2

# Breadboard LED (set 1 when wired)
LED_ENABLED=1
LED_PIN=17
LED_ACTIVE_HIGH=1
```

## LED wiring (breadboard)
Use BCM pin numbering.

Example for `LED_PIN=17`:
- Pi `GPIO17` (physical pin 11) -> 220 Ohm resistor -> LED anode (+)
- LED cathode (-) -> GND (for example physical pin 6)

If your LED is inverted due to wiring/transistor stage, set:
```bash
LED_ACTIVE_HIGH=0
```

## Run
With display preview:
```bash
python3 -m productive_pi.main --fullscreen
```

Headless:
```bash
python3 -m productive_pi.main --headless
```

## Useful checks on Pi
Check I2C device (LCD backpack):
```bash
i2cdetect -y 1
```

Quick camera check:
```bash
python3 - <<'PY'
import cv2
cap = cv2.VideoCapture(0)
print('camera open:', cap.isOpened())
cap.release()
PY
```

## Notes
- On Mac, LCD and GPIO errors are expected and safe.
- ElevenLabs must have a valid API key; a `401 invalid_api_key` means key issue.
- App reads `.env` automatically; shell exports can override `.env` values.
