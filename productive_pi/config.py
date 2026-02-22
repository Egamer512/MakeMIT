from dataclasses import dataclass
from pathlib import Path
import os


def _load_env_file() -> None:
    env_path = Path(os.getenv("APP_ENV_FILE", ".env"))
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Keep explicit shell exports higher priority than file values.
        os.environ.setdefault(key, value)


_load_env_file()


@dataclass
class AppConfig:
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    min_face_conf: float = float(os.getenv("MIN_FACE_CONF", "0.55"))

    # Focus scoring
    focus_up_per_sec: float = float(os.getenv("FOCUS_UP_PER_SEC", "1.5"))
    focus_down_per_sec: float = float(os.getenv("FOCUS_DOWN_PER_SEC", "3.0"))
    focus_phone_penalty: float = float(os.getenv("FOCUS_PHONE_PENALTY", "2.0"))

    # LCD (16x2 I2C common backpack: PCF8574)
    lcd_enabled: bool = os.getenv("LCD_ENABLED", "1") == "1"
    lcd_i2c_addr: int = int(os.getenv("LCD_I2C_ADDR", "0x27"), 16)
    lcd_cols: int = int(os.getenv("LCD_COLS", "16"))
    lcd_rows: int = int(os.getenv("LCD_ROWS", "2"))

    # Ollama local model (disabled by default for Pi bring-up)
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "0") == "1"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma:2b")
    coach_interval_sec: int = int(os.getenv("COACH_INTERVAL_SEC", "60"))

    # Optional YOLO model for phone detection
    yolo_enabled: bool = os.getenv("YOLO_ENABLED", "1") == "1"
    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt"))
    yolo_phone_conf: float = float(os.getenv("YOLO_PHONE_CONF", "0.55"))
    yolo_phone_min_area_ratio: float = float(os.getenv("YOLO_PHONE_MIN_AREA_RATIO", "0.01"))
    yolo_phone_center_x_margin: float = float(os.getenv("YOLO_PHONE_CENTER_X_MARGIN", "0.15"))
    yolo_phone_min_center_y_ratio: float = float(os.getenv("YOLO_PHONE_MIN_CENTER_Y_RATIO", "0.30"))
    yolo_phone_on_frames: int = int(os.getenv("YOLO_PHONE_ON_FRAMES", "10"))
    yolo_phone_off_frames: int = int(os.getenv("YOLO_PHONE_OFF_FRAMES", "6"))

    # Voice alert (free local TTS by default, ElevenLabs optional)
    voice_enabled: bool = os.getenv("VOICE_ENABLED", "1") == "1"
    voice_backend: str = os.getenv("VOICE_BACKEND", "local").lower()  # local | elevenlabs
    local_voice_name: str = os.getenv("LOCAL_VOICE_NAME", "Samantha")  # used by macOS `say`
    elevenlabs_enabled: bool = os.getenv("ELEVENLABS_ENABLED", "0") == "1"
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    elevenlabs_fallback_local: bool = os.getenv("ELEVENLABS_FALLBACK_LOCAL", "1") == "1"
    voice_debug: bool = os.getenv("VOICE_DEBUG", "0") == "1"
    distraction_trigger_seconds: float = float(os.getenv("DISTRACTION_TRIGGER_SECONDS", "5"))
    distraction_voice_cooldown_seconds: float = float(os.getenv("DISTRACTION_VOICE_COOLDOWN_SECONDS", "20"))
    off_task_reset_seconds: float = float(os.getenv("OFF_TASK_RESET_SECONDS", "1.5"))
    blink_eye_openness_threshold: float = float(os.getenv("BLINK_EYE_OPENNESS_THRESHOLD", "0.11"))
    gaze_off_grace_seconds: float = float(os.getenv("GAZE_OFF_GRACE_SECONDS", "0.8"))
    voice_test_on_start: bool = os.getenv("VOICE_TEST_ON_START", "0") == "1"
    first_alert_seconds: float = float(os.getenv("FIRST_ALERT_SECONDS", "10"))
    repeat_alert_seconds: float = float(os.getenv("REPEAT_ALERT_SECONDS", "30"))
    first_alert_message: str = os.getenv(
        "FIRST_ALERT_MESSAGE",
        "They there! Please return to your workspace!",
    )
    repeat_alert_message: str = os.getenv(
        "REPEAT_ALERT_MESSAGE",
        "It's been too long, please return to work!",
    )

    # Voice input gate: wait for phrase before monitoring starts.
    ready_phrase_enabled: bool = os.getenv("READY_PHRASE_ENABLED", "0") == "1"
    ready_phrase_text: str = os.getenv("READY_PHRASE_TEXT", "I'm ready")
    ready_whisper_model: str = os.getenv("READY_WHISPER_MODEL", "tiny.en")
    ready_chunk_seconds: float = float(os.getenv("READY_CHUNK_SECONDS", "2.0"))
    ready_timeout_seconds: float = float(os.getenv("READY_TIMEOUT_SECONDS", "0"))
    ready_debug: bool = os.getenv("READY_DEBUG", "0") == "1"

    # Posture detection (YOLO pose)
    posture_enabled: bool = os.getenv("POSTURE_ENABLED", "0") == "1"
    posture_model_path: str = os.getenv("POSTURE_MODEL_PATH", "yolo11n-pose.pt")
    posture_calibration_frames: int = int(os.getenv("POSTURE_CALIBRATION_FRAMES", "60"))
    posture_deviation_threshold: float = float(os.getenv("POSTURE_DEVIATION_THRESHOLD", "0.18"))
    posture_slouch_alert_seconds: float = float(os.getenv("POSTURE_SLOUCH_ALERT_SECONDS", "10"))
    posture_alert_message: str = os.getenv("POSTURE_ALERT_MESSAGE", "Hey! let's fix that posture of yours.")
    posture_recover_reset_seconds: float = float(os.getenv("POSTURE_RECOVER_RESET_SECONDS", "1.5"))
    posture_debug: bool = os.getenv("POSTURE_DEBUG", "0") == "1"

    # LED alert on breadboard (off-task indicator)
    led_enabled: bool = os.getenv("LED_ENABLED", "0") == "1"
    led_pin: int = int(os.getenv("LED_PIN", "17"))  # BCM numbering by default
    led_active_high: bool = os.getenv("LED_ACTIVE_HIGH", "1") == "1"


CONFIG = AppConfig()
