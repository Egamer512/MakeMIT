from dataclasses import dataclass
from pathlib import Path
import os


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

    # Ollama local model
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "1") == "1"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma:2b")
    coach_interval_sec: int = int(os.getenv("COACH_INTERVAL_SEC", "60"))

    # Optional YOLO model for phone detection
    yolo_enabled: bool = os.getenv("YOLO_ENABLED", "1") == "1"
    yolo_model_path: Path = Path(os.getenv("YOLO_MODEL_PATH", "models/yolov8n.pt"))


CONFIG = AppConfig()
