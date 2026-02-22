from dataclasses import dataclass
import time
import requests

from .state import ProductivityState


@dataclass
class CoachConfig:
    enabled: bool
    url: str
    model: str
    interval_sec: int


class ProductivityCoach:
    def __init__(self, cfg: CoachConfig):
        self.cfg = cfg
        self._last_sent = 0.0

    def maybe_generate_tip(self, state: ProductivityState) -> str | None:
        if not self.cfg.enabled:
            return None

        now = time.monotonic()
        if now - self._last_sent < self.cfg.interval_sec:
            return None

        prompt = (
            "You are a concise productivity coach for a student at a desk. "
            "Give one short actionable instruction (max 18 words). "
            f"State: user_in_frame={state.user_in_frame}, "
            f"phone_detected={state.phone_detected}, gaze_centered={state.gaze_centered}, "
            f"focus_score={state.focus_score:.1f}."
        )

        payload = {"model": self.cfg.model, "prompt": prompt, "stream": False}

        try:
            resp = requests.post(self.cfg.url, json=payload, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            self._last_sent = now
            tip = (data.get("response") or "").strip()
            return tip[:120] if tip else None
        except Exception as exc:
            print(f"[Coach] Ollama unavailable: {exc}")
            self._last_sent = now
            return None
