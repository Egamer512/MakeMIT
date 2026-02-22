from dataclasses import dataclass
from time import monotonic


@dataclass
class ProductivityState:
    user_in_frame: bool = False
    phone_detected: bool = False
    eye_openness: float = 0.0
    gaze_centered: bool = False
    focus_score: float = 50.0
    last_update: float = monotonic()

    def update_focus(self, dt: float, focus_up: float, focus_down: float, phone_penalty: float) -> None:
        productive = self.user_in_frame and (not self.phone_detected) and self.gaze_centered
        if productive:
            self.focus_score += focus_up * dt
        else:
            self.focus_score -= focus_down * dt

        if self.phone_detected:
            self.focus_score -= phone_penalty * dt

        self.focus_score = max(0.0, min(100.0, self.focus_score))

    def status_label(self) -> str:
        if not self.user_in_frame:
            return "AWAY"
        if self.phone_detected:
            return "PHONE"
        if self.gaze_centered:
            return "FOCUS"
        return "DISTRACTED"
