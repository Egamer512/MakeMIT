from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class VisionResult:
    user_in_frame: bool
    eye_openness: float
    gaze_centered: bool
    eye_yaw_deg: Optional[float]
    eye_pitch_deg: Optional[float]
    frame: np.ndarray


class VisionEngine:
    def __init__(self, min_face_conf: float = 0.55):
        # Haar cascade fallback for environments where MediaPipe FaceMesh is unavailable.
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    @staticmethod
    def _safe_ratio(n: float, d: float) -> float:
        if d <= 1e-6:
            return 0.0
        return float(n / d)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _estimate_eye_angles(self, dx: float, dy: float) -> tuple[float, float]:
        # Approximate gaze yaw/pitch from normalized pupil offsets inside eye boxes.
        max_yaw = 35.0
        max_pitch = 22.0
        yaw_deg = self._clamp(dx, -1.0, 1.0) * max_yaw
        pitch_deg = self._clamp(-dy, -1.0, 1.0) * max_pitch
        return yaw_deg, pitch_deg

    @staticmethod
    def _pupil_offset(eye_gray: np.ndarray) -> tuple[float, float]:
        if eye_gray.size == 0:
            return 0.0, 0.0
        blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)
        _, _, min_loc, _ = cv2.minMaxLoc(blur)
        h, w = eye_gray.shape[:2]
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        cx, cy = w * 0.5, h * 0.5
        dx = (float(min_loc[0]) - cx) / max(1.0, cx)
        dy = (float(min_loc[1]) - cy) / max(1.0, cy)
        return dx, dy

    def process(self, frame: np.ndarray) -> VisionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            cv2.putText(frame, "No user detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return VisionResult(False, 0.0, False, None, None, frame)

        # Choose largest face.
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        roi_gray = gray[fy : fy + fh, fx : fx + fw]
        roi_color = frame[fy : fy + fh, fx : fx + fw]

        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.15, minNeighbors=6, minSize=(20, 20), maxSize=(fw // 2, fh // 2)
        )

        # Keep most likely two eyes: upper half of face and largest areas.
        eye_candidates = []
        for (ex, ey, ew, eh) in eyes:
            if ey < int(fh * 0.65):
                eye_candidates.append((ex, ey, ew, eh))
        eye_candidates.sort(key=lambda e: e[2] * e[3], reverse=True)
        eye_candidates = eye_candidates[:2]
        eye_candidates.sort(key=lambda e: e[0])  # left-to-right

        eye_openness_vals = []
        pupil_offsets = []

        for (ex, ey, ew, eh) in eye_candidates:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_gray = roi_gray[ey : ey + eh, ex : ex + ew]
            dx, dy = self._pupil_offset(eye_gray)
            pupil_offsets.append((dx, dy))

            # Draw pupil marker relative to the detected dark point.
            px = int(ex + (dx + 1.0) * 0.5 * ew)
            py = int(ey + (dy + 1.0) * 0.5 * eh)
            cv2.circle(roi_color, (px, py), 3, (0, 0, 255), -1)

            eye_openness_vals.append(self._safe_ratio(float(eh), float(ew)))

        eye_openness = float(np.mean(eye_openness_vals)) if eye_openness_vals else 0.0

        if pupil_offsets:
            mean_dx = float(np.mean([v[0] for v in pupil_offsets]))
            mean_dy = float(np.mean([v[1] for v in pupil_offsets]))
            eye_yaw_deg, eye_pitch_deg = self._estimate_eye_angles(mean_dx, mean_dy)
            gaze_centered = abs(mean_dx) < 0.35 and abs(mean_dy) < 0.40 and eye_openness > 0.12
        else:
            eye_yaw_deg, eye_pitch_deg = None, None
            gaze_centered = False

        cv2.putText(frame, f"EyeOpen: {eye_openness:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gaze: {'CENTER' if gaze_centered else 'OFF'}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if eye_yaw_deg is not None and eye_pitch_deg is not None:
            cv2.putText(frame, f"EyeYaw: {eye_yaw_deg:+.1f} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                frame, f"EyePitch: {eye_pitch_deg:+.1f} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        else:
            cv2.putText(frame, "EyeYaw: n/a", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "EyePitch: n/a", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return VisionResult(True, eye_openness, gaze_centered, eye_yaw_deg, eye_pitch_deg, frame)


class PhoneDetector:
    def __init__(self, enabled: bool, model_path: str):
        self.enabled = enabled
        self.model = None
        if not enabled:
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
        except Exception as exc:
            self.enabled = False
            print(f"[PhoneDetector] Disabled: {exc}")

    def detect(self, frame: np.ndarray) -> bool:
        if not self.enabled or self.model is None:
            return False

        try:
            results = self.model.predict(frame, verbose=False, conf=0.35)
            if not results:
                return False
            boxes = results[0].boxes
            if boxes is None:
                return False

            names = results[0].names
            for cls_idx in boxes.cls.tolist():
                label = names.get(int(cls_idx), "") if isinstance(names, dict) else ""
                if str(label).lower() in {"cell phone", "mobile phone", "phone"}:
                    return True
        except Exception:
            return False

        return False
