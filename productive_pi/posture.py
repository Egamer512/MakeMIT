from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostureResult:
    enabled: bool
    calibrated: bool
    good_posture: bool | None
    head_deviation: float | None


class PostureMonitor:
    def __init__(
        self,
        enabled: bool,
        model_path: str,
        calibration_frames: int,
        deviation_threshold: float,
        debug: bool,
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.calibration_frames = max(1, calibration_frames)
        self.deviation_threshold = deviation_threshold
        self.debug = debug

        self._model = None
        self._baseline_sum = 0.0
        self._baseline_ratio = 0.0
        self._calib_frames = 0
        self._calibrated = False

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            print(f"[Posture] Enabled with model: {self.model_path}")
        except Exception as exc:
            self.enabled = False
            print(f"[Posture] Disabled due to init error: {exc}")

    def reset_calibration(self) -> None:
        self._baseline_sum = 0.0
        self._baseline_ratio = 0.0
        self._calib_frames = 0
        self._calibrated = False
        if self.enabled:
            print("[Posture] Calibration reset.")

    @staticmethod
    def _extract_head_ratio(keypoints: np.ndarray) -> float | None:
        # YOLO pose indices used here:
        # 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder
        if keypoints.shape[0] < 7:
            return None

        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        shoulder_mid = (left_shoulder + right_shoulder) * 0.5
        head_x = float(np.mean([nose[0], left_eye[0], right_eye[0], left_ear[0], right_ear[0]]))

        forward_head_distance = head_x - float(shoulder_mid[0])
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if shoulder_width <= 1e-6:
            return None

        return forward_head_distance / shoulder_width

    def process(self, frame: np.ndarray) -> PostureResult:
        if not self.enabled or self._model is None:
            return PostureResult(False, False, None, None)

        try:
            results = self._model(frame, verbose=False)
        except Exception as exc:
            if self.debug:
                print(f"[Posture][Debug] infer error: {exc}")
            return PostureResult(True, self._calibrated, None, None)

        if not results:
            return PostureResult(True, self._calibrated, None, None)

        r = results[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return PostureResult(True, self._calibrated, None, None)

        try:
            keypoints = r.keypoints.xy[0].cpu().numpy() if hasattr(r.keypoints.xy[0], "cpu") else np.array(r.keypoints.xy[0])
        except Exception:
            return PostureResult(True, self._calibrated, None, None)

        head_ratio = self._extract_head_ratio(keypoints)
        if head_ratio is None:
            return PostureResult(True, self._calibrated, None, None)

        if not self._calibrated:
            self._baseline_sum += head_ratio
            self._calib_frames += 1
            if self._calib_frames >= self.calibration_frames:
                self._baseline_ratio = self._baseline_sum / float(self._calib_frames)
                self._calibrated = True
                print(f"[Posture] Calibrated. Baseline ratio={self._baseline_ratio:.4f}")

            remaining = max(0, self.calibration_frames - self._calib_frames)
            cv2.putText(
                frame,
                f"Posture calibrating... {remaining}",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            return PostureResult(True, False, None, None)

        deviation = head_ratio - self._baseline_ratio
        bad_posture = deviation > self.deviation_threshold
        label = "BAD POSTURE" if bad_posture else "GOOD POSTURE"
        color = (0, 0, 255) if bad_posture else (0, 255, 0)

        cv2.putText(frame, f"{label}", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"PostureDev: {deviation:+.2f}", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return PostureResult(True, True, (not bad_posture), deviation)
