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
        forward_threshold: float,
        drop_threshold: float,
        debug: bool,
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.calibration_frames = max(1, calibration_frames)
        self.forward_threshold = forward_threshold
        self.drop_threshold = drop_threshold
        self.debug = debug

        self._model = None
        self._baseline_nose_forward = 0.0
        self._baseline_nose_height = 0.0
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
        self._baseline_nose_forward = 0.0
        self._baseline_nose_height = 0.0
        self._calib_frames = 0
        self._calibrated = False
        if self.enabled:
            print("[Posture] Calibration reset.")

    @staticmethod
    def _get_best_shoulder(keypoints: np.ndarray) -> np.ndarray | None:
        left = keypoints[5]
        right = keypoints[6]
        left_valid = left[0] > 0 and left[1] > 0
        right_valid = right[0] > 0 and right[1] > 0
        if left_valid and right_valid:
            return left if left[0] > right[0] else right
        if left_valid:
            return left
        if right_valid:
            return right
        return None

    @staticmethod
    def _get_best_ear(keypoints: np.ndarray) -> np.ndarray | None:
        left = keypoints[3]
        right = keypoints[4]
        left_valid = left[0] > 0 and left[1] > 0
        right_valid = right[0] > 0 and right[1] > 0
        if left_valid and right_valid:
            return left if left[0] > right[0] else right
        if left_valid:
            return left
        if right_valid:
            return right
        return None

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

        for r in results:
            if r.keypoints is None or len(r.keypoints.xy) == 0:
                continue

            try:
                keypoints = r.keypoints.xy[0].cpu().numpy() if hasattr(r.keypoints.xy[0], "cpu") else np.array(r.keypoints.xy[0])
            except Exception:
                continue

            if keypoints.shape[0] < 7:
                continue

            nose = keypoints[0]
            shoulder = self._get_best_shoulder(keypoints)
            ear = self._get_best_ear(keypoints)

            if shoulder is None or nose[0] == 0:
                cv2.putText(
                    frame,
                    "Can't detect keypoints - adjust camera",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
                continue

            if ear is not None and ear[0] > 0:
                scale = float(np.linalg.norm(ear - shoulder))
            else:
                scale = float(np.linalg.norm(nose - shoulder))
            if scale <= 1e-6:
                continue

            nose_forward = float((nose[0] - shoulder[0]) / scale)
            nose_height = float((shoulder[1] - nose[1]) / scale)

            frame[:] = r.plot()

            if not self._calibrated:
                self._baseline_nose_forward += nose_forward
                self._baseline_nose_height += nose_height
                self._calib_frames += 1

                cv2.putText(frame, "Sit straight - Calibrating...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{self._calib_frames}/{self.calibration_frames}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

                if self._calib_frames >= self.calibration_frames:
                    self._baseline_nose_forward /= float(self._calib_frames)
                    self._baseline_nose_height /= float(self._calib_frames)
                    self._calibrated = True
                    print(
                        "[Posture] Calibrated. "
                        f"forward={self._baseline_nose_forward:.3f}, "
                        f"height={self._baseline_nose_height:.3f}"
                    )

                return PostureResult(True, False, None, None)

            forward_deviation = nose_forward - self._baseline_nose_forward
            drop_deviation = self._baseline_nose_height - nose_height

            forward_bad = forward_deviation > self.forward_threshold
            neck_drop_bad = drop_deviation > self.drop_threshold

            if forward_bad or neck_drop_bad:
                reasons = []
                if forward_bad:
                    reasons.append("head forward")
                if neck_drop_bad:
                    reasons.append("neck bent")
                posture = "BAD: " + " + ".join(reasons)
                color = (0, 0, 255)
                good_posture = False
            else:
                posture = "GOOD POSTURE"
                color = (0, 255, 0)
                good_posture = True

            cv2.putText(frame, posture, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(
                frame,
                f"Fwd: {forward_deviation:+.2f}  Drop: {drop_deviation:+.2f}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            return PostureResult(True, True, good_posture, forward_deviation)

        return PostureResult(True, self._calibrated, None, None)
