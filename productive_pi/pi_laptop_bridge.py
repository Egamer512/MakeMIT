from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request

from .config import CONFIG
from .posture import PostureMonitor
from .vision import VisionEngine


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Pi Bridge Client</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .ok { color: #0a8a00; }
    .warn { color: #b36b00; }
    .err { color: #c60000; }
    video { width: 320px; border: 1px solid #ccc; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>Productivity Pi Bridge</h2>
  <p id=\"status\" class=\"warn\">Starting webcam...</p>
  <video id=\"v\" autoplay muted playsinline></video>
  <script>
    const statusEl = document.getElementById('status');
    const video = document.getElementById('v');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    function setStatus(msg, cls='warn') {
      statusEl.textContent = msg;
      statusEl.className = cls;
    }

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        await video.play();
        setStatus('Connected. Webcam frames are streaming to Pi.', 'ok');

        canvas.width = 640;
        canvas.height = 480;

        setInterval(async () => {
          try {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.7));
            if (!blob) return;
            await fetch('/api/frame', {
              method: 'POST',
              headers: { 'Content-Type': 'image/jpeg' },
              body: blob,
            });
          } catch (_) {}
        }, 120);

        setInterval(async () => {
          try {
            const r = await fetch('/api/command');
            const cmd = await r.json();
            if (!cmd || !cmd.type) return;

            if (cmd.type === 'play_audio' && cmd.id) {
              const a = new Audio(`/api/audio/${cmd.id}.mp3`);
              await a.play();
            } else if (cmd.type === 'speak_text' && cmd.text) {
              const u = new SpeechSynthesisUtterance(cmd.text);
              speechSynthesis.speak(u);
            }
          } catch (_) {}
        }, 350);
      } catch (err) {
        setStatus('Camera permission failed: ' + err, 'err');
      }
    }

    start();
  </script>
</body>
</html>
"""


@dataclass
class Command:
    type: str
    payload: dict


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_ts: float = 0.0
        self.commands: deque[Command] = deque(maxlen=128)
        self.audio: dict[str, bytes] = {}
        self.metrics: dict = {
            "user_in_frame": None,
            "gaze_centered": None,
            "off_task": None,
            "off_task_seconds": 0.0,
            "posture_enabled": None,
            "posture_calibrated": None,
            "posture_good": None,
            "posture_slouch_seconds": 0.0,
            "last_alert": "",
            "updated_at": 0.0,
        }

    def set_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.latest_frame = frame
            self.latest_frame_ts = time.monotonic()

    def get_frame(self) -> tuple[Optional[np.ndarray], float]:
        with self.lock:
            if self.latest_frame is None:
                return None, self.latest_frame_ts
            return self.latest_frame.copy(), self.latest_frame_ts

    def push_command(self, cmd: Command) -> None:
        with self.lock:
            self.commands.append(cmd)

    def pop_command(self) -> Optional[Command]:
        with self.lock:
            if not self.commands:
                return None
            return self.commands.popleft()

    def put_audio(self, audio_id: str, data: bytes) -> None:
        with self.lock:
            self.audio[audio_id] = data
            # Keep memory bounded.
            if len(self.audio) > 32:
                oldest = list(self.audio.keys())[:8]
                for k in oldest:
                    self.audio.pop(k, None)

    def get_audio(self, audio_id: str) -> Optional[bytes]:
        with self.lock:
            return self.audio.get(audio_id)

    def set_metrics(self, **kwargs) -> None:
        with self.lock:
            self.metrics.update(kwargs)
            self.metrics["updated_at"] = time.monotonic()

    def get_metrics(self) -> dict:
        with self.lock:
            return dict(self.metrics)


def generate_elevenlabs_audio(text: str) -> Optional[bytes]:
    if not CONFIG.elevenlabs_enabled or not CONFIG.elevenlabs_api_key:
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{CONFIG.elevenlabs_voice_id}"
    headers = {
        "xi-api-key": CONFIG.elevenlabs_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": CONFIG.elevenlabs_model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=(5, 20))
        if resp.status_code >= 400:
            body = (resp.text or "")[:240].replace("\n", " ")
            print(f"[Bridge] ElevenLabs HTTP {resp.status_code}: {body}")
            return None
        return resp.content if resp.content else None
    except Exception as exc:
        print(f"[Bridge] ElevenLabs request failed: {exc}")
        return None


def enqueue_speak(state: SharedState, text: str) -> None:
    audio = generate_elevenlabs_audio(text)
    if audio:
        aid = uuid.uuid4().hex
        state.put_audio(aid, audio)
        state.push_command(Command("play_audio", {"id": aid, "text": text}))
    else:
        # Browser fallback if ElevenLabs unavailable.
        state.push_command(Command("speak_text", {"text": text}))


def _load_eye_calibration(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == 2 and "derived" in data:
            return data
        # Backward compatibility with older single-baseline calibration.
        yaw = float(data.get("baseline_yaw"))
        pitch = float(data.get("baseline_pitch"))
        return {
            "version": 1,
            "baseline_yaw": yaw,
            "baseline_pitch": pitch,
        }
    except Exception:
        return None


def _save_eye_calibration(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[Bridge] Could not save eye calibration: {exc}")


def run_monitor_loop(
    state: SharedState,
    posture_camera_index: int,
    show_windows: bool,
    eye_calib_file: Path,
    reuse_eye_calibration: bool,
    save_eye_calibration: bool,
) -> None:
    vision = VisionEngine(min_face_conf=CONFIG.min_face_conf)
    posture = PostureMonitor(
        enabled=CONFIG.posture_enabled,
        model_path=CONFIG.posture_model_path,
        calibration_frames=CONFIG.posture_calibration_frames,
        forward_threshold=CONFIG.posture_forward_threshold,
        drop_threshold=CONFIG.posture_drop_threshold,
        debug=CONFIG.posture_debug,
    )

    posture_cap = cv2.VideoCapture(posture_camera_index)
    if not posture_cap.isOpened():
        print("[Bridge] Warning: posture camera unavailable; posture disabled.")
        posture = PostureMonitor(False, CONFIG.posture_model_path, 1, 0.15, 0.12, False)

    gaze_off_since: Optional[float] = None
    distracted_since: Optional[float] = None
    on_task_since: Optional[float] = None
    next_alert_elapsed = CONFIG.first_alert_seconds
    first_alert_fired = False

    # Guided eye calibration profile.
    eye_calibrated = False
    eye_profile: Optional[dict] = None
    calib_targets = ["center", "left", "right", "up", "down", "away"]
    calib_target_index = 0
    calib_collecting = False
    calib_samples: list[tuple[float, float]] = []
    calib_capture_count = 18
    collected_points: dict[str, tuple[float, float]] = {}

    slouch_since: Optional[float] = None
    posture_good_since: Optional[float] = None
    slouch_alert_sent = False

    if show_windows:
        cv2.namedWindow("Pi Vision (Laptop Cam Stream)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Pi Posture Camera", cv2.WINDOW_NORMAL)

    if reuse_eye_calibration:
        loaded = _load_eye_calibration(eye_calib_file)
        if loaded is not None:
            if loaded.get("version") == 2:
                eye_profile = loaded
                eye_calibrated = True
                calib_target_index = len(calib_targets)
                print(f"[Bridge] Loaded guided eye calibration from {eye_calib_file}")
            elif loaded.get("version") == 1:
                # Legacy baseline fallback.
                yaw = float(loaded["baseline_yaw"])
                pitch = float(loaded["baseline_pitch"])
                eye_profile = {
                    "version": 1,
                    "baseline_yaw": yaw,
                    "baseline_pitch": pitch,
                    "yaw_threshold": 10.0,
                    "pitch_threshold": 8.0,
                }
                eye_calibrated = True
                calib_target_index = len(calib_targets)
                print(
                    f"[Bridge] Loaded legacy eye calibration from {eye_calib_file}: "
                    f"yaw/pitch={yaw:+.1f}/{pitch:+.1f}"
                )

    while True:
        now = time.monotonic()
        eye_frame, frame_ts = state.get_frame()

        if eye_frame is None or (now - frame_ts) > 2.0:
            time.sleep(0.05)
            continue

        vis = vision.process(eye_frame)
        # Guided calibration collection after pressing 'n'.
        if (
            calib_collecting
            and vis.user_in_frame
            and vis.eye_yaw_deg is not None
            and vis.eye_pitch_deg is not None
        ):
            calib_samples.append((vis.eye_yaw_deg, vis.eye_pitch_deg))
            if len(calib_samples) >= calib_capture_count:
                avg_yaw = float(np.mean([v[0] for v in calib_samples]))
                avg_pitch = float(np.mean([v[1] for v in calib_samples]))
                target_name = calib_targets[calib_target_index]
                collected_points[target_name] = (avg_yaw, avg_pitch)
                print(
                    f"[Bridge] Captured {target_name}: "
                    f"{avg_yaw:+.1f}/{avg_pitch:+.1f}"
                )
                calib_samples = []
                calib_collecting = False
                calib_target_index += 1

                if calib_target_index >= len(calib_targets):
                    center = collected_points["center"]
                    on_points = [collected_points[t] for t in ["center", "left", "right", "up", "down"]]
                    away = collected_points["away"]

                    yaw_vals = [p[0] for p in on_points]
                    pitch_vals = [p[1] for p in on_points]
                    yaw_margin = 2.5
                    pitch_margin = 2.5
                    yaw_min = min(yaw_vals) - yaw_margin
                    yaw_max = max(yaw_vals) + yaw_margin
                    pitch_min = min(pitch_vals) - pitch_margin
                    pitch_max = max(pitch_vals) + pitch_margin

                    on_radius = max(
                        np.hypot(p[0] - center[0], p[1] - center[1]) for p in on_points
                    ) + 1.5
                    away_radius = max(4.0, np.hypot(away[0] - center[0], away[1] - center[1]) * 0.45)

                    eye_profile = {
                        "version": 2,
                        "points": {k: [v[0], v[1]] for k, v in collected_points.items()},
                        "derived": {
                            "center": [center[0], center[1]],
                            "yaw_min": yaw_min,
                            "yaw_max": yaw_max,
                            "pitch_min": pitch_min,
                            "pitch_max": pitch_max,
                            "on_radius": on_radius,
                            "away_radius": away_radius,
                        },
                    }
                    eye_calibrated = True
                    print("[Bridge] Guided eye calibration complete.")
                    if save_eye_calibration:
                        _save_eye_calibration(eye_calib_file, eye_profile)
                        print(f"[Bridge] Saved eye calibration to {eye_calib_file}")

        # Off-task (eye stream)
        blink_now = vis.user_in_frame and (vis.eye_openness > 0.0) and (vis.eye_openness < CONFIG.blink_eye_openness_threshold)
        yaw_dev = 0.0
        pitch_dev = 0.0
        eye_away = False
        if vis.eye_yaw_deg is not None and vis.eye_pitch_deg is not None:
            if eye_calibrated and eye_profile is not None:
                if eye_profile.get("version") == 2:
                    d = eye_profile["derived"]
                    center = d["center"]
                    dist_center = float(np.hypot(vis.eye_yaw_deg - center[0], vis.eye_pitch_deg - center[1]))
                    yaw_dev = abs(vis.eye_yaw_deg - center[0])
                    pitch_dev = abs(vis.eye_pitch_deg - center[1])
                    within_box = (
                        d["yaw_min"] <= vis.eye_yaw_deg <= d["yaw_max"]
                        and d["pitch_min"] <= vis.eye_pitch_deg <= d["pitch_max"]
                    )
                    eye_away = (not within_box) or (dist_center > d["on_radius"])
                else:
                    base_yaw = float(eye_profile["baseline_yaw"])
                    base_pitch = float(eye_profile["baseline_pitch"])
                    yaw_thr = float(eye_profile.get("yaw_threshold", 10.0))
                    pitch_thr = float(eye_profile.get("pitch_threshold", 8.0))
                    yaw_dev = abs(vis.eye_yaw_deg - base_yaw)
                    pitch_dev = abs(vis.eye_pitch_deg - base_pitch)
                    eye_away = (yaw_dev > yaw_thr) or (pitch_dev > pitch_thr)
            else:
                # Pre-calibration fallback.
                eye_away = not vis.gaze_centered

        if not vis.user_in_frame:
            gaze_off_since = None
            gaze_off_long = False
        else:
            if (not eye_away) or blink_now:
                gaze_off_since = None
                gaze_off_long = False
            else:
                if gaze_off_since is None:
                    gaze_off_since = now
                gaze_off_long = (now - gaze_off_since) >= CONFIG.gaze_off_grace_seconds

        off_task = (not vis.user_in_frame) or gaze_off_long
        if off_task:
            on_task_since = None
            if distracted_since is None:
                distracted_since = now
                next_alert_elapsed = CONFIG.first_alert_seconds
                first_alert_fired = False

            distracted_for = now - distracted_since
            while distracted_for >= next_alert_elapsed:
                msg = CONFIG.first_alert_message if not first_alert_fired else CONFIG.repeat_alert_message
                enqueue_speak(state, msg)
                print(f"[Bridge] Off-task alert at {next_alert_elapsed:.1f}s")
                state.set_metrics(last_alert=f"off_task:{next_alert_elapsed:.1f}s")
                first_alert_fired = True
                next_alert_elapsed += CONFIG.repeat_alert_seconds
        else:
            if on_task_since is None:
                on_task_since = now
            if (now - on_task_since) >= CONFIG.off_task_reset_seconds:
                distracted_since = None
                first_alert_fired = False
                next_alert_elapsed = CONFIG.first_alert_seconds
            distracted_for = 0.0

        # Posture (pi external cam)
        if posture_cap.isOpened() and posture.enabled:
            ok_posture, pframe = posture_cap.read()
            if ok_posture:
                pres = posture.process(pframe)
                slouch_for = 0.0
                if pres.enabled and pres.calibrated and pres.good_posture is not None:
                    if not pres.good_posture:
                        posture_good_since = None
                        if slouch_since is None:
                            slouch_since = now
                        slouch_for = now - slouch_since
                        if slouch_for >= CONFIG.posture_slouch_alert_seconds and not slouch_alert_sent:
                            enqueue_speak(state, CONFIG.posture_alert_message)
                            print(f"[Bridge] Posture alert at {slouch_for:.1f}s")
                            state.set_metrics(last_alert=f"posture:{slouch_for:.1f}s")
                            slouch_alert_sent = True
                    else:
                        if posture_good_since is None:
                            posture_good_since = now
                        if (now - posture_good_since) >= CONFIG.posture_recover_reset_seconds:
                            slouch_since = None
                            slouch_alert_sent = False
            else:
                pres = None
                slouch_for = 0.0
        else:
            pres = None
            slouch_for = 0.0
            pframe = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                pframe,
                "Posture camera unavailable",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        state.set_metrics(
            user_in_frame=bool(vis.user_in_frame),
            gaze_centered=bool(vis.gaze_centered),
            off_task=bool(off_task),
            off_task_seconds=float(distracted_for),
            posture_enabled=bool(posture.enabled and posture_cap.isOpened()),
            posture_calibrated=(None if pres is None else bool(pres.calibrated)),
            posture_good=(None if pres is None else pres.good_posture),
            posture_slouch_seconds=float(slouch_for),
            eye_calibrated=bool(eye_calibrated),
            eye_calibration_progress=int(calib_target_index),
            eye_calibration_target=int(len(calib_targets)),
            eye_yaw_deviation=float(yaw_dev),
            eye_pitch_deviation=float(pitch_dev),
        )

        if show_windows:
            if not eye_calibrated:
                if calib_target_index < len(calib_targets):
                    target = calib_targets[calib_target_index].upper()
                    calib_line = f"CALIBRATE: LOOK {target}, press N to capture ({calib_target_index+1}/{len(calib_targets)})"
                else:
                    calib_line = "CALIBRATION: processing..."
                cv2.putText(
                    vis.frame,
                    calib_line,
                    (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                if calib_collecting:
                    cv2.putText(
                        vis.frame,
                        f"Capturing... {len(calib_samples)}/{calib_capture_count}",
                        (20, 455),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

            h, w = vis.frame.shape[:2]
            status_text = "ON TASK" if not off_task else "OFF TASK"
            status_color = (0, 170, 0) if not off_task else (0, 0, 200)
            cv2.rectangle(vis.frame, (0, 0), (w, 52), status_color, -1)
            cv2.putText(
                vis.frame,
                status_text,
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            eye_dbg = (
                f"EyeCal:{'Y' if eye_calibrated else 'N'} "
                f"Dev(Y/P):{yaw_dev:.1f}/{pitch_dev:.1f} "
                f"Away:{'Y' if eye_away else 'N'}"
            )
            cv2.putText(
                vis.frame,
                eye_dbg,
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Pi Vision (Laptop Cam Stream)", vis.frame)
            cv2.imshow("Pi Posture Camera", pframe)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                posture.reset_calibration()
                slouch_since = None
                posture_good_since = None
                slouch_alert_sent = False
            if key == ord("c"):
                eye_calibrated = False
                eye_profile = None
                calib_target_index = 0
                calib_collecting = False
                calib_samples = []
                collected_points = {}
                print("[Bridge] Eye calibration reset. Use guided calibration with N.")
            if key == ord("n"):
                if not eye_calibrated and (vis.eye_yaw_deg is not None) and (vis.eye_pitch_deg is not None):
                    calib_collecting = True
                    calib_samples = []

        time.sleep(0.03)


def create_app(state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @app.post("/api/frame")
    def frame():
        raw = request.get_data(cache=False)
        if not raw:
            return jsonify({"ok": False, "error": "empty frame"}), 400
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "error": "decode failed"}), 400
        state.set_frame(img)
        return jsonify({"ok": True})

    @app.get("/api/command")
    def command():
        cmd = state.pop_command()
        if cmd is None:
            return jsonify({})
        out = {"type": cmd.type}
        out.update(cmd.payload)
        return jsonify(out)

    @app.get("/api/audio/<audio_id>.mp3")
    def audio(audio_id: str):
        data = state.get_audio(audio_id)
        if data is None:
            return jsonify({"error": "not found"}), 404
        return Response(data, mimetype="audio/mpeg")

    @app.get("/api/health")
    def health():
        _, ts = state.get_frame()
        return jsonify({"ok": True, "last_frame_age_sec": time.monotonic() - ts if ts else None})

    @app.get("/api/state")
    def api_state():
        _, ts = state.get_frame()
        out = state.get_metrics()
        out["last_frame_age_sec"] = (time.monotonic() - ts) if ts else None
        return jsonify(out)

    @app.post("/api/test_speak")
    def test_speak():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "Bridge test audio")).strip()
        if not text:
            return jsonify({"ok": False, "error": "empty text"}), 400
        enqueue_speak(state, text)
        state.set_metrics(last_alert="manual_test")
        return jsonify({"ok": True, "queued_text": text})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi-only bridge: laptop browser provides eye camera + speakers")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--posture-camera-index", type=int, default=0)
    parser.add_argument("--https", action="store_true", help="Serve over HTTPS (recommended for browser camera access)")
    parser.add_argument("--show-windows", action="store_true", help="Show eye+posture previews on Pi display")
    parser.add_argument("--eye-calib-file", default=str(Path.home() / ".productive_pi" / "eye_calibration.json"))
    parser.add_argument("--no-reuse-eye-calibration", action="store_true")
    parser.add_argument("--no-save-eye-calibration", action="store_true")
    args = parser.parse_args()

    state = SharedState()
    app = create_app(state)

    t = threading.Thread(
        target=run_monitor_loop,
        args=(
            state,
            args.posture_camera_index,
            args.show_windows,
            Path(args.eye_calib_file),
            not args.no_reuse_eye_calibration,
            not args.no_save_eye_calibration,
        ),
        daemon=True,
    )
    t.start()

    scheme = "https" if args.https else "http"
    print("[Bridge] Open this URL on laptop and allow camera:")
    print(f"[Bridge] {scheme}://<PI_IP>:{args.port}/")
    if args.https:
        app.run(host=args.host, port=args.port, threaded=True, ssl_context="adhoc")
    else:
        app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
