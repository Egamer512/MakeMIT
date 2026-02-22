from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
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


def run_monitor_loop(state: SharedState, posture_camera_index: int, show_windows: bool) -> None:
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

    slouch_since: Optional[float] = None
    posture_good_since: Optional[float] = None
    slouch_alert_sent = False

    if show_windows:
        cv2.namedWindow("Pi Vision (Laptop Cam Stream)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Pi Posture Camera", cv2.WINDOW_NORMAL)

    while True:
        now = time.monotonic()
        eye_frame, frame_ts = state.get_frame()

        if eye_frame is None or (now - frame_ts) > 2.0:
            time.sleep(0.05)
            continue

        vis = vision.process(eye_frame)

        # Off-task (eye stream)
        blink_now = vis.user_in_frame and (vis.eye_openness > 0.0) and (vis.eye_openness < CONFIG.blink_eye_openness_threshold)
        if not vis.user_in_frame:
            gaze_off_since = None
            gaze_off_long = False
        else:
            if vis.gaze_centered or blink_now:
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
        )

        if show_windows:
            cv2.imshow("Pi Vision (Laptop Cam Stream)", vis.frame)
            cv2.imshow("Pi Posture Camera", pframe)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                posture.reset_calibration()
                slouch_since = None
                posture_good_since = None
                slouch_alert_sent = False

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
    args = parser.parse_args()

    state = SharedState()
    app = create_app(state)

    t = threading.Thread(target=run_monitor_loop, args=(state, args.posture_camera_index, args.show_windows), daemon=True)
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
