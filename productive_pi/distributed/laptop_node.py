import argparse
import json
import socket
import time

import cv2

from ..speech import VoiceAlerter, VoiceConfig
from ..vision import VisionEngine


def draw_banner(frame, user_in_frame: bool) -> None:
    h, w = frame.shape[:2]
    color = (0, 160, 0) if user_in_frame else (0, 0, 200)
    text = "USER IN FRAME" if user_in_frame else "USER NOT IN FRAME"
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop node: eye tracking + voice output + posture events receiver")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--listen-port", type=int, default=5006)
    parser.add_argument("--first-alert-seconds", type=float, default=10.0)
    parser.add_argument("--repeat-alert-seconds", type=float, default=30.0)
    parser.add_argument("--first-alert-message", default="Hey Anfal, Please stay on task, remember that your pset is due soon!")
    parser.add_argument("--repeat-alert-message", default="ANFAL! GET BACK TO WORK NOW.")
    parser.add_argument("--gaze-off-grace-seconds", type=float, default=0.8)
    parser.add_argument("--blink-threshold", type=float, default=0.11)

    # ElevenLabs / voice
    parser.add_argument("--voice-backend", default="elevenlabs")
    parser.add_argument("--elevenlabs-enabled", action="store_true")
    parser.add_argument("--elevenlabs-api-key", default="")
    parser.add_argument("--elevenlabs-voice-id", default="EXAVITQu4vr4xnSDxMaL")
    parser.add_argument("--elevenlabs-model-id", default="eleven_turbo_v2_5")
    parser.add_argument("--elevenlabs-fallback-local", action="store_true")
    parser.add_argument("--voice-debug", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam.")

    vision = VisionEngine(min_face_conf=0.55)

    voice = VoiceAlerter(
        VoiceConfig(
            enabled=True,
            backend="elevenlabs" if args.elevenlabs_enabled else args.voice_backend,
            local_voice_name="Samantha",
            api_key=args.elevenlabs_api_key,
            voice_id=args.elevenlabs_voice_id,
            model_id=args.elevenlabs_model_id,
            elevenlabs_fallback_local=args.elevenlabs_fallback_local,
            debug=args.voice_debug,
            trigger_seconds=0,
            cooldown_seconds=20,
        )
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.listen_port))
    sock.setblocking(False)

    gaze_off_since = None
    distracted_since = None
    next_alert_elapsed = args.first_alert_seconds
    first_alert_fired = False

    posture_state = "POSTURE: n/a"
    last_posture_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.monotonic()

            # receive posture events from Pi
            while True:
                try:
                    data, _ = sock.recvfrom(8192)
                except BlockingIOError:
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    continue

                t = msg.get("type")
                if t == "posture_alert":
                    text = msg.get("message", "Hey! let's fix that posture of yours.")
                    if voice.maybe_speak(text, force=True):
                        print(f"[Laptop] Spoke posture alert: {text}")
                elif t == "posture_status":
                    last_posture_ts = now
                    calibrated = msg.get("calibrated")
                    good = msg.get("good_posture")
                    if not calibrated:
                        posture_state = "POSTURE: calibrating"
                    elif good is True:
                        posture_state = "POSTURE: good"
                    elif good is False:
                        posture_state = f"POSTURE: bad ({msg.get('slouch_for', 0):.1f}s)"

            vis = vision.process(frame)

            blink_now = vis.user_in_frame and (vis.eye_openness > 0.0) and (vis.eye_openness < args.blink_threshold)
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
                    gaze_off_long = (now - gaze_off_since) >= args.gaze_off_grace_seconds

            off_task = (not vis.user_in_frame) or gaze_off_long
            if off_task:
                if distracted_since is None:
                    distracted_since = now
                    next_alert_elapsed = args.first_alert_seconds
                    first_alert_fired = False
                distracted_for = now - distracted_since

                while distracted_for >= next_alert_elapsed:
                    msg = args.first_alert_message if not first_alert_fired else args.repeat_alert_message
                    if voice.maybe_speak(msg, force=True):
                        print(f"[Laptop] Spoke focus alert at {next_alert_elapsed:.1f}s")
                        first_alert_fired = True
                        next_alert_elapsed += args.repeat_alert_seconds
                    else:
                        break
            else:
                distracted_since = None
                distracted_for = 0.0
                first_alert_fired = False
                next_alert_elapsed = args.first_alert_seconds

            draw_banner(vis.frame, vis.user_in_frame)
            cv2.putText(vis.frame, f"Gaze: {'CENTER' if vis.gaze_centered else 'OFF'}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
            if vis.eye_yaw_deg is not None and vis.eye_pitch_deg is not None:
                cv2.putText(
                    vis.frame,
                    f"Eye Yaw/Pitch: {vis.eye_yaw_deg:+.1f}/{vis.eye_pitch_deg:+.1f}",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 220, 255),
                    2,
                )
            if off_task:
                cv2.putText(vis.frame, f"Off-task: {distracted_for:.1f}s", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)

            if now - last_posture_ts > 2.0:
                posture_state = "POSTURE: stale/no signal"
            cv2.putText(vis.frame, posture_state, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.imshow("Laptop Eye + Audio Node", vis.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
