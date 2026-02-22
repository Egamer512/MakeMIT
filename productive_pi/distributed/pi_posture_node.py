import argparse
import json
import socket
import time

import cv2

from ..posture import PostureMonitor


def send_udp(sock: socket.socket, host: str, port: int, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, (host, port))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi node: posture detection only, sends events to laptop")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", required=True, help="Laptop IP address")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--calibration-frames", type=int, default=60)
    parser.add_argument("--forward-threshold", type=float, default=0.15)
    parser.add_argument("--drop-threshold", type=float, default=0.12)
    parser.add_argument("--slouch-seconds", type=float, default=10.0)
    parser.add_argument("--recover-reset-seconds", type=float, default=1.5)
    parser.add_argument("--alert-message", default="Hey! let's fix that posture of yours.")
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open posture webcam on Pi.")

    posture = PostureMonitor(
        enabled=True,
        model_path=args.model,
        calibration_frames=args.calibration_frames,
        forward_threshold=args.forward_threshold,
        drop_threshold=args.drop_threshold,
        debug=False,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    slouch_since = None
    posture_good_since = None
    slouch_alert_sent = False
    last_status_sent = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.monotonic()
            posture_res = posture.process(frame)

            slouch_for = 0.0
            if posture_res.enabled and posture_res.calibrated and posture_res.good_posture is not None:
                if not posture_res.good_posture:
                    posture_good_since = None
                    if slouch_since is None:
                        slouch_since = now
                    slouch_for = now - slouch_since
                    if slouch_for >= args.slouch_seconds and not slouch_alert_sent:
                        send_udp(
                            sock,
                            args.host,
                            args.port,
                            {
                                "type": "posture_alert",
                                "message": args.alert_message,
                                "slouch_for": slouch_for,
                            },
                        )
                        print(f"[PiPosture] Sent posture alert at {slouch_for:.1f}s")
                        slouch_alert_sent = True
                else:
                    if posture_good_since is None:
                        posture_good_since = now
                    if (now - posture_good_since) >= args.recover_reset_seconds:
                        slouch_since = None
                        slouch_alert_sent = False

            if now - last_status_sent >= 0.5:
                send_udp(
                    sock,
                    args.host,
                    args.port,
                    {
                        "type": "posture_status",
                        "enabled": posture_res.enabled,
                        "calibrated": posture_res.calibrated,
                        "good_posture": posture_res.good_posture,
                        "slouch_for": slouch_for,
                    },
                )
                last_status_sent = now

            if args.show_window:
                cv2.imshow("Pi Posture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    posture.reset_calibration()
                    slouch_since = None
                    posture_good_since = None
                    slouch_alert_sent = False

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
