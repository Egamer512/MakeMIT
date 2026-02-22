import argparse
import cv2
from flask import Flask, Response


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream laptop webcam as MJPEG over HTTP")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    app = Flask(__name__)
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam.")

    def gen_frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            ok_enc, buffer = cv2.imencode(".jpg", frame)
            if not ok_enc:
                continue
            jpg = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    @app.route("/video_feed")
    def video_feed():
        return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"Laptop webcam stream ready at http://{args.host}:{args.port}/video_feed")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
