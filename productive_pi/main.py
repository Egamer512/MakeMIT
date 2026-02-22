import argparse
import time

import cv2

from .config import CONFIG
from .coach import CoachConfig, ProductivityCoach
from .lcd import LcdDisplay
from .state import ProductivityState
from .vision import PhoneDetector, VisionEngine


def format_lcd(state: ProductivityState) -> tuple[str, str]:
    line1 = f"{state.status_label():<9} {state.focus_score:5.1f}"
    line2 = f"U:{int(state.user_in_frame)} P:{int(state.phone_detected)} G:{int(state.gaze_centered)}"
    return line1, line2


def draw_banner(frame, user_in_frame: bool) -> None:
    h, w = frame.shape[:2]
    color = (0, 160, 0) if user_in_frame else (0, 0, 200)
    text = "USER IN FRAME" if user_in_frame else "USER NOT IN FRAME"
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def run(show_window: bool, fullscreen: bool) -> None:
    cap = cv2.VideoCapture(CONFIG.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_INDEX and camera wiring.")

    vision = VisionEngine(min_face_conf=CONFIG.min_face_conf)
    phone_detector = PhoneDetector(enabled=CONFIG.yolo_enabled, model_path=str(CONFIG.yolo_model_path))

    coach = ProductivityCoach(
        CoachConfig(
            enabled=CONFIG.ollama_enabled,
            url=CONFIG.ollama_url,
            model=CONFIG.ollama_model,
            interval_sec=CONFIG.coach_interval_sec,
        )
    )

    lcd = LcdDisplay(
        enabled=CONFIG.lcd_enabled,
        address=CONFIG.lcd_i2c_addr,
        cols=CONFIG.lcd_cols,
        rows=CONFIG.lcd_rows,
    )

    state = ProductivityState()
    last = time.monotonic()

    if show_window:
        cv2.namedWindow("Productivity Pi", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("Productivity Pi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] Frame read failed.")
                break

            now = time.monotonic()
            dt = max(0.001, now - last)
            last = now

            vis = vision.process(frame)
            phone = phone_detector.detect(vis.frame)

            state.user_in_frame = vis.user_in_frame
            state.eye_openness = vis.eye_openness
            state.gaze_centered = vis.gaze_centered
            state.phone_detected = phone
            state.update_focus(
                dt,
                focus_up=CONFIG.focus_up_per_sec,
                focus_down=CONFIG.focus_down_per_sec,
                phone_penalty=CONFIG.focus_phone_penalty,
            )

            tip = coach.maybe_generate_tip(state)
            if tip:
                print(f"[Coach] {tip}")

            lcd1, lcd2 = format_lcd(state)
            lcd.write(lcd1, lcd2)

            draw_banner(vis.frame, state.user_in_frame)
            cv2.putText(vis.frame, f"Status: {state.status_label()}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
            cv2.putText(vis.frame, f"Focus: {state.focus_score:.1f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
            if vis.eye_yaw_deg is not None and vis.eye_pitch_deg is not None:
                cv2.putText(
                    vis.frame,
                    f"Angles Yaw/Pitch: {vis.eye_yaw_deg:+.1f}/{vis.eye_pitch_deg:+.1f}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 220, 255),
                    2,
                )
            if phone:
                cv2.putText(vis.frame, "PHONE DETECTED", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if show_window:
                cv2.imshow("Productivity Pi", vis.frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        lcd.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Productivity + anti-phone Raspberry Pi app")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--fullscreen", action="store_true", help="Show preview in fullscreen for external monitor")
    args = parser.parse_args()

    run(show_window=not args.headless, fullscreen=args.fullscreen)


if __name__ == "__main__":
    main()
