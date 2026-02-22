import argparse
import time
from typing import Optional

import cv2

from .config import CONFIG
from .coach import CoachConfig, ProductivityCoach
from .lcd import LcdDisplay
from .posture import PostureMonitor
from .ready_listener import ReadyPhraseListener
from .speech import VoiceAlerter, VoiceConfig
from .state import ProductivityState
from .vision import VisionEngine


def format_lcd(state: ProductivityState) -> tuple[str, str]:
    line1 = f"{state.status_label():<9} {state.focus_score:5.1f}"
    line2 = f"U:{int(state.user_in_frame)} G:{int(state.gaze_centered)}"
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
    posture = PostureMonitor(
        enabled=CONFIG.posture_enabled,
        model_path=CONFIG.posture_model_path,
        calibration_frames=CONFIG.posture_calibration_frames,
        deviation_threshold=CONFIG.posture_deviation_threshold,
        debug=CONFIG.posture_debug,
    )

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
    # LED support is currently disabled.
    # led = FocusLed(
    #     enabled=CONFIG.led_enabled,
    #     pin=CONFIG.led_pin,
    #     active_high=CONFIG.led_active_high,
    # )
    voice = VoiceAlerter(
        VoiceConfig(
            enabled=CONFIG.voice_enabled,
            backend="elevenlabs" if CONFIG.elevenlabs_enabled else CONFIG.voice_backend,
            local_voice_name=CONFIG.local_voice_name,
            api_key=CONFIG.elevenlabs_api_key,
            voice_id=CONFIG.elevenlabs_voice_id,
            model_id=CONFIG.elevenlabs_model_id,
            elevenlabs_fallback_local=CONFIG.elevenlabs_fallback_local,
            debug=CONFIG.voice_debug,
            trigger_seconds=CONFIG.distraction_trigger_seconds,
            cooldown_seconds=CONFIG.distraction_voice_cooldown_seconds,
        )
    )
    if CONFIG.voice_test_on_start:
        voice.maybe_speak("Voice system ready.", force=True)
    ready_listener = ReadyPhraseListener(
        enabled=CONFIG.ready_phrase_enabled,
        phrase=CONFIG.ready_phrase_text,
        model_name=CONFIG.ready_whisper_model,
        chunk_seconds=CONFIG.ready_chunk_seconds,
        timeout_seconds=CONFIG.ready_timeout_seconds,
        debug=CONFIG.ready_debug,
    )
    ready_listener.wait_for_phrase()

    state = ProductivityState()
    last = time.monotonic()
    distracted_since: Optional[float] = None
    on_task_since: Optional[float] = None
    gaze_off_since: Optional[float] = None
    next_alert_elapsed = CONFIG.first_alert_seconds
    first_alert_fired = False
    slouch_since: Optional[float] = None
    posture_good_since: Optional[float] = None
    slouch_alert_sent = False

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
            posture_res = posture.process(vis.frame)

            state.user_in_frame = vis.user_in_frame
            state.eye_openness = vis.eye_openness
            state.gaze_centered = vis.gaze_centered
            state.phone_detected = False
            state.update_focus(
                dt,
                focus_up=CONFIG.focus_up_per_sec,
                focus_down=CONFIG.focus_down_per_sec,
                phone_penalty=CONFIG.focus_phone_penalty,
            )

            blink_now = state.user_in_frame and (state.eye_openness > 0.0) and (
                state.eye_openness < CONFIG.blink_eye_openness_threshold
            )

            if not state.user_in_frame:
                gaze_off_since = None
                gaze_off_long = False
            else:
                if state.gaze_centered or blink_now:
                    gaze_off_since = None
                    gaze_off_long = False
                else:
                    if gaze_off_since is None:
                        gaze_off_since = now
                    gaze_off_long = (now - gaze_off_since) >= CONFIG.gaze_off_grace_seconds

            off_task = (not state.user_in_frame) or gaze_off_long
            if off_task:
                on_task_since = None
                if distracted_since is None:
                    distracted_since = now
                    next_alert_elapsed = CONFIG.first_alert_seconds
                    first_alert_fired = False
                distracted_for = now - distracted_since
                while distracted_for >= next_alert_elapsed:
                    if not first_alert_fired:
                        msg = CONFIG.first_alert_message
                        first_alert_fired = True
                    else:
                        msg = CONFIG.repeat_alert_message
                    voice.maybe_speak(msg, force=True)
                    print(f"[Voice] Triggered off-task alert at {next_alert_elapsed:.1f}s.")
                    next_alert_elapsed += CONFIG.repeat_alert_seconds
            else:
                if on_task_since is None:
                    on_task_since = now
                on_task_for = now - on_task_since
                if on_task_for >= CONFIG.off_task_reset_seconds:
                    distracted_since = None
                    next_alert_elapsed = CONFIG.first_alert_seconds
                    first_alert_fired = False
                distracted_for = 0.0

            if posture_res.enabled and posture_res.calibrated and posture_res.good_posture is not None:
                if not posture_res.good_posture:
                    posture_good_since = None
                    if slouch_since is None:
                        slouch_since = now
                    slouch_for = now - slouch_since
                    if slouch_for >= CONFIG.posture_slouch_alert_seconds and not slouch_alert_sent:
                        voice.maybe_speak(CONFIG.posture_alert_message, force=True)
                        print(f"[Voice] Triggered posture alert at {slouch_for:.1f}s slouch.")
                        slouch_alert_sent = True
                else:
                    if posture_good_since is None:
                        posture_good_since = now
                    if (now - posture_good_since) >= CONFIG.posture_recover_reset_seconds:
                        slouch_since = None
                        slouch_alert_sent = False
                    slouch_for = 0.0
            else:
                slouch_for = 0.0

            # LED support is currently disabled.
            # led.set_off_task(off_task)

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
            if off_task:
                cv2.putText(
                    vis.frame,
                    f"Off-task: {distracted_for:0.1f}s",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 120, 255),
                    2,
                )
            if posture_res.enabled and posture_res.calibrated and posture_res.good_posture is False:
                cv2.putText(
                    vis.frame,
                    f"Slouch: {slouch_for:0.1f}s",
                    (20, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            if show_window:
                cv2.imshow("Productivity Pi", vis.frame)
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
        lcd.close()
        # LED support is currently disabled.
        # led.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Productivity + anti-phone Raspberry Pi app")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--fullscreen", action="store_true", help="Show preview in fullscreen for external monitor")
    args = parser.parse_args()

    run(show_window=not args.headless, fullscreen=args.fullscreen)


if __name__ == "__main__":
    main()
