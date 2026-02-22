class FocusLed:
    def __init__(self, enabled: bool, pin: int, active_high: bool = True):
        self.enabled = enabled
        self.pin = pin
        self._led = None

        if not enabled:
            return

        try:
            from gpiozero import LED

            self._led = LED(pin, active_high=active_high)
            self._led.off()
            print(f"[LED] Enabled on GPIO {pin} (active_high={active_high}).")
        except Exception as exc:
            self.enabled = False
            print(f"[LED] Disabled due to init error: {exc}")

    def set_off_task(self, off_task: bool) -> None:
        if not self.enabled or self._led is None:
            return
        try:
            if off_task:
                self._led.on()
            else:
                self._led.off()
        except Exception:
            pass

    def close(self) -> None:
        if self._led is not None:
            try:
                self._led.off()
                self._led.close()
            except Exception:
                pass
