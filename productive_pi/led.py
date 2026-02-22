class FocusLed:
    def __init__(self, enabled: bool, pin: int, active_high: bool = True):
        self.enabled = enabled
        self.pin = pin
        self.active_high = active_high
        self._gpio = None

        if not enabled:
            return

        try:
            import RPi.GPIO as GPIO

            self._gpio = GPIO
            self._gpio.setwarnings(False)
            self._gpio.setmode(self._gpio.BCM)
            self._gpio.setup(self.pin, self._gpio.OUT, initial=self._inactive_level())
            print(f"[LED] Enabled on GPIO {pin} (active_high={active_high}).")
        except Exception as exc:
            self.enabled = False
            print(f"[LED] Disabled due to init error: {exc}")

    def _active_level(self) -> int:
        if self.active_high:
            return 1
        return 0

    def _inactive_level(self) -> int:
        if self.active_high:
            return 0
        return 1

    def set_off_task(self, off_task: bool) -> None:
        if not self.enabled or self._gpio is None:
            return
        try:
            if off_task:
                self._gpio.output(self.pin, self._active_level())
            else:
                self._gpio.output(self.pin, self._inactive_level())
        except Exception:
            pass

    def close(self) -> None:
        if self._gpio is not None:
            try:
                self._gpio.output(self.pin, self._inactive_level())
                self._gpio.cleanup(self.pin)
            except Exception:
                pass
