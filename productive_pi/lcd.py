class LcdDisplay:
    def __init__(self, enabled: bool, address: int, cols: int, rows: int):
        self.enabled = enabled
        self.cols = cols
        self.rows = rows
        self._lcd = None
        if not enabled:
            return

        try:
            from RPLCD.i2c import CharLCD

            self._lcd = CharLCD("PCF8574", address, cols=cols, rows=rows)
            self._lcd.clear()
        except Exception as exc:
            print(f"[LCD] Disabled due to init error: {exc}")
            self.enabled = False

    def write(self, line1: str, line2: str) -> None:
        if not self.enabled or self._lcd is None:
            return

        text1 = line1[: self.cols].ljust(self.cols)
        text2 = line2[: self.cols].ljust(self.cols)
        self._lcd.cursor_pos = (0, 0)
        self._lcd.write_string(text1)
        self._lcd.cursor_pos = (1, 0)
        self._lcd.write_string(text2)

    def close(self) -> None:
        if self._lcd is not None:
            try:
                self._lcd.clear()
            except Exception:
                pass
