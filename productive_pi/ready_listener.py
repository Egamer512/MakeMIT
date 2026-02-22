from __future__ import annotations

import re
import time

import numpy as np


class ReadyPhraseListener:
    def __init__(
        self,
        enabled: bool,
        phrase: str,
        model_name: str,
        chunk_seconds: float,
        timeout_seconds: float,
        debug: bool,
    ):
        self.enabled = enabled
        self.phrase = phrase.strip() or "i'm ready"
        self.model_name = model_name
        self.chunk_seconds = max(1.0, chunk_seconds)
        self.timeout_seconds = max(0.0, timeout_seconds)
        self.debug = debug

        self._model = None
        self._sd = None

        if not self.enabled:
            return

        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel

            self._sd = sd
            # CPU-friendly default for Pi; change model in .env if needed.
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            print(f"[Ready] Voice gate enabled. Say: {self.phrase}")
        except Exception as exc:
            self.enabled = False
            print(f"[Ready] Voice gate disabled (init error): {exc}")

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def wait_for_phrase(self) -> bool:
        if not self.enabled:
            return True
        if self._model is None or self._sd is None:
            return True

        phrase_norm = self._normalize(self.phrase)
        sample_rate = 16000
        start = time.monotonic()

        while True:
            if self.timeout_seconds > 0 and (time.monotonic() - start) > self.timeout_seconds:
                print("[Ready] Timeout reached; continuing without voice confirmation.")
                return True

            frames = int(sample_rate * self.chunk_seconds)
            audio = self._sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
            self._sd.wait()

            audio = np.squeeze(audio)
            if audio.size == 0:
                continue

            # Skip near-silence chunks.
            if float(np.sqrt(np.mean(audio * audio))) < 0.01:
                continue

            try:
                segments, _ = self._model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
                transcript = " ".join(seg.text for seg in segments).strip()
                text_norm = self._normalize(transcript)

                if self.debug and text_norm:
                    print(f"[Ready][Debug] heard: {text_norm}")

                if phrase_norm and phrase_norm in text_norm:
                    print("[Ready] Phrase detected. Starting monitoring.")
                    return True
            except Exception as exc:
                if self.debug:
                    print(f"[Ready][Debug] transcribe error: {exc}")

        return True
