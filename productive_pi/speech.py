from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import requests


@dataclass
class VoiceConfig:
    enabled: bool
    backend: str
    local_voice_name: str
    api_key: str
    voice_id: str
    model_id: str
    elevenlabs_fallback_local: bool
    debug: bool
    trigger_seconds: float
    cooldown_seconds: float


class VoiceAlerter:
    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        self._last_spoken = 0.0
        self._busy = False
        self._lock = threading.Lock()
        self._player = self._choose_player() if self.cfg.backend == "elevenlabs" else None
        self._local_tts_cmd = self._choose_local_tts()

        if self.cfg.enabled and self.cfg.backend == "elevenlabs" and not self._player:
            print("[Voice] ElevenLabs audio playback tool missing (need one of: mpg123, ffplay, cvlc).")
        if self.cfg.enabled and self.cfg.backend == "local" and not self._local_tts_cmd:
            print("[Voice] No local TTS command found (need one of: say, espeak, spd-say).")
        if self.cfg.debug:
            print(
                f"[Voice][Debug] enabled={self.cfg.enabled} backend={self.cfg.backend} "
                f"voice_id={self.cfg.voice_id} api_key_set={bool(self.cfg.api_key)} "
                f"player={self._player} local_tts={self._local_tts_cmd}"
            )

    @staticmethod
    def _choose_player() -> list[str] | None:
        if shutil.which("afplay"):
            # macOS built-in audio player; works with mp3.
            return ["afplay"]
        if shutil.which("mpg123"):
            return ["mpg123", "-q"]
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
        if shutil.which("cvlc"):
            return ["cvlc", "--play-and-exit", "--quiet"]
        return None

    def _choose_local_tts(self) -> list[str] | None:
        if shutil.which("say"):
            # macOS built-in, free/offline.
            return ["say", "-v", self.cfg.local_voice_name]
        if shutil.which("espeak"):
            return ["espeak"]
        if shutil.which("spd-say"):
            return ["spd-say"]
        return None

    def _speak_local(self, text: str) -> None:
        if not self._local_tts_cmd:
            return
        subprocess.run([*self._local_tts_cmd, text], check=False)

    def _speak_elevenlabs(self, text: str) -> None:
        if not self.cfg.api_key:
            print("[Voice] ELEVENLABS_API_KEY is missing.")
            return
        if not self._player:
            return

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.cfg.voice_id}"
        headers = {
            "xi-api-key": self.cfg.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.cfg.model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
            },
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=(5, 20))
        if resp.status_code >= 400:
            body = (resp.text or "")[:280].replace("\n", " ")
            raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {body}")
        if not resp.content:
            raise RuntimeError("ElevenLabs returned empty audio payload.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)

        try:
            subprocess.run([*self._player, str(tmp_path)], check=False)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _speak_worker(self, text: str) -> None:
        try:
            if self.cfg.backend == "elevenlabs":
                try:
                    self._speak_elevenlabs(text)
                except Exception as exc:
                    if self.cfg.elevenlabs_fallback_local:
                        print(f"[Voice] ElevenLabs failed, falling back to local TTS: {exc}")
                        self._speak_local(text)
                    else:
                        print(f"[Voice] ElevenLabs failed (fallback disabled): {exc}")
            else:
                self._speak_local(text)
        except Exception as exc:
            print(f"[Voice] TTS failed: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def maybe_speak(self, text: str, force: bool = False) -> bool:
        if not self.cfg.enabled:
            return False

        now = time.monotonic()
        if (not force) and (now - self._last_spoken < self.cfg.cooldown_seconds):
            return False

        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._last_spoken = now

        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()
        return True
