"""Sesler: dosya gerektirmeden numpy ile sentezlenir.

Ses aygıtı yoksa (ör. sunucu, test) sessizce devre dışı kalır.
"""

from __future__ import annotations

import numpy as np

RATE = 44100


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


def _slam(duration: float = 0.9) -> np.ndarray:
    # Derin gümleme + şaklama sesi (düşen sinekliğin çarpması)
    n = int(RATE * duration)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(1)
    freq = 55 + 90 * np.exp(-t * 18)                     # aşağı kayan bas
    thump = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * _env(n, 0.002, 0.22)
    crack = rng.standard_normal(n) * _env(n, 0.0005, 0.035)
    rumble = np.convolve(rng.standard_normal(n), np.ones(60) / 60, "same") * _env(n, 0.01, 0.35) * 3
    return 1.1 * thump + 0.7 * crack + 0.5 * rumble


def _whoosh(duration: float) -> np.ndarray:
    # Yaklaşan sineklik: yükselen, gittikçe güçlenen hava sesi
    n = int(RATE * duration)
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n)
    # Basit alçak geçiren filtre; kesim frekansı zamanla yükselir
    out = np.empty(n)
    y = 0.0
    alpha = 0.02 + 0.25 * t ** 2
    for i in range(n):
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    return out * (t ** 2.2) * 2.2


def _blip(f0: float, f1: float, duration: float = 0.12) -> np.ndarray:
    # Meyve toplama sesi: kısa, tatlı bir yukarı kayış
    n = int(RATE * duration)
    freq = np.linspace(f0, f1, n)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    return 0.5 * tone * _env(n, 0.003, duration / 3)


def _chime(notes: list[float], step: float = 0.11) -> np.ndarray:
    # Sonuç/geri sayım için kısa melodi
    parts = []
    for f in notes:
        n = int(RATE * step * 2)
        t = np.arange(n) / RATE
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
        parts.append((tone * _env(n, 0.005, step * 0.8), int(RATE * step)))
    total = sum(off for _, off in parts) + len(parts[-1][0])
    out = np.zeros(total)
    pos = 0
    for tone, off in parts:
        out[pos:pos + len(tone)] += tone
        pos += off
    return out * 0.35


class Sounds:
    def __init__(self, enabled: bool, volume: float, loom_s: float):
        self.ok = False
        if not enabled:
            return
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(RATE, -16, 2, 512)
            self._pg = pygame
            self._s = {
                "slam": self._make(_slam(), volume),
                "whoosh": self._make(_whoosh(loom_s), volume * 0.55),
                "fruit": self._make(_blip(660, 1320), volume * 0.6),
                "fruit_other": self._make(_blip(520, 780), volume * 0.2),
                "tick": self._make(_blip(880, 880, 0.08), volume * 0.5),
                "go": self._make(_blip(880, 1760, 0.25), volume * 0.6),
                "win": self._make(_chime([523, 659, 784, 1047]), volume),
                "lose": self._make(_chime([392, 330, 262], 0.16), volume),
            }
            self.ok = True
        except Exception as e:  # ses aygıtı yoksa oyun sessiz devam eder
            print(f"Sound disabled: {e}")

    def _make(self, wave: np.ndarray, volume: float):
        wave = np.clip(wave / max(1e-9, np.max(np.abs(wave))) * volume, -1, 1)
        pcm = (wave * 32767).astype(np.int16)
        return self._pg.sndarray.make_sound(np.ascontiguousarray(np.column_stack([pcm, pcm])))

    def play(self, name: str) -> None:
        if self.ok:
            self._s[name].play()
