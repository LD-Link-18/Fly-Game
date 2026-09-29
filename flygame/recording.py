"""Koşu kaydı ve tekrarı.

Simülasyon belirlenimci olduğu için bir koşuyu yeniden üretmek için
tohum + ayarlar + adım adım eylem dizisi yeterlidir. Ek olarak ajanın
telemetrisi (ör. nöron aktivitesi) de saklanabilir; böylece yavaş bir
sinek beyni simülasyonu önceden koşturulup "hayalet" olarak gösterilebilir.

Dosya biçimi: gzip'li JSON (.json.gz).

Sinek beyni telemetrisindeki "fired" listeleri (o adımda ateşleyen tüm nöronlar, beyin
haritası için) JSON'da adım adım yazılsa tur başına ~10-30 MB tutardı. Bu yüzden
kaydederken tek bir bit matrisine paketlenir (satır = adım, sütun = turda en az bir kez
ateşleyen nöron; zlib + base64, tur başına ~0.5 MB) ve yüklerken geri açılır.
"""

from __future__ import annotations

import base64
import gzip
import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .actions import Action
from .config import GameConfig

FORMAT_VERSION = 1
SPIKE_KEY = "fired"  # telemetri anahtarı: o adımda ateşleyen nöronların model indeksleri


@dataclass
class Recording:
    seed: int
    agent: dict                      # {"name": ..., "kind": ...}
    config: dict                     # kayıt anındaki tüm ayarlar
    config_hash: str                 # oyun sonucunu etkileyen ayarların özeti
    actions: list[tuple[float, float]] = field(default_factory=list)
    telemetry: list | None = None    # adım başına isteğe bağlı veri
    final_score: int | None = None
    fruits: int | None = None
    hits: int | None = None
    created: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)  # ör. ölçülen simülasyon hızı

    def add(self, action: Action, telemetry=None) -> None:
        self.actions.append((action.turn, action.forward))
        if telemetry is not None:
            if self.telemetry is None:
                self.telemetry = [None] * (len(self.actions) - 1)
            self.telemetry.append(telemetry)
        elif self.telemetry is not None:
            self.telemetry.append(None)

    def action_at(self, tick: int) -> Action:
        if 0 <= tick < len(self.actions):
            t, f = self.actions[tick]
            return Action(t, f)
        return Action()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": FORMAT_VERSION, **self.__dict__}
        if self.telemetry:
            data["telemetry"], spikes = pack_spikes(self.telemetry)
            if spikes:
                data["spikes"] = spikes
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(data))  # tek parça yazmak, json.dump'ın küçük parçalarından çok hızlı
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.pop("format", None) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported recording format")
        data["actions"] = [tuple(a) for a in data["actions"]]
        spikes = data.pop("spikes", None)
        rec = cls(**data)
        if spikes and rec.telemetry:
            unpack_spikes(rec.telemetry, spikes)
        return rec


def new_recording(seed: int, agent_info: dict, cfg: GameConfig) -> Recording:
    return Recording(seed=seed, agent=agent_info, config=cfg.to_dict(), config_hash=cfg.gameplay_hash())


def pack_spikes(telemetry: list) -> tuple[list, dict | None]:
    """Adım başına "fired" listelerini tek bir sıkıştırılmış bit matrisine çevir.

    Dönüş: ("fired" anahtarı çıkarılmış telemetri kopyası, paket veya None)."""
    ticks = [i for i, t in enumerate(telemetry) if t and SPIKE_KEY in t]
    if not ticks:
        return telemetry, None
    lists = [np.asarray(telemetry[i][SPIKE_KEY], dtype=np.int64) for i in ticks]
    neurons = np.unique(np.concatenate(lists))
    bits = np.zeros((len(telemetry), len(neurons)), dtype=bool)
    for i, idx in zip(ticks, lists):
        bits[i, np.searchsorted(neurons, idx)] = True
    blob = zlib.compress(np.packbits(bits, axis=1).tobytes(), 6)
    stripped = [{k: v for k, v in t.items() if k != SPIKE_KEY} if t else t for t in telemetry]
    return stripped, {"neurons": neurons.tolist(), "ticks": len(telemetry),
                      "bits": base64.b64encode(blob).decode("ascii")}


def unpack_spikes(telemetry: list, spikes: dict) -> None:
    """pack_spikes'ın tersi: her telemetri adımına "fired" dizisini geri koy (yerinde)."""
    neurons = np.asarray(spikes["neurons"], dtype=np.int32)
    T, U = int(spikes["ticks"]), len(neurons)
    raw = np.frombuffer(zlib.decompress(base64.b64decode(spikes["bits"])), dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(T, (U + 7) // 8), axis=1, count=U).astype(bool)
    for i, t in enumerate(telemetry[:T]):
        if t is not None:
            t[SPIKE_KEY] = neurons[bits[i]]
