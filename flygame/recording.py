"""Koşu kaydı ve tekrarı.

Simülasyon belirlenimci olduğu için bir koşuyu yeniden üretmek için
tohum + ayarlar + adım adım eylem dizisi yeterlidir. Ek olarak ajanın
telemetrisi (ör. nöron aktivitesi) de saklanabilir; böylece yavaş bir
sinek beyni simülasyonu önceden koşturulup "hayalet" olarak gösterilebilir.

Dosya biçimi: gzip'li JSON (.json.gz).
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .actions import Action
from .config import GameConfig

FORMAT_VERSION = 1


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
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.pop("format", None) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported recording format")
        data["actions"] = [tuple(a) for a in data["actions"]]
        return cls(**data)


def new_recording(seed: int, agent_info: dict, cfg: GameConfig) -> Recording:
    return Recording(seed=seed, agent=agent_info, config=cfg.to_dict(), config_hash=cfg.gameplay_hash())
