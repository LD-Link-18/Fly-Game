"""Kayıttan tekrar ("hayalet") oyuncu.

CANLI DEĞİLDİR: daha önce kaydedilmiş bir koşunun eylemlerini adım adım
geri oynatır. Arayüzde her zaman "RECORDED REPLAY" olarak ve kaydın
kaynağıyla (ör. sinek beyni koşusu) birlikte etiketlenir.

Tekrar yalnızca kaydın tohumu ve aynı oyun ayarlarıyla doğru sonucu verir;
bu yüzden tur tohumunu bu ajan belirler (pick_seed).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from ..actions import Action
from ..config import GameConfig
from ..recording import Recording
from ..sensing import GameState
from .base import Agent, AgentKind

SOURCE_NAMES = {
    "en": {
        "fly_brain": "of a fly-brain simulation run",
        "scripted": "of a scripted bot",
        "heuristic": "of a heuristic bot",
        "rl": "of an RL agent",
        "human": "of a human player",
    },
    "tr": {
        "fly_brain": "- sinek beyni simülasyonu koşusu",
        "scripted": "- betikli bot",
        "heuristic": "- sezgisel bot",
        "rl": "- RL ajanı",
        "human": "- insan oyuncu",
    },
}


class ReplayAgent(Agent):
    kind = AgentKind.REPLAY

    def __init__(self, path: str | Path, rng: random.Random | None = None):
        p = Path(path)
        # Klasör verilirse her turda içinden rastgele bir kayıt seçilir
        self.paths = sorted(p.glob("*.json.gz")) if p.is_dir() else [p]
        if not self.paths:
            raise FileNotFoundError(f"No recordings found in {p}")
        self._rng = rng or random.Random()
        self.rec: Recording = Recording.load(self.paths[0])
        self.name = f"replay:{self.rec.agent.get('name', '?')}"

    def pick_seed(self) -> int:
        """Tur başlamadan çağrılır; bir kayıt seçer ve onun tohumunu döndürür."""
        self.rec = Recording.load(self._rng.choice(self.paths))
        self.name = f"replay:{self.rec.agent.get('name', '?')}"
        return self.rec.seed

    def reset(self, seed: int, cfg: GameConfig) -> None:
        if seed != self.rec.seed:
            raise ValueError(f"Replay needs seed {self.rec.seed}, round uses {seed}")
        if cfg.gameplay_hash() != self.rec.config_hash:
            print(
                "WARNING: recording was made with different gameplay settings; "
                "the replay may diverge from the original run.",
                file=sys.stderr,
            )

    def get_action(self, state: GameState) -> Action:
        self._tick = state.raw.tick
        return self.rec.action_at(self._tick)

    def telemetry(self) -> dict | None:
        # Kayıtta telemetri (ör. nöron aktivitesi) varsa o adımınkini aynen geri ver
        tel = self.rec.telemetry
        tick = getattr(self, "_tick", -1)
        if tel and 0 <= tick < len(tel):
            return tel[tick]
        return None

    def ui_note(self, lang: str = "en") -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.rec.created))
        return f"kayıt: {when}, tohum {self.rec.seed}" if lang == "tr" else f"recorded {when}, seed {self.rec.seed}"

    @property
    def source_kind(self) -> str:
        return self.rec.agent.get("kind", "?")

    def label(self, lang: str = "en") -> str:
        base = super().label(lang)
        src = SOURCE_NAMES.get(lang, SOURCE_NAMES["en"]).get(self.source_kind, self.source_kind)
        return f"{base} {src}"
