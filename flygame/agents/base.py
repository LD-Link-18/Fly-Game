"""Ortak oyuncu arayüzü.

İnsan, betik (scripted) bot, sezgisel ajan, RL ajanı ve sinek beyni
simülasyonu aynı arayüzü uygular: get_action(game_state) -> Action.

Dürüstlük kuralı: her ajan türünü (kind) açıkça bildirir ve arayüzde bu
etiket gösterilir. Betikli veya kayıttan oynatılan bir oyuncu asla
"canlı sinek beyni" gibi gösterilmez.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from ..actions import Action
from ..config import GameConfig
from ..sensing import GameState


class AgentKind(str, Enum):
    HUMAN = "human"
    SCRIPTED = "scripted"      # elle yazılmış basit kurallar
    HEURISTIC = "heuristic"    # elle yazılmış daha akıllı kurallar
    RL = "rl"                  # öğrenilmiş politika
    FLY_BRAIN = "fly_brain"    # canlı çalışan konnektom simülasyonu
    REPLAY = "replay"          # daha önce kaydedilmiş bir koşunun tekrarı


# Arayüzde gösterilen dürüst etiketler
KIND_LABELS = {
    "en": {
        AgentKind.HUMAN: "HUMAN",
        AgentKind.SCRIPTED: "SCRIPTED BOT - not a fly brain",
        AgentKind.HEURISTIC: "HEURISTIC BOT - not a fly brain",
        AgentKind.RL: "RL AGENT - not a fly brain",
        AgentKind.FLY_BRAIN: "LIVE FLY-BRAIN SIMULATION",
        AgentKind.REPLAY: "RECORDED REPLAY",
    },
    "tr": {
        AgentKind.HUMAN: "İNSAN",
        AgentKind.SCRIPTED: "BETİKLİ BOT - sinek beyni değil",
        AgentKind.HEURISTIC: "SEZGİSEL BOT - sinek beyni değil",
        AgentKind.RL: "RL AJANI - sinek beyni değil",
        AgentKind.FLY_BRAIN: "CANLI SİNEK BEYNİ SİMÜLASYONU",
        AgentKind.REPLAY: "KAYITTAN TEKRAR",
    },
}


class Agent(ABC):
    kind: AgentKind = AgentKind.SCRIPTED
    name: str = "agent"

    def reset(self, seed: int, cfg: GameConfig) -> None:
        """Her tur başında çağrılır."""

    @abstractmethod
    def get_action(self, state: GameState) -> Action:
        """Bir simülasyon adımı için eylem döndür."""

    def telemetry(self) -> dict | None:
        """İsteğe bağlı: arayüzde gösterilecek iç durum (ör. nöron aktivitesi)."""
        return None

    def label(self, lang: str = "en") -> str:
        return KIND_LABELS.get(lang, KIND_LABELS["en"])[self.kind]

    def describe(self) -> dict:
        # Kayıt dosyalarına yazılan kimlik bilgisi
        return {"name": self.name, "kind": self.kind.value}
