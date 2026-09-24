"""Basit betikli rakip (Faz 1).

SİNEK BEYNİ DEĞİLDİR. Kurallar: en yakın meyveye dön, hep ileri git.
Sinekliklerden kaçmaz. Arayüzde "SCRIPTED BOT" olarak etiketlenir.
"""

from __future__ import annotations

import math

from ..actions import Action
from ..sensing import GameState
from .base import Agent, AgentKind


class ScriptedAgent(Agent):
    kind = AgentKind.SCRIPTED
    name = "scripted-seeker"

    def __init__(self, turn_gain: float = 3.0):
        self.turn_gain = turn_gain

    def get_action(self, state: GameState) -> Action:
        s = state.sensory
        if math.isinf(s.nearest_fruit_distance):
            # Meyve yoksa: duvara çarpmamak için yavaşça dön
            return Action(0.6, 0.5)
        b = s.nearest_fruit_bearing
        turn = max(-1.0, min(1.0, self.turn_gain * b))
        # Meyve arkadaysa önce yerinde dön, sonra ilerle
        forward = 1.0 if abs(b) < math.radians(60) else 0.3
        return Action(turn, forward)
