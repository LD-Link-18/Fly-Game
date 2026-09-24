"""Tüm oyuncuların (insan, bot, RL, sinek beyni) ürettiği eylem tipi."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    # turn: -1 = tam sola, +1 = tam sağa (sinek kendi baktığı yöne göre)
    # forward: 0 = dur, 1 = tam hız ileri
    turn: float = 0.0
    forward: float = 0.0

    def clamped(self) -> "Action":
        return Action(
            turn=max(-1.0, min(1.0, float(self.turn))),
            forward=max(0.0, min(1.0, float(self.forward))),
        )


IDLE = Action(0.0, 0.0)
