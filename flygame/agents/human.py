"""Klavyeyle oynayan insan oyuncu (ok tuşları veya WASD)."""

from __future__ import annotations

from typing import Callable, Sequence

from ..actions import Action
from ..sensing import GameState
from .base import Agent, AgentKind


class HumanAgent(Agent):
    kind = AgentKind.HUMAN
    name = "human"

    def __init__(self, key_state: Callable[[], Sequence[bool]]):
        # key_state: pygame.key.get_pressed gibi bir fonksiyon
        import pygame

        self._keys = key_state
        self._left = (pygame.K_LEFT, pygame.K_a)
        self._right = (pygame.K_RIGHT, pygame.K_d)
        self._fwd = (pygame.K_UP, pygame.K_w)

    def get_action(self, state: GameState) -> Action:
        k = self._keys()
        turn = float(any(k[c] for c in self._right)) - float(any(k[c] for c in self._left))
        forward = 1.0 if any(k[c] for c in self._fwd) else 0.0
        return Action(turn, forward)
