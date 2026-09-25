"""Ajan fabrikası: komut satırındaki tanımdan ajan oluşturur.

Örnekler:
  scripted              -> basit betikli bot
  heuristic             -> meyve arayan, sinekliklerden kaçan sezgisel bot
  flybrain              -> canlı FlyWire v783 beyin simülasyonu (GPU, bkz. requirements-brain.txt)
  replay:runs/x.json.gz -> kayıttan tekrar
  replay:runs/          -> klasördeki kayıtlardan her tur rastgele biri
"""

from __future__ import annotations

from ..config import GameConfig
from .base import Agent, AgentKind
from .scripted import ScriptedAgent


def make_agent(spec: str, cfg: GameConfig | None = None) -> Agent:
    name, _, arg = spec.partition(":")
    if name == "scripted":
        return ScriptedAgent()
    if name == "heuristic":
        from .heuristic import HeuristicAgent

        return HeuristicAgent()
    if name == "flybrain":
        from .flybrain import FlyBrainAgent

        return FlyBrainAgent(cfg or GameConfig())
    if name == "replay":
        from .replay import ReplayAgent

        if not arg:
            raise ValueError("replay needs a path: replay:<file-or-dir>")
        return ReplayAgent(arg)
    raise ValueError(f"Unknown agent: {spec!r} (known: scripted, heuristic, flybrain, replay:<path>)")


__all__ = ["Agent", "AgentKind", "make_agent"]
