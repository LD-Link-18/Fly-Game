"""Ajan fabrikası: komut satırındaki tanımdan ajan oluşturur.

Örnekler:
  scripted              -> basit betikli bot
  replay:runs/x.json.gz -> kayıttan tekrar
  replay:runs/          -> klasördeki kayıtlardan her tur rastgele biri
"""

from __future__ import annotations

from .base import Agent, AgentKind
from .scripted import ScriptedAgent


def make_agent(spec: str) -> Agent:
    name, _, arg = spec.partition(":")
    if name == "scripted":
        return ScriptedAgent()
    if name == "replay":
        from .replay import ReplayAgent

        if not arg:
            raise ValueError("replay needs a path: replay:<file-or-dir>")
        return ReplayAgent(arg)
    raise ValueError(f"Unknown agent: {spec!r} (known: scripted, replay:<path>)")


__all__ = ["Agent", "AgentKind", "make_agent"]
