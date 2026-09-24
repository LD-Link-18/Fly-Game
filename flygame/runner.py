"""Ekransız tur koşturucu: bir ajanı verilen tohumla baştan sona oynatır.

Testler, kayıt alma ve (Faz 4'te) parametre ayarlama bunu kullanır.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .agents.base import Agent
from .config import GameConfig
from .recording import Recording, new_recording
from .sensing import build_state
from .world import World


@dataclass
class RoundResult:
    seed: int
    score: int
    fruits: int
    hits: int
    wall_time_s: float     # koşunun gerçek süresi
    sim_time_s: float      # oyun içi süre
    recording: Recording | None = None

    @property
    def realtime_factor(self) -> float:
        # > 1 ise gerçek zamandan hızlı
        return self.sim_time_s / self.wall_time_s if self.wall_time_s > 0 else float("inf")


def run_round(agent: Agent, seed: int, cfg: GameConfig, record: bool = False) -> RoundResult:
    world = World(cfg, seed)
    agent.reset(seed, cfg)
    rec = new_recording(seed, agent.describe(), cfg) if record else None
    t0 = time.perf_counter()
    while not world.done:
        action = agent.get_action(build_state(world)).clamped()
        if rec is not None:
            rec.add(action, agent.telemetry())
        world.step(action)
    wall = time.perf_counter() - t0
    if rec is not None:
        rec.final_score, rec.fruits, rec.hits = world.score, world.fruits_collected, world.hits
        rec.meta["wall_time_s"] = wall
    return RoundResult(seed, world.score, world.fruits_collected, world.hits, wall, world.t, rec)
