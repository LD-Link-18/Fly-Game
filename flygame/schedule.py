"""Tur deseni: meyve ve sineklik olaylarının tohumdan (seed) önceden üretilmesi.

Bölünmüş ekranda iki tarafın AYNI deseni oynaması için tüm rastgelelik
burada, tur başlamadan önce üretilir. Simülasyon sırasında oyun RNG'si
hiç kullanılmaz; böylece oyuncunun hareketleri deseni değiştiremez.

Not: Sineklik "oyuncuya nişan al" türündeyse hedef noktası, olayın
başladığı andaki sinek konumuna göre hesaplanır. Zamanlama, nişan türü ve
sapma iki tarafta birebir aynıdır; yalnızca nişan her tarafın kendi
sineğine göre alınır.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import GameConfig


@dataclass(frozen=True)
class FruitSpawn:
    t: float
    x: float
    y: float
    kind: int  # yalnızca görünüm (renk/tür)


@dataclass(frozen=True)
class SwatterDrop:
    t: float
    aim_at_player: bool
    x: float        # aim_at_player False ise kullanılan sabit hedef
    y: float
    jitter_x: float  # nişana eklenen sapma
    jitter_y: float


@dataclass(frozen=True)
class Schedule:
    seed: int
    fruits: tuple[FruitSpawn, ...]
    swatters: tuple[SwatterDrop, ...]


FRUIT_KINDS = 4


def make_schedule(seed: int, cfg: GameConfig) -> Schedule:
    # Meyve ve sineklik için ayrı RNG akışları: birinin ayarı değişince
    # diğerinin deseni kaymasın
    fruit_rng = random.Random(seed * 7919 + 1)
    swat_rng = random.Random(seed * 7919 + 2)
    w, h = cfg.arena.width, cfg.arena.height
    duration = cfg.round.duration_s

    fc = cfg.fruit
    fruits: list[FruitSpawn] = []

    def rand_pos(rng: random.Random, margin: float) -> tuple[float, float]:
        return rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)

    for _ in range(fc.initial_count):
        x, y = rand_pos(fruit_rng, fc.margin)
        fruits.append(FruitSpawn(0.0, x, y, fruit_rng.randrange(FRUIT_KINDS)))
    t = 0.0
    while True:
        t += max(0.05, fc.spawn_interval_s + fruit_rng.uniform(-fc.spawn_jitter_s, fc.spawn_jitter_s))
        if t >= duration:
            break
        x, y = rand_pos(fruit_rng, fc.margin)
        fruits.append(FruitSpawn(t, x, y, fruit_rng.randrange(FRUIT_KINDS)))

    sc = cfg.swatter
    swatters: list[SwatterDrop] = []
    t = sc.first_at_s
    # Tur bitmeden çarpamayacak sineklikleri üretme
    while t + sc.loom_s <= duration:
        aim = swat_rng.random() < sc.aim_at_player_prob
        x, y = rand_pos(swat_rng, sc.radius * 0.5)
        jx = swat_rng.gauss(0.0, sc.aim_jitter)
        jy = swat_rng.gauss(0.0, sc.aim_jitter)
        swatters.append(SwatterDrop(t, aim, x, y, jx, jy))
        t += swat_rng.uniform(sc.interval_min_s, sc.interval_max_s)

    return Schedule(seed, tuple(fruits), tuple(swatters))
