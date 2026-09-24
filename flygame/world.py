"""Oyun simülasyonu: bir oyuncunun arenası.

Bu modül pygame kullanmaz; başsız (headless) çalışır, böylece ajan
ayarlama ve testler ekran olmadan yapılabilir. Simülasyon tamamen
belirlenimcidir: aynı ayar + aynı tohum + aynı eylem dizisi = aynı sonuç.

Koordinat kuralları (ekran koordinatları, y aşağı doğru):
  - heading: radyan, 0 = +x (sağa), açı arttıkça saat yönünde döner.
  - "Sağa dön" heading'i artırır, "sola dön" azaltır.
  - Göreli yön (bearing): hedefin sineğin baş yönüne göre açısı,
    pozitif = sineğin SAĞINDA, negatif = SOLUNDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .actions import Action
from .config import ArenaConfig, FlyConfig, GameConfig
from .schedule import Schedule, make_schedule


def wrap_angle(a: float) -> float:
    # Açıyı (-pi, pi] aralığına getir
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class FlyBody:
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0


def move_fly(fly: FlyBody, turn: float, forward: float, fc: FlyConfig, arena: ArenaConfig, dt: float) -> None:
    """Sineğin bir adımlık hareketi (oyun ve planlayıcı ajanlar aynı fiziği kullanır)."""
    fly.heading = wrap_angle(fly.heading + turn * math.radians(fc.turn_rate_deg) * dt)
    target_speed = forward * fc.max_speed
    tvx = math.cos(fly.heading) * target_speed
    tvy = math.sin(fly.heading) * target_speed
    k = min(1.0, fc.accel * dt)
    fly.vx += (tvx - fly.vx) * k
    fly.vy += (tvy - fly.vy) * k
    fly.x += fly.vx * dt
    fly.y += fly.vy * dt

    # Duvarlar: sineği içeride tut, duvara dik hız bileşenini sıfırla
    r = fc.radius
    if fly.x < r:
        fly.x, fly.vx = r, max(0.0, fly.vx)
    elif fly.x > arena.width - r:
        fly.x, fly.vx = arena.width - r, min(0.0, fly.vx)
    if fly.y < r:
        fly.y, fly.vy = r, max(0.0, fly.vy)
    elif fly.y > arena.height - r:
        fly.y, fly.vy = arena.height - r, min(0.0, fly.vy)


@dataclass
class Fruit:
    id: int
    x: float
    y: float
    kind: int
    spawned_at: float


@dataclass
class Swatter:
    id: int
    x: float
    y: float
    radius: float
    started_at: float
    loom_s: float
    impacted: bool = False
    hit_player: bool = False

    def progress(self, t: float) -> float:
        # 0 = gölge yeni belirdi, 1 = çarpma anı
        return min(1.0, max(0.0, (t - self.started_at) / self.loom_s))


@dataclass(frozen=True)
class WorldEvent:
    # Görüntü/ses için bu adımda olanlar (oyun mantığını etkilemez)
    kind: str  # "fruit", "fruit_expired", "swatter_spawn", "slam"
    x: float
    y: float
    hit: bool = False
    points: int = 0


@dataclass
class World:
    cfg: GameConfig
    seed: int
    schedule: Schedule | None = None

    t: float = 0.0
    tick: int = 0
    score: int = 0
    fruits_collected: int = 0
    hits: int = 0
    stun_left: float = 0.0
    fruits: list[Fruit] = field(default_factory=list)
    swatters: list[Swatter] = field(default_factory=list)
    events: list[WorldEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schedule is None:
            self.schedule = make_schedule(self.seed, self.cfg)
        a = self.cfg.arena
        # Sinek arenanın ortasında, yukarı bakarak başlar
        self.fly = FlyBody(a.width / 2, a.height / 2, -math.pi / 2)
        self._next_fruit = 0
        self._next_swatter = 0
        self._next_id = 0
        self.dt = 1.0 / self.cfg.round.tick_hz
        self.total_ticks = int(round(self.cfg.round.duration_s * self.cfg.round.tick_hz))
        self._spawn_due()

    # -- yardımcılar -------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.tick >= self.total_ticks

    @property
    def time_left(self) -> float:
        return max(0.0, (self.total_ticks - self.tick) * self.dt)

    @property
    def stunned(self) -> bool:
        return self.stun_left > 0.0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _spawn_due(self) -> None:
        # Zamanı gelen meyve ve sineklik olaylarını sahneye ekle
        sched = self.schedule
        eps = 1e-9
        while self._next_fruit < len(sched.fruits) and sched.fruits[self._next_fruit].t <= self.t + eps:
            s = sched.fruits[self._next_fruit]
            self.fruits.append(Fruit(self._new_id(), s.x, s.y, s.kind, self.t))
            self._next_fruit += 1

        sc = self.cfg.swatter
        a = self.cfg.arena
        while self._next_swatter < len(sched.swatters) and sched.swatters[self._next_swatter].t <= self.t + eps:
            d = sched.swatters[self._next_swatter]
            if d.aim_at_player:
                # Sineğin çarpma anında olacağı yeri kısmen tahmin et
                lead = sc.aim_lead * sc.loom_s
                tx = self.fly.x + self.fly.vx * lead + d.jitter_x
                ty = self.fly.y + self.fly.vy * lead + d.jitter_y
            else:
                tx, ty = d.x, d.y
            tx = min(max(tx, 0.0), a.width)
            ty = min(max(ty, 0.0), a.height)
            self.swatters.append(Swatter(self._new_id(), tx, ty, sc.radius, self.t, sc.loom_s))
            self.events.append(WorldEvent("swatter_spawn", tx, ty))
            self._next_swatter += 1

    # -- ana adım ----------------------------------------------------------

    def step(self, action: Action) -> None:
        if self.done:
            return
        self.events = []
        dt = self.dt
        fc = self.cfg.fly
        fly = self.fly
        action = action.clamped()

        # 1) Hareket: sersemlemişse eylemler yok sayılır, sinek yavaşlar
        if self.stunned:
            self.stun_left = max(0.0, self.stun_left - dt)
            move_fly(fly, 0.0, 0.0, fc, self.cfg.arena, dt)
        else:
            move_fly(fly, action.turn, action.forward, fc, self.cfg.arena, dt)
        r = fc.radius

        # Zaman ilerler
        self.tick += 1
        self.t = self.tick * dt

        # 2) Meyve toplama ve süresi dolan meyveler
        fr = self.cfg.fruit
        reach2 = (r + fr.radius) ** 2
        kept: list[Fruit] = []
        for f in self.fruits:
            if (f.x - fly.x) ** 2 + (f.y - fly.y) ** 2 <= reach2:
                self._add_score(fr.points)
                self.fruits_collected += 1
                self.events.append(WorldEvent("fruit", f.x, f.y, points=fr.points))
            elif self.t - f.spawned_at >= fr.lifetime_s:
                self.events.append(WorldEvent("fruit_expired", f.x, f.y))
            else:
                kept.append(f)
        self.fruits = kept

        # 3) Sineklikler: çarpma anında vuruş kontrolü
        sc = self.cfg.swatter
        alive: list[Swatter] = []
        for s in self.swatters:
            if not s.impacted and s.progress(self.t) >= 1.0:
                s.impacted = True
                hit = (s.x - fly.x) ** 2 + (s.y - fly.y) ** 2 <= s.radius ** 2
                s.hit_player = hit
                if hit:
                    self.hits += 1
                    self._add_score(-sc.hit_penalty_points)
                    if sc.stun_s > 0:
                        self.stun_left = max(self.stun_left, sc.stun_s)
                self.events.append(
                    WorldEvent("slam", s.x, s.y, hit=hit, points=-sc.hit_penalty_points if hit else 0)
                )
            if s.impacted and self.t - (s.started_at + s.loom_s) >= sc.impact_show_s:
                continue
            alive.append(s)
        self.swatters = alive

        # 4) Yeni olaylar
        self._spawn_due()

    def _add_score(self, delta: int) -> None:
        self.score += delta
        if not self.cfg.score.allow_negative:
            self.score = max(0, self.score)
