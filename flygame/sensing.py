"""Ajanlara verilen oyun durumu: ham (raw) ve duyusal (sensory) biçim.

- raw: konumlar, hızlar, meyveler, sineklikler (klasik bot/RL için).
- sensory: sineğin kendi bakış açısından, sol/sağ ayrılmış sinyaller.
  Sinek beyni adaptörü bu sinyalleri duyusal nöron uyarımına çevirir.

Duyusal sinyaller (hepsi >= 0, çoğu kabaca 0..1 aralığında):
  odor_left/right   : iki antendeki toplam meyve "kokusu", exp(-d/L) toplamı.
                      Fark küçüktür; yön bilgisi farkta saklıdır.
  fruit_left/right  : görsel meyve belirginliği, sol/sağ görüş alanına bölünmüş.
  loom_left/right   : sineklik yaklaşma (looming) sinyali: gölgenin sinekten
                      görülen açısal boyutunun büyüme hızı (rad/sn), sol/sağa bölünmüş.
  loom_size         : en büyük sinekliğin açısal boyutu (radyan).
  wall_left/front/right : duvar yakınlığı (1 = dibinde, 0 = menzil dışı).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .world import World, wrap_angle


@dataclass(frozen=True)
class FlyRaw:
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    speed: float
    stunned: bool
    stun_left: float


@dataclass(frozen=True)
class FruitRaw:
    id: int
    x: float
    y: float
    distance: float
    bearing: float     # pozitif = sağda
    time_left: float


@dataclass(frozen=True)
class SwatterRaw:
    id: int
    x: float
    y: float
    radius: float
    distance: float        # sinek merkezinden vuruş merkezine
    bearing: float         # pozitif = sağda
    progress: float        # 0..1 (1 = çarptı)
    time_to_impact: float  # sn (çarptıysa 0)
    impacted: bool


@dataclass(frozen=True)
class RawState:
    t: float
    time_left: float
    tick: int
    dt: float
    arena_w: float
    arena_h: float
    score: int
    fly: FlyRaw
    fruits: tuple[FruitRaw, ...]
    swatters: tuple[SwatterRaw, ...]


@dataclass(frozen=True)
class SensoryState:
    odor_left: float
    odor_right: float
    fruit_left: float
    fruit_right: float
    nearest_fruit_bearing: float   # meyve yoksa 0
    nearest_fruit_distance: float  # meyve yoksa inf
    loom_left: float
    loom_right: float
    loom_size: float
    wall_left: float
    wall_front: float
    wall_right: float
    stunned: bool


@dataclass(frozen=True)
class GameState:
    raw: RawState
    sensory: SensoryState


def _ray_to_wall(x: float, y: float, ang: float, w: float, h: float) -> float:
    # Noktadan verilen yönde dikdörtgen arenanın kenarına uzaklık
    dx, dy = math.cos(ang), math.sin(ang)
    best = math.inf
    if dx > 1e-9:
        best = min(best, (w - x) / dx)
    elif dx < -1e-9:
        best = min(best, -x / dx)
    if dy > 1e-9:
        best = min(best, (h - y) / dy)
    elif dy < -1e-9:
        best = min(best, -y / dy)
    return max(0.0, best)


def swatter_angular_size(horizontal_dist: float, radius: float, height: float) -> float:
    # Sinekten bakınca sinekliğin kapladığı açı (basit 3B model):
    # sineklik yarıçapı R, yatay uzaklık d, yükseklik z'de bir disk.
    return 2.0 * math.atan2(radius, math.hypot(horizontal_dist, height))


def swatter_height(progress: float, drop_height: float) -> float:
    # İvmeli düşüş: başta yavaş, sona doğru hızlı (serbest düşüş gibi)
    return drop_height * (1.0 - progress * progress)


def build_state(world: World) -> GameState:
    cfg = world.cfg
    sc = cfg.sensory
    fly = world.fly
    t = world.t

    def bearing_to(x: float, y: float) -> float:
        return wrap_angle(math.atan2(y - fly.y, x - fly.x) - fly.heading)

    # --- meyveler -----------------------------------------------------------
    fruits = []
    odor_l = odor_r = 0.0
    vis_l = vis_r = 0.0
    ant = math.radians(sc.antenna_angle_deg)
    # Sol anten baş yönünün solunda (açı azalır), sağ anten sağında
    lx = fly.x + math.cos(fly.heading - ant) * sc.antenna_offset
    ly = fly.y + math.sin(fly.heading - ant) * sc.antenna_offset
    rx = fly.x + math.cos(fly.heading + ant) * sc.antenna_offset
    ry = fly.y + math.sin(fly.heading + ant) * sc.antenna_offset
    nearest_d, nearest_b = math.inf, 0.0
    for f in world.fruits:
        d = math.hypot(f.x - fly.x, f.y - fly.y)
        b = bearing_to(f.x, f.y)
        fruits.append(FruitRaw(f.id, f.x, f.y, d, b, cfg.fruit.lifetime_s - (t - f.spawned_at)))
        odor_l += math.exp(-math.hypot(f.x - lx, f.y - ly) / sc.odor_length)
        odor_r += math.exp(-math.hypot(f.x - rx, f.y - ry) / sc.odor_length)
        sal = math.exp(-d / sc.odor_length)
        wr = 0.5 * (1.0 + math.sin(b))  # sağ görüş alanı ağırlığı
        vis_r += sal * wr
        vis_l += sal * (1.0 - wr)
        if d < nearest_d:
            nearest_d, nearest_b = d, b

    # --- sineklikler (looming) --------------------------------------------
    swatters = []
    loom_l = loom_r = 0.0
    loom_size = 0.0
    drop = cfg.swatter.drop_height
    for s in world.swatters:
        d = math.hypot(s.x - fly.x, s.y - fly.y)
        b = bearing_to(s.x, s.y)
        p = s.progress(t)
        ttl = 0.0 if s.impacted else max(0.0, s.started_at + s.loom_s - t)
        swatters.append(SwatterRaw(s.id, s.x, s.y, s.radius, d, b, p, ttl, s.impacted))
        if s.impacted:
            continue
        # Açısal boyut ve büyüme hızı (sayısal türev, sadece nesnenin hareketi)
        theta = swatter_angular_size(d, s.radius, swatter_height(p, drop))
        p_prev = max(0.0, p - world.dt / s.loom_s)
        theta_prev = swatter_angular_size(d, s.radius, swatter_height(p_prev, drop))
        rate = max(0.0, (theta - theta_prev) / world.dt)
        loom_size = max(loom_size, theta)
        # Tam tepedeyse iki tarafa eşit; uzaklaştıkça yan bilgisi netleşir
        lateral = math.sin(b) * min(1.0, d / max(1e-6, s.radius))
        wr = 0.5 * (1.0 + lateral)
        loom_r += rate * wr
        loom_l += rate * (1.0 - wr)

    # --- duvarlar -------------------------------------------------------------
    w, h = cfg.arena.width, cfg.arena.height
    wa = math.radians(sc.wall_ray_angle_deg)

    def prox(ang: float) -> float:
        return max(0.0, 1.0 - _ray_to_wall(fly.x, fly.y, ang, w, h) / sc.wall_ray_range)

    speed = math.hypot(fly.vx, fly.vy)
    raw = RawState(
        t=t,
        time_left=world.time_left,
        tick=world.tick,
        dt=world.dt,
        arena_w=w,
        arena_h=h,
        score=world.score,
        fly=FlyRaw(fly.x, fly.y, fly.heading, fly.vx, fly.vy, speed, world.stunned, world.stun_left),
        fruits=tuple(fruits),
        swatters=tuple(swatters),
    )
    sensory = SensoryState(
        odor_left=odor_l,
        odor_right=odor_r,
        fruit_left=vis_l,
        fruit_right=vis_r,
        nearest_fruit_bearing=nearest_b,
        nearest_fruit_distance=nearest_d,
        loom_left=loom_l,
        loom_right=loom_r,
        loom_size=loom_size,
        wall_left=prox(fly.heading - wa),
        wall_front=prox(fly.heading),
        wall_right=prox(fly.heading + wa),
        stunned=world.stunned,
    )
    return GameState(raw, sensory)
