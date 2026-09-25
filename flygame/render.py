"""Çizim: bölünmüş ekran, sinek/meyve/sineklik görselleri, efektler.

Bu modüldeki rastgelelik (parçacıklar, sarsıntı) yalnızca görseldir ve
oyun RNG'sinden tamamen ayrıdır; oyun sonucunu etkilemez.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame
import pygame.gfxdraw

from .agents.base import AgentKind
from .config import GameConfig
from .sensing import swatter_height
from .world import World, WorldEvent

SS = 4  # sprite'lar bu kat büyük çizilip küçültülür (kenar yumuşatma için)

BG = (18, 20, 28)
TEXT = (240, 240, 245)
DIM = (150, 155, 170)
HUMAN_COLOR = (70, 200, 255)
OPPONENT_COLOR = (255, 150, 50)
KIND_BADGE = {
    AgentKind.HUMAN: (70, 200, 255),
    AgentKind.SCRIPTED: (150, 150, 160),
    AgentKind.HEURISTIC: (150, 150, 160),
    AgentKind.RL: (170, 140, 220),
    AgentKind.FLY_BRAIN: (90, 220, 120),
    AgentKind.REPLAY: (240, 200, 60),
}
FRUIT_COLORS = [(220, 40, 50), (255, 140, 20), (130, 60, 170), (245, 205, 40)]


# --- sprite üretimi --------------------------------------------------------

def _aa_ellipse(surf, color, rect):
    pygame.gfxdraw.filled_ellipse(surf, int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2),
                                  int(rect[2] / 2), int(rect[3] / 2), color)
    pygame.gfxdraw.aaellipse(surf, int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2),
                             int(rect[2] / 2), int(rect[3] / 2), color)


def _aa_circle(surf, color, center, r):
    pygame.gfxdraw.filled_circle(surf, int(center[0]), int(center[1]), int(r), color)
    pygame.gfxdraw.aacircle(surf, int(center[0]), int(center[1]), int(r), color)


def make_fly_sprite(length_px: float) -> pygame.Surface:
    # Sağa (+x) bakan, yukarıdan görünen sinek
    L = max(8, int(length_px * SS))
    s = pygame.Surface((L, L), pygame.SRCALPHA)
    c = L / 2
    u = L / 10  # birim
    # Bacaklar
    for side in (-1, 1):
        for i, ang in enumerate((-50, 0, 45)):
            a = math.radians(ang)
            x0, y0 = c + (0.6 - i * 0.6) * u, c + side * 0.5 * u
            x1 = x0 + math.sin(a) * 1.6 * u * (1 if i != 2 else -1) + (0.5 * u if i == 0 else -0.3 * u)
            y1 = y0 + side * 1.9 * u
            pygame.draw.line(s, (40, 35, 30), (x0, y0), (x1, y1), max(2, int(u * 0.25)))
    # Kanatlar (yarı saydam, geriye açılı)
    for side in (-1, 1):
        wing = pygame.Surface((int(4.4 * u), int(1.9 * u)), pygame.SRCALPHA)
        _aa_ellipse(wing, (200, 220, 240, 150), (0, 0, wing.get_width(), wing.get_height()))
        pygame.draw.ellipse(wing, (120, 140, 160, 180), wing.get_rect(), max(1, int(u * 0.12)))
        wing = pygame.transform.rotate(wing, side * 22)
        wr = wing.get_rect(center=(c - 1.7 * u, c + side * 1.25 * u))
        s.blit(wing, wr)
    # Karın, göğüs, baş
    _aa_ellipse(s, (70, 55, 40, 255), (c - 3.6 * u, c - 1.05 * u, 3.6 * u, 2.1 * u))
    for k in range(3):  # karın çizgileri
        x = c - 3.0 * u + k * 0.9 * u
        pygame.draw.line(s, (40, 30, 22), (x, c - 0.8 * u), (x, c + 0.8 * u), max(1, int(u * 0.2)))
    _aa_ellipse(s, (95, 80, 55, 255), (c - 0.5 * u, c - 0.95 * u, 2.2 * u, 1.9 * u))
    _aa_circle(s, (90, 70, 50, 255), (c + 2.2 * u, c), 0.8 * u)
    # Kırmızı bileşik gözler
    for side in (-1, 1):
        _aa_circle(s, (200, 30, 30, 255), (c + 2.35 * u, c + side * 0.62 * u), 0.55 * u)
        _aa_circle(s, (255, 120, 110, 255), (c + 2.5 * u, c + side * 0.5 * u), 0.18 * u)
    return pygame.transform.smoothscale(s, (L // SS, L // SS))


def make_fruit_sprite(kind: int, radius_px: float) -> pygame.Surface:
    R = max(4, int(radius_px * SS))
    size = int(R * 2.6)
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    col = FRUIT_COLORS[kind % len(FRUIT_COLORS)]
    dark = tuple(int(v * 0.65) for v in col)
    if kind == 2:  # üzüm salkımı
        for dx, dy in ((-0.45, -0.3), (0.45, -0.3), (0, 0.15), (-0.45, 0.55), (0.45, 0.55), (0, -0.7)):
            _aa_circle(s, dark, (c + dx * R, c + dy * R), R * 0.5)
            _aa_circle(s, col, (c + dx * R - R * 0.05, c + dy * R - R * 0.05), R * 0.42)
    else:
        _aa_circle(s, dark, (c, c), R)
        _aa_circle(s, col, (c - R * 0.06, c - R * 0.06), R * 0.9)
        _aa_circle(s, (255, 255, 255, 110), (c - R * 0.35, c - R * 0.35), R * 0.25)
    # Sap ve yaprak
    pygame.draw.line(s, (90, 60, 30), (c, c - R * 0.8), (c + R * 0.15, c - R * 1.2), max(2, R // 6))
    _aa_ellipse(s, (60, 170, 60, 255), (c + R * 0.1, c - R * 1.25, R * 0.7, R * 0.35))
    return pygame.transform.smoothscale(s, (size // SS, size // SS))


def make_swatter_sprite(radius_px: float) -> pygame.Surface:
    # Delikli (ızgaralı) plastik sineklik başı + sap. Baş merkezi sprite merkezinde.
    # Köşeler çok yuvarlatılır: görünen baş, dairesel vuruş alanıyla örtüşsün
    # (ziyaretçi "altındaydım ama vurulmadım" demesin).
    R = int(radius_px * SS)
    handle = int(R * 2.2)
    size = 2 * (R + handle)
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    head = pygame.Rect(c - R, c - R, 2 * R, 2 * R)
    # Sap
    hw = max(4, R // 7)
    pygame.draw.rect(s, (40, 110, 200), (c + R - hw, c - hw, handle + hw, 2 * hw), border_radius=hw)
    pygame.draw.rect(s, (25, 70, 140), (c + R - hw, c - hw, handle + hw, 2 * hw), max(2, hw // 3), border_radius=hw)
    # Baş
    pygame.draw.rect(s, (220, 50, 60), head, border_radius=int(R * 0.7))
    grid = pygame.Surface(head.size, pygame.SRCALPHA)
    step = max(6, R // 6)
    hole = max(3, int(step * 0.55))
    for gx in range(step // 2, head.width - hole, step):
        for gy in range(step // 2, head.height - hole, step):
            if math.hypot(gx + hole / 2 - R, gy + hole / 2 - R) > R * 0.88:
                continue
            pygame.draw.rect(grid, (120, 20, 30, 255), (gx, gy, hole, hole), border_radius=hole // 3)
    s.blit(grid, head.topleft)
    pygame.draw.rect(s, (130, 20, 30), head, max(3, R // 18), border_radius=int(R * 0.7))
    size_px = max(4, size // SS)
    return pygame.transform.smoothscale(s, (size_px, size_px))


def make_table(size: int) -> pygame.Surface:
    # Piknik örtüsü (pötikare) zemin; meyveler öne çıksın diye soluk renkli
    s = pygame.Surface((size, size))
    s.fill((242, 234, 220))
    n = 16
    cell = size / n
    stripe = pygame.Surface((size, size), pygame.SRCALPHA)
    for i in range(0, n, 2):
        stripe.fill((215, 130, 115, 28), (i * cell, 0, cell, size))
        stripe.fill((215, 130, 115, 28), (0, i * cell, size, cell))
    s.blit(stripe, (0, 0))
    return s


# --- efekt durumu ------------------------------------------------------------

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple
    size: float


@dataclass
class Floater:
    text: str
    x: float
    y: float
    age: float
    color: tuple


class PanelFx:
    def __init__(self):
        self.shake = 0.0
        self.flash = 0.0
        self.flash_color = (255, 255, 255)
        self.particles: list[Particle] = []
        self.floaters: list[Floater] = []


# --- ana çizici ----------------------------------------------------------------

class Renderer:
    def __init__(self, screen: pygame.Surface, cfg: GameConfig, texts: dict, side_panel: bool = False):
        self.screen = screen
        self.cfg = cfg
        self.tx = texts
        self.side_panel = side_panel  # rakibin sağında nöron paneli için yer ayır
        self.vis_rng = random.Random()  # yalnızca görsel efektler için
        self.fx = [PanelFx(), PanelFx()]
        self._text_cache: dict = {}
        self.layout()

    # -- yerleşim ------------------------------------------------------------

    def layout(self) -> None:
        W, H = self.screen.get_size()
        self.W, self.H = W, H
        self.top = int(H * 0.13)
        self.bottom = int(H * 0.07)
        gap = int(W * 0.03)
        side_w = int(W * 0.23) if self.side_panel else 0
        usable = W - side_w - (gap if side_w else 0)
        size = int(min((usable - 3 * gap) / 2, H - self.top - self.bottom))
        x0 = (usable - 2 * size - gap) // 2
        self.panel_size = size
        self.panels = [pygame.Rect(x0, self.top, size, size), pygame.Rect(x0 + size + gap, self.top, size, size)]
        # Nöron paneli: rakip arenasının sağında, arenalarla aynı yükseklikte
        sx = self.panels[1].right + gap
        self.side_rect = pygame.Rect(sx, self.top, max(0, W - sx - gap // 2), H - self.top - gap // 2)
        self.scale = size / self.cfg.arena.width
        sc = self.scale
        self.table = make_table(size)
        self.fly_sprite = make_fly_sprite(self.cfg.fly.radius * 4.2 * sc)
        self.fruit_sprites = [make_fruit_sprite(k, self.cfg.fruit.radius * sc) for k in range(4)]
        self.swatter_sprite = make_swatter_sprite(self.cfg.swatter.radius * sc)
        self._fly_rot_cache: dict = {}
        self.fonts = {
            "huge": pygame.font.Font(None, int(H * 0.22)),
            "big": pygame.font.Font(None, int(H * 0.11)),
            "mid": pygame.font.Font(None, int(H * 0.055)),
            "small": pygame.font.Font(None, int(H * 0.034)),
            "tiny": pygame.font.Font(None, int(H * 0.026)),
        }

    def text(self, s: str, font: str, color=TEXT) -> pygame.Surface:
        key = (s, font, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 400:
                self._text_cache.clear()
            surf = self.fonts[font].render(s, True, color)
            self._text_cache[key] = surf
        return surf

    def blit_text(self, s: str, font: str, color, **pos) -> pygame.Rect:
        surf = self.text(s, font, color)
        r = surf.get_rect(**pos)
        self.screen.blit(surf, r)
        return r

    def wrap(self, s: str, font: str, width: int, sep: str = " ") -> list[str]:
        """Metni verilen piksel genişliğine sığacak satırlara böl."""
        f = self.fonts[font]
        words, lines, cur = s.split(sep), [], ""
        for w in words:
            cand = w if not cur else cur + sep + w
            if f.size(cand)[0] <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # -- olaylar ve efektler --------------------------------------------------

    def reset_fx(self) -> None:
        self.fx = [PanelFx(), PanelFx()]

    def on_events(self, idx: int, events: list[WorldEvent]) -> None:
        fx = self.fx[idx]
        rng = self.vis_rng
        shake_px = self.cfg.display.shake_px
        for e in events:
            if e.kind == "fruit":
                fx.floaters.append(Floater(f"+{e.points}", e.x, e.y, 0.0, (40, 160, 60)))
                for _ in range(14):
                    a = rng.uniform(0, 2 * math.pi)
                    v = rng.uniform(60, 220)
                    fx.particles.append(Particle(e.x, e.y, math.cos(a) * v, math.sin(a) * v, 0.5, 0.5,
                                                 (255, 230, 90), rng.uniform(2, 5)))
            elif e.kind == "slam":
                fx.shake = max(fx.shake, shake_px * (1.6 if e.hit else 1.0))
                fx.flash = 1.0
                fx.flash_color = (255, 60, 60) if e.hit else (255, 255, 255)
                if e.hit and e.points:
                    fx.floaters.append(Floater(f"{e.points}", e.x, e.y - 30, 0.0, (220, 30, 30)))
                r = self.cfg.swatter.radius
                for _ in range(40):
                    a = rng.uniform(0, 2 * math.pi)
                    v = rng.uniform(120, 420)
                    fx.particles.append(Particle(e.x + math.cos(a) * r * 0.9, e.y + math.sin(a) * r * 0.9,
                                                 math.cos(a) * v, math.sin(a) * v, 0.6, 0.6,
                                                 (160, 140, 120), rng.uniform(3, 8)))

    def update_fx(self, dt: float) -> None:
        decay = math.exp(-self.cfg.display.shake_decay * dt)
        for fx in self.fx:
            fx.shake *= decay
            fx.flash = max(0.0, fx.flash - dt * 5)
            for p in fx.particles:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vx *= 0.9
                p.vy *= 0.9
                p.life -= dt
            fx.particles = [p for p in fx.particles if p.life > 0]
            for f in fx.floaters:
                f.age += dt
                f.y -= 60 * dt
            fx.floaters = [f for f in fx.floaters if f.age < 1.0]

    # -- panel çizimi ------------------------------------------------------------

    def _fly_rotated(self, heading: float) -> pygame.Surface:
        key = int(round(math.degrees(heading))) % 360
        s = self._fly_rot_cache.get(key)
        if s is None:
            s = pygame.transform.rotate(self.fly_sprite, -key)
            self._fly_rot_cache[key] = s
        return s

    def draw_panel(self, idx: int, world: World, color: tuple, now: float) -> None:
        rect = self.panels[idx]
        fx = self.fx[idx]
        sc = self.scale
        surf = pygame.Surface(rect.size)
        surf.blit(self.table, (0, 0))
        t = world.t

        # Meyveler (çıkarken büyüyerek belirir, bitmeden önce yanıp söner)
        for f in world.fruits:
            age = t - f.spawned_at
            left = self.cfg.fruit.lifetime_s - age
            if left < 1.5 and int(now * 8) % 2 == 0:
                continue
            spr = self.fruit_sprites[f.kind]
            grow = min(1.0, age / 0.2) if f.spawned_at > 0 else 1.0
            if grow < 1.0:
                spr = pygame.transform.smoothscale(
                    spr, (max(1, int(spr.get_width() * grow)), max(1, int(spr.get_height() * grow))))
            surf.blit(spr, spr.get_rect(center=(f.x * sc, f.y * sc)))

        # Sineklik gölgeleri (yaklaşma/looming)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        drop = self.cfg.swatter.drop_height
        for s in world.swatters:
            if s.impacted:
                continue
            p = s.progress(t)
            cx, cy = s.x * sc, s.y * sc
            R = s.radius * sc
            # Gölge: yükseklik azaldıkça büyür ve koyulaşır
            h = swatter_height(p, drop) / drop  # 1 = yüksekte, 0 = yerde
            shadow_r = R * (0.25 + 0.75 * (1 - h))
            for k, frac in enumerate((1.25, 1.1, 1.0)):
                alpha = int((40 + 150 * (1 - h)) * (0.35 + 0.3 * k))
                pygame.draw.circle(overlay, (20, 10, 10, min(255, alpha)), (cx, cy), shadow_r * frac)
            # Uyarı halkası: tehlike bölgesi, çarpmaya yaklaştıkça hızlı yanıp söner
            pulse = 0.5 + 0.5 * math.sin(now * (8 + 30 * p))
            ring_a = int(60 + 150 * p * pulse)
            pygame.draw.circle(overlay, (220, 30, 30, ring_a), (cx, cy), R, max(2, int(3 * sc)))
        surf.blit(overlay, (0, 0))

        # Sinek
        fly = world.fly
        fs = self._fly_rotated(fly.heading)
        wob = 0.0
        if world.stunned:
            wob = math.sin(now * 40) * 3
        pygame.draw.circle(surf, color, (fly.x * sc, fly.y * sc), self.cfg.fly.radius * sc * 1.6, max(2, int(3 * sc)))
        surf.blit(fs, fs.get_rect(center=(fly.x * sc + wob, fly.y * sc)))
        if world.stunned:
            for k in range(4):
                a = now * 6 + k * math.pi / 2
                sx = fly.x * sc + math.cos(a) * 22 * sc
                sy = fly.y * sc - 24 * sc + math.sin(a) * 8 * sc
                _aa_circle(surf, (255, 220, 40), (sx, sy), max(2, 4 * sc))

        # Sineklik: son anda yukarıdan iner, çarpar, sonra kaybolur
        for s in world.swatters:
            p = s.progress(t)
            after = t - (s.started_at + s.loom_s)
            if not s.impacted and p < 0.85:
                continue
            if s.impacted:
                zoom = 1.0
                alpha = 255 if after < self.cfg.swatter.impact_show_s * 0.6 else 140
            else:
                q = (p - 0.85) / 0.15  # 0..1 inerken
                zoom = 1.0 + (1 - q) * 0.9
                alpha = int(90 + 165 * q)
            ang = (s.id * 67) % 360  # yalnızca görsel: sapın yönü
            spr = pygame.transform.rotozoom(self.swatter_sprite, ang, zoom)
            spr.set_alpha(alpha)
            surf.blit(spr, spr.get_rect(center=(s.x * sc, s.y * sc)))

        # Parçacıklar ve uçan puan yazıları
        for pt in fx.particles:
            a = pt.life / pt.max_life
            _aa_circle(surf, (*pt.color, int(255 * a)), (pt.x * sc, pt.y * sc), max(1, pt.size * sc * a))
        for fl in fx.floaters:
            ts = self.text(fl.text, "mid", fl.color)
            ts.set_alpha(int(255 * (1 - fl.age)))
            surf.blit(ts, ts.get_rect(center=(fl.x * sc, fl.y * sc)))
            ts.set_alpha(255)

        # Çarpma flaşı
        if fx.flash > 0:
            fl = pygame.Surface(rect.size, pygame.SRCALPHA)
            fl.fill((*fx.flash_color, int(150 * fx.flash)))
            surf.blit(fl, (0, 0))

        # Sarsıntıyla birlikte ekrana yerleştir
        ox = oy = 0
        if fx.shake > 0.5:
            ox = self.vis_rng.uniform(-fx.shake, fx.shake)
            oy = self.vis_rng.uniform(-fx.shake, fx.shake)
        self.screen.blit(surf, (rect.x + ox, rect.y + oy))
        pygame.draw.rect(self.screen, color, rect.move(ox, oy).inflate(6, 6), 3, border_radius=4)

    # -- başlık/alt bilgi ---------------------------------------------------------

    def draw_header(self, idx: int, name: str, score: int, color: tuple, world: World | None = None) -> None:
        # Skorlar dış kenarlarda, ortadaki sayaçla çakışmasın diye
        rect = self.panels[idx]
        y = self.top - int(self.H * 0.012)
        gap = int(self.W * 0.012)
        score_s = self.text(str(score), "big", TEXT)
        name_s = self.text(name, "mid", color)
        if idx == 0:
            sr = score_s.get_rect(bottomleft=(rect.x, y + int(self.H * 0.012)))
            nr = name_s.get_rect(bottomleft=(sr.right + gap, y))
        else:
            sr = score_s.get_rect(bottomright=(rect.right, y + int(self.H * 0.012)))
            nr = name_s.get_rect(bottomright=(sr.left - gap, y))
        self.screen.blit(score_s, sr)
        self.screen.blit(name_s, nr)
        if world is not None and world.stunned:
            self.blit_text(self.tx["stunned"], "small", (255, 90, 90), midbottom=(rect.centerx, y))

    def draw_badge(self, idx: int, label: str, kind: AgentKind, note: str | None = None) -> None:
        rect = self.panels[idx]
        col = KIND_BADGE.get(kind, DIM)
        surf = self.text(label, "small", (15, 15, 20))
        pad = 8
        box = surf.get_rect(midtop=(rect.centerx, rect.bottom + int(self.H * 0.012))).inflate(2 * pad, pad)
        pygame.draw.rect(self.screen, col, box, border_radius=6)
        self.screen.blit(surf, surf.get_rect(center=box.center))
        if note:
            # Uzun açıklama " · " noktalarından satırlara bölünür (arena genişliğini aşmasın)
            y = box.bottom + 3
            for line in self.wrap(note, "tiny", int(rect.width * 1.05), sep=" · "):
                r = self.blit_text(line, "tiny", DIM, midtop=(rect.centerx, y))
                y = r.bottom

    def draw_timer(self, time_left: float) -> None:
        secs = int(math.ceil(time_left))
        col = (255, 80, 80) if secs <= 10 else TEXT
        cx = (self.panels[0].right + self.panels[1].left) // 2  # iki arenanın ortası
        self.blit_text(f"{secs}", "big", col, midtop=(cx, int(self.H * 0.01)))

    # -- tam ekran katmanlar --------------------------------------------------------

    def dim(self, alpha: int = 170) -> None:
        d = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        d.fill((10, 10, 16, alpha))
        self.screen.blit(d, (0, 0))

    def draw_center_text(self, s: str, font: str = "huge", color=TEXT, dy: float = 0.0) -> None:
        # Gölgeli büyük yazı
        shadow = self.text(s, font, (0, 0, 0))
        pos = (self.W // 2, int(self.H * (0.5 + dy)))
        self.screen.blit(shadow, shadow.get_rect(center=(pos[0] + 4, pos[1] + 4)))
        self.blit_text(s, font, color, center=pos)
