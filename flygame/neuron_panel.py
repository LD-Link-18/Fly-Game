"""Beyin haritası paneli: rakibin yanında, sinek beyninin o anda ne yaptığını gösterir.

- Harita: modeldeki 138.639 nöronun her biri, FlyWire'daki gerçek konumunda bir noktadır
  (bkz. brain/brainmap.py). Tüm nöronların yoğunluğu soluk bir "dijital" (LED ızgarası)
  beyin görüntüsü oluşturur; bir nöron ateşlediğinde bulunduğu yer parlar ve söner.
  İki izdüşüm: arkadan (sineğin solu ekranın solunda) ve üstten (baş yukarıda, oyundaki
  sinek gibi). Aynı spike'lar iki görünümde birden yanar.
- Renkler: girdi verilen göz nöronları (meyve / sineklik), kararın okunduğu inen nöronlar
  ve diğer tüm nöronlar. Hangi nöronların hangi rol olduğu ajanın panel_info()'sundadır.
- Haritanın altında "bacaklara" çıktısı: beynin dönüş komutu ve dev lif kaçış sinyali.
- Başlık: CANLI simülasyon mu, KAYIT mı; tüm beyindeki saniyedeki spike sayısı.

Veri, ajanın telemetry() çıktısıdır ("fired": o adımda ateşleyen tüm nöronlar); kayıttan
tekrarda aynı veri kayıttan gelir. Beyni olmayan rakipler için "beyin yok" kartı gösterilir.
Gösterilen her parıltı simülasyonda gerçekten ateşlemiş bir nörondur; süsleme amaçlı
rastgele ışık yoktur.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from .agents.base import Agent, AgentKind
from .texts import fmt_int

BG = (24, 27, 38)
MAP_BG = (8, 10, 16)
LINE = (60, 65, 85)
TEXT = (235, 235, 240)
DIM = (150, 155, 170)
LIVE_COL = (90, 220, 120)
REC_COL = (240, 200, 60)
ESC_COL = (255, 80, 200)

# Harita renkleri: soluk taban (nöron yoğunluğu) ve rol başına ateşleme rengi
BASE_LO = np.array([12, 18, 32], np.float32)
BASE_HI = np.array([40, 64, 108], np.float32)
EDGE = np.array([70, 110, 170], np.float32)   # beynin dış çizgisi
ROLE_COLORS = {
    "other": (120, 200, 255),
    "fruit": (90, 235, 120),
    "loom": (255, 90, 80),
    "decision": (255, 170, 50),
}
ROLE_ORDER = ("fruit", "loom", "decision", "other")  # açıklama sırası
GLOW_TAU_S = 0.06       # parıltının sönme süresi (saniye)
SPIKE_GAIN = 150.0      # tek spike'ın bir hücreye kattığı parlaklık (0-255 ölçeğinde)
PING_S = 0.4            # karar nöronu ateşleyince çevresinde büyüyen halkanın ömrü
MAX_PINGS = 60

_POSITIONS: dict[str, np.ndarray | None] = {}


def load_positions_cached(data_dir: str) -> np.ndarray | None:
    """Nöron konumları (µm); veri yoksa None. Süreç başına bir kez okunur."""
    if data_dir not in _POSITIONS:
        from .brain.brainmap import load_positions

        try:
            _POSITIONS[data_dir] = load_positions(data_dir)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Brain map unavailable: {e}")
            _POSITIONS[data_dir] = None
    return _POSITIONS[data_dir]


class BrainView:
    """Bir izdüşüm (arkadan veya üstten): taban görüntüsü + ateşleme parıltısı.

    Görüntü hücrelerden oluşur (cell x cell piksel, aralarında ince boşluk: LED ızgarası).
    Her nöron bir hücreye düşer; ateşleyen nöronların rengi hücresinin ısısına eklenir,
    ısı zamanla söner.
    """

    def __init__(self, pos: np.ndarray, axes: tuple[int, int], lo: np.ndarray, scale: float,
                 size: tuple[int, int], cell: int):
        self.W, self.H = size
        self.cell = cell
        gw, gh = max(1, self.W // cell), max(1, self.H // cell)
        self.gw, self.gh = gw, gh
        u = (pos[:, axes[0]] - lo[axes[0]]) * scale / cell
        v = (pos[:, axes[1]] - lo[axes[1]]) * scale / cell
        ok = np.isfinite(u) & np.isfinite(v)
        gx = np.where(ok, u, -1).astype(np.int32)
        gy = np.where(ok, v, -1).astype(np.int32)
        ok &= (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        self.valid = ok
        self.gx, self.gy = np.where(ok, gx, 0), np.where(ok, gy, 0)

        # Taban: hücre başına nöron sayısı, hafif bulanıklaştırılmış, karekök ölçeğinde
        dens = np.zeros((gw, gh), np.float32)
        np.add.at(dens, (self.gx[ok], self.gy[ok]), 1.0)
        p = np.pad(dens, 1)
        blur = sum(p[1 + dx:1 + dx + gw, 1 + dy:1 + dy + gh] for dx in (-1, 0, 1) for dy in (-1, 0, 1)) / 9.0
        mix = 0.6 * dens + 0.4 * blur
        top = np.percentile(mix[mix > 0], 99.5) if (mix > 0).any() else 1.0
        val = np.clip(mix / top, 0.0, 1.0)[..., None] ** 0.8
        inside = blur > 0.25
        base = np.where(inside[..., None], BASE_LO + val * (BASE_HI - BASE_LO), np.array(MAP_BG, np.float32))
        # Dış çizgi: içeride olup bir komşusu dışarıda kalan hücreler
        q = np.pad(inside, 1)
        edge = inside & ~(q[:-2, 1:-1] & q[2:, 1:-1] & q[1:-1, :-2] & q[1:-1, 2:])
        base[edge] = EDGE
        self.base = base.astype(np.float32)
        self.heat = np.zeros((gw, gh, 3), np.float32)
        self.hot = False       # sönmemiş parıltı var mı
        self.dirty = True      # görüntü yeniden çizilmeli mi
        self.surf: pygame.Surface | None = None

        # LED ızgarası: hücreler arası boşlukları karartan çarpım maskesi
        cw, ch = gw * cell, gh * cell
        self.px_size = (cw, ch)
        mask = pygame.Surface((cw, ch))
        mask.fill((255, 255, 255))
        if cell >= 3:
            dark = (95, 95, 105)
            for i in range(gw + 1):
                mask.fill(dark, (i * cell - 1, 0, 1, ch))
            for j in range(gh + 1):
                mask.fill(dark, (0, j * cell - 1, cw, 1))
        self.mask = mask

    def cell_center(self, i: int) -> tuple[int, int] | None:
        if not self.valid[i]:
            return None
        c = self.cell
        return int(self.gx[i]) * c + c // 2, int(self.gy[i]) * c + c // 2

    def add_spikes(self, idx: np.ndarray, colors: np.ndarray) -> None:
        idx = idx[self.valid[idx]]
        if len(idx):
            np.add.at(self.heat, (self.gx[idx], self.gy[idx]), colors[idx])
            self.hot = self.dirty = True

    def fade(self, factor: float) -> None:
        if self.hot:
            self.heat *= factor
            self.dirty = True
            if self.heat.max() < 0.5:   # tamamen söndü: artık yeniden çizmeye gerek yok
                self.heat[:] = 0.0
                self.hot = False

    def redraw(self) -> None:
        self.surf = self.render()
        self.dirty = False

    def render(self) -> pygame.Surface:
        # Isıyı doyurarak ekle; renk tonu korunur (üst üste binen spike'lar beyaza dönmez,
        # renk açıklamadaki gibi kalır)
        peak = self.heat.max(axis=2, keepdims=True)
        glow = self.heat * (255.0 * (1.0 - np.exp(-peak / 255.0)) / np.maximum(peak, 1e-6))
        img = np.clip(self.base + glow, 0, 255).astype(np.uint8)
        small = pygame.surfarray.make_surface(img)
        big = pygame.transform.scale(small, self.px_size)
        big.blit(self.mask, (0, 0), special_flags=pygame.BLEND_MULT)
        # Hale: parıltının yumuşatılmış hali, ızgaranın üstüne toplanır
        halo = pygame.surfarray.make_surface((glow * 0.55).astype(np.uint8))
        halo = pygame.transform.smoothscale(halo, (max(1, self.gw // 2), max(1, self.gh // 2)))
        big.blit(pygame.transform.smoothscale(halo, self.px_size), (0, 0), special_flags=pygame.BLEND_ADD)
        return big


class NeuronPanel:
    def __init__(self, renderer):
        self.r = renderer
        self.info = None
        self.agent: Agent | None = None
        self.views: list = []
        self.escape_flash = 0.0  # reset() öncesi (başlık ekranında) update() çağrılabilir
        self.frame = 0

    # -- kurulum ----------------------------------------------------------------------

    def reset(self, agent: Agent, tx: dict, lang: str) -> None:
        self.agent, self.tx, self.lang = agent, tx, lang
        self.info = agent.panel_info()
        self.spikes_per_s = 0.0
        self.active = 0.0
        self.turn = 0.0
        self.escape_flash = 0.0
        self.got_data = False
        self.pings: dict[int, float] = {}  # nöron -> yaş (s): ateşleyen karar nöronları
        self.pos = None
        self.map_error = None
        mp = (self.info or {}).get("map")
        if self.info and not mp:
            self.map_error = "old"             # eski kayıt: harita verisi yok
        elif mp:
            self.pos = load_positions_cached(self.r.cfg.brain.data_dir)
            if self.pos is None:
                self.map_error = "missing"     # FlyWire verisi indirilmemiş
            else:
                self.colors = np.tile(np.array(ROLE_COLORS["other"], np.float32) / 255.0 * SPIKE_GAIN,
                                      (len(self.pos), 1))
                self.is_decision = np.zeros(len(self.pos), bool)
                for role in ("fruit", "loom", "decision"):
                    idx = np.asarray(mp.get(role, []), dtype=np.int64)
                    idx = idx[(idx >= 0) & (idx < len(self.pos))]
                    self.colors[idx] = np.array(ROLE_COLORS[role], np.float32) / 255.0 * SPIKE_GAIN
                    if role == "decision":
                        self.is_decision[idx] = True
        self._layout()

    def _layout(self) -> None:
        rect = self.r.side_rect
        H = self.r.H
        self.pad = max(6, int(rect.width * 0.04))
        self.line_h = int(H * 0.028)
        # Başlık: ad (0.05 H) + ağ boyutu + spike sayısı + simülasyon hızı satırları
        self.header_h = self.pad + int(H * 0.05) + self.line_h + 2 * int(H * 0.026) + int(H * 0.012)
        self.views = []
        if self.pos is None:
            return
        pos = self.pos
        lo = np.nanpercentile(pos, 0.05, axis=0)
        hi = np.nanpercentile(pos, 99.95, axis=0)
        margin = (hi - lo) * 0.02
        lo, hi = lo - margin, hi + margin
        xr, yr, zr = hi - lo
        # Dikey yerleşim: başlık, harita başlığı, [etiket + arkadan görünüm], bacaklara
        # çıktısı, [etiket + üstten görünüm], açıklama (panelin altında). Haritalar genişliğe
        # sığacak kadar büyük; yükseklik yetmezse küçülür, çok küçülecekse üstten görünüm
        # çıkarılır. Artan yükseklik bölümler arasına eşit dağıtılır.
        self.stub_h = int(H * 0.11)
        self.legend_h = 3 * self.line_h + int(H * 0.008)
        title_h = 2 * self.line_h
        fixed = self.header_h + title_h + self.line_h + self.stub_h + self.legend_h + self.pad
        width = rect.width - 2 * self.pad
        avail = rect.height - fixed
        scale = min(width / xr, max(1.0, avail - self.line_h) / (yr + zr))
        with_top = zr * scale >= H * 0.07
        if with_top:
            fixed += self.line_h
        else:
            scale = min(width / xr, max(1.0, avail) / yr)
        cell = max(2, round(H / 360))
        self.cell = cell
        wpx = int(xr * scale)
        x0 = rect.x + (rect.width - wpx) // 2
        used = fixed + int(yr * scale) + (int(zr * scale) if with_top else 0)
        gap = max(0, rect.height - used) // (3 if with_top else 2)

        y = rect.y + self.header_h
        self.title_top = y
        y += title_h + gap + self.line_h
        back = BrainView(pos, (0, 1), lo, scale, (wpx, int(yr * scale)), cell)
        self.views.append(("map_back", back, pygame.Rect((x0, y), back.px_size)))
        y += back.px_size[1]
        self.stub_top = y
        y += self.stub_h
        if with_top:
            y += gap + self.line_h
            top = BrainView(pos, (0, 2), lo, scale, (wpx, int(zr * scale)), cell)
            self.views.append(("map_top", top, pygame.Rect((x0, y), top.px_size)))
        self.legend_top = rect.bottom - self.pad - self.legend_h
        # Boyun: arkadan görünümde orta hattaki (±40 µm) nöronların en alt noktası
        mid = (np.abs(pos[:, 0] - (lo[0] + xr / 2)) < 40) & np.isfinite(pos[:, 1])
        neck_y = np.nanpercentile(pos[mid, 1], 99.5) if mid.any() else hi[1]
        vr = self.views[0][2]
        self.neck = (vr.x + int(xr / 2 * scale), min(vr.bottom, vr.y + int((neck_y - lo[1]) * scale)))
        self.pings = {}

    def relayout(self) -> None:
        """Ekran boyutu değişince (F11) haritayı yeniden kur; parıltı sıfırlanır."""
        if self.agent is not None:
            self._layout()

    # -- veri -------------------------------------------------------------------------

    def push(self, tel: dict | None) -> None:
        """Her oyun adımında ajanın telemetrisiyle çağrılır."""
        if not tel:
            return
        self.got_data = True
        ms = tel.get("ms", 0.0)
        if ms:
            sps = tel.get("spikes", 0) / (ms / 1000.0)
            self.spikes_per_s += 0.1 * (sps - self.spikes_per_s)
        self.turn = tel.get("turn", 0.0)
        if tel.get("escape"):
            self.escape_flash = 1.0
        fired = tel.get("fired")
        if fired is not None and self.views and ms:
            idx = np.asarray(fired, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(self.pos))]
            self.active += 0.1 * (len(idx) - self.active)
            for _, view, _ in self.views:
                view.add_spikes(idx, self.colors)
            for i in idx[self.is_decision[idx]].tolist():
                # Halka bitmeden yeniden ateşlerse, halka yarıdan sonra baştan başlar
                if self.pings.get(i, PING_S) > PING_S / 2 and len(self.pings) < MAX_PINGS:
                    self.pings[i] = 0.0

    def update(self, dt: float) -> None:
        self.escape_flash = max(0.0, self.escape_flash - dt * 2.5)
        if self.views:
            f = math.exp(-dt / GLOW_TAU_S)
            for _, view, _ in self.views:
                view.fade(f)
            self.pings = {i: a + dt for i, a in self.pings.items() if a + dt < PING_S}

    # -- çizim ------------------------------------------------------------------------

    def _text(self, s, font, color, **pos):
        return self.r.blit_text(s, font, color, **pos)

    def _card(self, x0, y, title, lines) -> None:
        rect = self.r.side_rect
        self._text(title, "big", DIM, topleft=(x0, y))
        y += int(self.r.H * 0.12)
        for line in lines:
            for part in self.r.wrap(line, "small", rect.width - 2 * self.pad):
                self._text(part, "small", DIM, topleft=(x0, y))
                y += int(self.r.H * 0.04)

    def draw(self) -> None:
        if self.agent is None:
            return
        scr = self.r.screen
        rect = self.r.side_rect
        pygame.draw.rect(scr, BG, rect, border_radius=10)
        pygame.draw.rect(scr, LINE, rect, 2, border_radius=10)
        tx = self.tx
        x0 = rect.x + self.pad
        name = self.agent.display_name(self.lang)

        if not self.info:
            # Beyni olmayan rakip (veya nöron verisi olmayan kayıt): dürüst açıklama
            self._text(name, "mid", TEXT, topleft=(x0, rect.y + self.pad))
            y = rect.y + int(self.r.H * 0.12)
            if self.agent.kind == AgentKind.REPLAY:
                self._card(x0, y, tx["panel_recorded"], [tx["panel_no_data"]])
            else:
                self._card(x0, y, tx["panel_bot_title"], [tx["panel_bot_1"], tx["panel_bot_2"]])
            return

        self._header(x0)
        if self.map_error:
            msg = tx["map_old"] if self.map_error == "old" else tx["map_missing"]
            y = rect.y + self.header_h + int(self.r.H * 0.03)
            for part in self.r.wrap(msg, "small", rect.width - 2 * self.pad):
                self._text(part, "small", DIM, topleft=(x0, y))
                y += int(self.r.H * 0.04)
            return

        # Harita başlığı
        y = self.title_top
        self._text(tx["map_title"], "tiny", TEXT, topleft=(x0, y))
        self._text(tx["map_sub"], "tiny", DIM, topleft=(x0, y + self.line_h))

        self._stub()
        # Her karede en çok bir görünüm yeniden çizilir (sırayla; parıltı ~60 ms'de
        # söndüğü için 30 Hz yeterli): panelin kare başına maliyeti yarıya iner
        self.frame += 1
        for k, (key, view, vr) in enumerate(self.views):
            if view.surf is None or (view.dirty and self.frame % len(self.views) == k):
                view.redraw()
            scr.blit(view.surf, vr)
            lab = self.r.text(tx[key], "tiny", DIM)
            scr.blit(lab, lab.get_rect(bottomleft=(vr.x, vr.y - 2)))
            if key == "map_top":
                self._head_marker(vr, lab)
            self._draw_pings(view, vr)
            self._tag(tx["map_l"], bottomleft=(vr.x, vr.bottom))
            self._tag(tx["map_r"], bottomright=(vr.right, vr.bottom))
            if key == "map_back":
                # Kaynak: sığarsa görünüm etiketiyle aynı satırda, sağda
                src = self.r.text(tx["map_source"], "tiny", DIM)
                if lab.get_width() + src.get_width() + 12 <= vr.width:
                    scr.blit(src, src.get_rect(bottomright=(vr.right, vr.y - 2)))
        self._legend(x0)

    def _tag(self, s: str, **pos) -> None:
        # Harita köşesinde okunur kalsın diye yarı saydam koyu zemin üzerinde yazı
        lab = self.r.text(s, "tiny", DIM)
        bg = pygame.Surface((lab.get_width() + 8, lab.get_height() + 2), pygame.SRCALPHA)
        bg.fill((*MAP_BG, 200))
        r = bg.get_rect(**pos)
        self.r.screen.blit(bg, r)
        self.r.screen.blit(lab, lab.get_rect(center=r.center))

    def _draw_pings(self, view: "BrainView", vr: pygame.Rect) -> None:
        # Karar nöronu ateşledi: konumunda büyüyüp sönen turuncu halka
        col = np.array(ROLE_COLORS["decision"], float)
        for i, age in self.pings.items():
            c = view.cell_center(i)
            if c is None:
                continue
            k = age / PING_S
            rgb = (col * (1.0 - k) + np.array(MAP_BG) * k).astype(int)
            pygame.draw.circle(self.r.screen, tuple(rgb), (vr.x + c[0], vr.y + c[1]),
                               int(3 + k * self.cell * 5), 2)

    def _header(self, x0) -> None:
        """Ad, CANLI/KAYIT rozeti, ağ boyutu, spike sayısı."""
        scr = self.r.screen
        rect = self.r.side_rect
        tx = self.tx
        name = self.agent.display_name(self.lang)
        y = rect.y + self.pad
        live = self.info.get("live", False)
        tag = tx["panel_live"] if live else tx["panel_recorded"]
        tag_s = self.r.text(tag, "small", (15, 15, 20))
        box = tag_s.get_rect(topright=(rect.right - self.pad, y + 4)).inflate(12, 6)
        pygame.draw.rect(scr, LIVE_COL if live else REC_COL, box, border_radius=5)
        scr.blit(tag_s, tag_s.get_rect(center=box.center))
        # Kayıtta rozet zaten "KAYIT" diyor: başlıkta kaynağın adı yeterli; sığmazsa küçült
        src = getattr(self.agent, "source_kind", None)
        if src:
            from .agents.base import DISPLAY_NAMES
            try:
                name = DISPLAY_NAMES.get(self.lang, DISPLAY_NAMES["en"])[AgentKind(src)]
            except ValueError:
                pass
        room = box.left - x0 - self.pad
        font = "mid" if self.r.fonts["mid"].size(name)[0] <= room else "small"
        self._text(name, font, TEXT, topleft=(x0, y))
        y += int(self.r.H * 0.05)
        m = self.info.get("connections", 0) / 1e6
        net = tx["panel_net"].format(n=fmt_int(self.info.get("neurons", 0), self.lang),
                                     m=f"{m:.1f}".replace(".", ",") if self.lang == "tr" else f"{m:.1f}")
        self._text(net, "tiny", DIM, topleft=(x0, y))
        y += int(self.r.H * 0.028)
        if self.got_data and self.spikes_per_s > 0:
            self._text(tx["panel_spikes"].format(n=fmt_int(self.spikes_per_s, self.lang)), "tiny", TEXT,
                       topleft=(x0, y))
        if live and hasattr(self.agent, "realtime_factor") and self.got_data:
            rt = self.agent.realtime_factor
            if rt != float("inf"):
                self._text(tx["panel_rt"].format(x=rt), "tiny", DIM, topleft=(x0, y + int(self.r.H * 0.026)))

    def _stub(self) -> None:
        """Boyundan "bacaklara" çıkış: beynin dönüş komutu ve kaçış sinyali."""
        scr = self.r.screen
        tx = self.tx
        H = self.r.H
        nx, ny = self.neck
        col = ROLE_COLORS["decision"]
        lab = self.r.text(tx["map_legs"], "tiny", col)
        lab_top = self.stub_top + int(H * 0.018)
        pygame.draw.line(scr, col, (nx, ny), (nx, lab_top), 2)
        scr.blit(lab, lab.get_rect(midtop=(nx, lab_top)))
        # Dönüş oku: uzunluk ve yön beynin komutu; altında sözcükle
        ay = lab_top + self.line_h + int(H * 0.018)
        half = int(self.r.side_rect.width * 0.3)
        pygame.draw.line(scr, LINE, (nx - half, ay), (nx + half, ay), 1)
        pygame.draw.line(scr, TEXT, (nx, ay - 6), (nx, ay + 6), 1)
        L = int(max(-1.0, min(1.0, self.turn)) * (half - 10))
        if abs(L) > 2:
            d = 1 if L > 0 else -1
            tip = nx + L
            pygame.draw.line(scr, (120, 200, 255), (nx, ay), (tip, ay), 4)
            pygame.draw.polygon(scr, (120, 200, 255), [(tip + d * 10, ay), (tip, ay - 7), (tip, ay + 7)])
        if self.escape_flash > 0:
            esc = self.r.text(tx["panel_escape"], "small", ESC_COL)
            esc.set_alpha(int(255 * min(1.0, self.escape_flash)))
            scr.blit(esc, esc.get_rect(midtop=(nx, ay + 9)))
            esc.set_alpha(255)
        else:
            word = tx["map_turn_l"] if L < -2 else tx["map_turn_r"] if L > 2 else tx["map_straight"]
            self._text(word, "tiny", DIM, midtop=(nx, ay + 10))

    def _head_marker(self, vr: pygame.Rect, lab: pygame.Surface) -> None:
        # Üstten görünümde başın yönü: küçük yukarı ok (yazı tipi ok karakterini çizemiyor)
        x = vr.x + lab.get_width() + 12
        y = vr.y - 2 - lab.get_height() // 2
        pygame.draw.polygon(self.r.screen, DIM, [(x, y - 6), (x - 5, y + 4), (x + 5, y + 4)])

    def _legend(self, x0) -> None:
        tx = self.tx
        y = self.legend_top
        w = self.r.side_rect.width - 2 * self.pad
        col_w = w // 2
        keys = {"fruit": "map_fruit", "loom": "map_loom", "decision": "map_decision", "other": "map_other"}
        for i, role in enumerate(ROLE_ORDER):
            cx = x0 + (i % 2) * col_w
            cy = y + (i // 2) * self.line_h + self.line_h // 2
            pygame.draw.circle(self.r.screen, ROLE_COLORS[role], (cx + 5, cy), 5)
            s = tx[keys[role]]
            font = "tiny"
            self.r.screen.blit(self.r.text(s, font, DIM), (cx + 14, cy - self.r.fonts[font].get_height() // 2),
                               area=pygame.Rect(0, 0, col_w - 18, 100))
        y += 2 * self.line_h + int(self.r.H * 0.006)
        if self.got_data:
            self._text(tx["map_active"].format(k=fmt_int(self.active, self.lang)), "tiny", TEXT, topleft=(x0, y))
