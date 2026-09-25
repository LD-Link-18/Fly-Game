"""Nöron etkinliği paneli: rakibin yanında, beynin o anda ne yaptığını gösterir.

- Kayan spike raster'ı: gösterilen her nöron bir satır, her oyun adımı bir sütun;
  nöron o adımda en az bir kez ateşlediyse nokta yanar. Satırlar gruplara ayrılır
  (göz nöronları, merkez beyin, inen nöronlar); hangi nöronların seçildiği
  FlyBrainAgent._build_panel_sample içindedir.
- Çubuklar: gözlere verilen girdi (meyve / sineklik, sol-sağ) ve beynin çıktısı
  (yönlendirme nöronları, dönüş, hız, dev lif kaçış sinyali).
- Başlık: CANLI simülasyon mu, KAYIT mı; tüm beyindeki saniyedeki spike sayısı.

Veri, ajanın telemetry() çıktısıdır; kayıttan tekrarda aynı veri kayıttan gelir.
Beyni olmayan rakipler için "beyin yok" kartı gösterilir.
"""

from __future__ import annotations

import pygame

from .agents.base import Agent, AgentKind
from .texts import fmt_int

BG = (24, 27, 38)
RASTER_BG = (10, 12, 18)
LINE = (60, 65, 85)
TEXT = (235, 235, 240)
DIM = (150, 155, 170)
GREEN = (90, 220, 120)
RED = (255, 110, 90)
ORANGE = (255, 160, 60)
LIVE_COL = (90, 220, 120)
REC_COL = (240, 200, 60)


class NeuronPanel:
    def __init__(self, renderer):
        self.r = renderer
        self.info = None
        self.agent: Agent | None = None
        self.surf = None
        self.escape_flash = 0.0  # reset() öncesi (başlık ekranında) update() çağrılabilir

    # -- kurulum ----------------------------------------------------------------------

    def reset(self, agent: Agent, tx: dict, lang: str) -> None:
        self.agent, self.tx, self.lang = agent, tx, lang
        self.info = agent.panel_info()
        self.rin: dict = {}
        self.rout: dict = {}
        self.spikes_per_s = 0.0
        self.turn = 0.0
        self.fwd = 0.0
        self.escape_flash = 0.0
        self.got_data = False
        self._layout()

    def _layout(self) -> None:
        rect = self.r.side_rect
        H = self.r.H
        self.pad = max(6, int(rect.width * 0.04))
        self.header_h = int(H * 0.115)
        self.bars_h = int(H * 0.25)
        self.surf = None
        if not self.info:
            return
        groups = self.info["groups"]
        rows = sum(g["n"] for g in groups)
        avail = rect.height - self.header_h - self.bars_h - self.pad
        self.row_h = max(1, min(3, avail // max(1, rows)))
        self.raster_rect = pygame.Rect(rect.x + self.pad, rect.y + self.header_h,
                                       rect.width - 2 * self.pad, rows * self.row_h)
        self.surf = pygame.Surface(self.raster_rect.size)
        self.surf.fill(RASTER_BG)
        self.row_color = []
        self.bands = []  # (ilk satır, satır sayısı, etiket, renk)
        start = 0
        for g in groups:
            col = tuple(g["color"])
            self.row_color += [col] * g["n"]
            self.bands.append((start, g["n"], g.get(self.lang, g["en"]), col))
            start += g["n"]

    def relayout(self) -> None:
        """Ekran boyutu değişince (F11) raster'ı yeniden kur; geçmiş silinir."""
        if self.agent is not None:
            self._layout()

    # -- veri -------------------------------------------------------------------------

    def push(self, tel: dict | None) -> None:
        """Her oyun adımında ajanın telemetrisiyle çağrılır."""
        if not tel:
            return
        self.got_data = True
        self.rin = tel.get("in", {})
        self.rout = tel.get("out", {})
        ms = tel.get("ms", 0.0)
        if ms:
            sps = tel.get("spikes", 0) / (ms / 1000.0)
            self.spikes_per_s += 0.1 * (sps - self.spikes_per_s)
        self.turn = tel.get("turn", 0.0)
        self.fwd = tel.get("fwd", 0.0)
        if tel.get("escape"):
            self.escape_flash = 1.0
        if self.surf is not None and "raster" in tel:
            w, h = self.surf.get_size()
            self.surf.scroll(-1, 0)
            self.surf.fill(RASTER_BG, (w - 1, 0, 1, h))
            for row in tel["raster"]:
                if row < len(self.row_color):
                    self.surf.fill(self.row_color[row], (w - 1, row * self.row_h, 1, self.row_h))

    def update(self, dt: float) -> None:
        self.escape_flash = max(0.0, self.escape_flash - dt * 2.5)

    # -- çizim ------------------------------------------------------------------------

    def _text(self, s, font, color, **pos):
        return self.r.blit_text(s, font, color, **pos)

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
            # Beyni olmayan rakip (veya nöron verisi olmayan eski kayıt): dürüst açıklama
            self._text(name, "mid", TEXT, topleft=(x0, rect.y + self.pad))
            if self.agent.kind == AgentKind.REPLAY:
                lines = [tx["panel_no_data"]]
                title = tx["panel_recorded"]
            else:
                lines = [tx["panel_bot_1"], tx["panel_bot_2"]]
                title = tx["panel_bot_title"]
            y = rect.y + int(self.r.H * 0.12)
            self._text(title, "big", DIM, topleft=(x0, y))
            y += int(self.r.H * 0.12)
            for line in lines:
                for part in self.r.wrap(line, "small", rect.width - 2 * self.pad):
                    self._text(part, "small", DIM, topleft=(x0, y))
                    y += int(self.r.H * 0.04)
            return

        # Başlık: ad, CANLI/KAYIT rozeti, ağ boyutu, spike sayısı
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

        # Raster ve grup etiketleri
        rr = self.raster_rect
        scr.blit(self.surf, rr)
        for start, n, label, col in self.bands:
            top = rr.y + start * self.row_h
            pygame.draw.line(scr, LINE, (rr.x, top), (rr.right - 1, top))
            lab = self.r.text(label, "tiny", col)
            bg = pygame.Surface((min(lab.get_width() + 8, rr.width), lab.get_height()), pygame.SRCALPHA)
            bg.fill((10, 12, 18, 190))
            scr.blit(bg, (rr.x, top + 1))
            scr.blit(lab, (rr.x + 4, top + 1), area=pygame.Rect(0, 0, rr.width - 8, lab.get_height()))
        pygame.draw.rect(scr, LINE, rr, 1)
        k = sum(n for _, n, _, _ in self.bands)
        self._text(tx["panel_sample"].format(k=k, n=fmt_int(self.info.get("neurons", 0), self.lang)),
                   "tiny", DIM, topright=(rr.right, rr.bottom + 2))

        # Çubuklar: girdi ve çıktı
        y = rr.bottom + int(self.r.H * 0.035)
        w = rect.width - 2 * self.pad
        self._text(tx["panel_eyes"], "tiny", DIM, topleft=(x0, y))
        y += int(self.r.H * 0.028)
        vmax_in = 200.0
        y = self._lr_bar(x0, y, w, tx["panel_fruit"], self.rin.get("fruit_L", 0), self.rin.get("fruit_R", 0),
                         vmax_in, GREEN)
        y = self._lr_bar(x0, y, w, tx["panel_loom"], self.rin.get("loom_L", 0), self.rin.get("loom_R", 0),
                         vmax_in, RED)
        y += int(self.r.H * 0.008)
        self._text(tx["panel_cmds"], "tiny", DIM, topleft=(x0, y))
        y += int(self.r.H * 0.028)
        y = self._lr_bar(x0, y, w, tx["panel_steer"], self.rout.get("steer_L", 0), self.rout.get("steer_R", 0),
                         150.0, ORANGE)
        y = self._turn_speed(x0, y, w)
        if self.escape_flash > 0:
            a = int(255 * min(1.0, self.escape_flash))
            esc = self.r.text(tx["panel_escape"], "small", (255, 80, 200))
            esc.set_alpha(a)
            scr.blit(esc, esc.get_rect(midtop=(rect.centerx, y + 2)))
            esc.set_alpha(255)

    def _lr_bar(self, x, y, w, label, left, right, vmax, color) -> int:
        """Ortadan iki yana büyüyen sol/sağ çubuk."""
        scr = self.r.screen
        lab_w = int(w * 0.30)
        self._text(label, "tiny", TEXT, midleft=(x, y + 8))
        bx, bw = x + lab_w, w - lab_w
        cx = bx + bw // 2
        h = 14
        pygame.draw.rect(scr, (40, 44, 60), (bx, y + 1, bw, h), border_radius=3)
        lw = int(min(1.0, left / vmax) * (bw // 2 - 2))
        rw = int(min(1.0, right / vmax) * (bw // 2 - 2))
        if lw > 0:
            pygame.draw.rect(scr, color, (cx - lw, y + 1, lw, h), border_radius=3)
        if rw > 0:
            pygame.draw.rect(scr, color, (cx, y + 1, rw, h), border_radius=3)
        pygame.draw.line(scr, TEXT, (cx, y - 1), (cx, y + h + 2))
        self._text(self.tx["panel_left"], "tiny", DIM, topleft=(bx + 3, y + h + 2))
        self._text(self.tx["panel_right"], "tiny", DIM, topright=(bx + bw - 3, y + h + 2))
        return y + h + int(self.r.H * 0.03)

    def _turn_speed(self, x, y, w) -> int:
        """Beynin kararı: dönüş (ok) ve hız (çubuk)."""
        scr = self.r.screen
        lab_w = int(w * 0.30)
        bx, bw = x + lab_w, w - lab_w
        cx = bx + bw // 2
        h = 14
        # Dönüş: merkezden sola/sağa ok
        self._text(self.tx["panel_turn"], "tiny", TEXT, midleft=(x, y + 8))
        pygame.draw.rect(scr, (40, 44, 60), (bx, y + 1, bw, h), border_radius=3)
        L = int(max(-1.0, min(1.0, self.turn)) * (bw // 2 - 10))
        if abs(L) > 2:
            tip = cx + L
            base = cx
            pygame.draw.rect(scr, (120, 200, 255), (min(base, tip), y + 4, abs(L), h - 6))
            d = 1 if L > 0 else -1
            pygame.draw.polygon(scr, (120, 200, 255), [(tip + d * 9, y + 1 + h // 2), (tip, y - 1), (tip, y + h + 3)])
        pygame.draw.line(scr, TEXT, (cx, y - 1), (cx, y + h + 2))
        y += h + int(self.r.H * 0.022)
        # Hız
        self._text(self.tx["panel_speed"], "tiny", TEXT, midleft=(x, y + 8))
        pygame.draw.rect(scr, (40, 44, 60), (bx, y + 1, bw, h), border_radius=3)
        sw = int(max(0.0, min(1.0, self.fwd)) * bw)
        if sw > 0:
            pygame.draw.rect(scr, (120, 200, 255), (bx, y + 1, sw, h), border_radius=3)
        return y + h + int(self.r.H * 0.015)
