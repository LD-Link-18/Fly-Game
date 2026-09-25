"""Oyun uygulaması (stant modu dahil).

Durumlar:
  title     : çekici ekran; sayfalar sırayla değişir: başlık / sinek nasıl oynuyor / skor tablosu
  choose    : birden fazla rakip varsa ziyaretçi seçer (SOL/SAĞ, BOŞLUK)
  countdown : 3-2-1
  play      : tur (solda ziyaretçi, sağda rakip, aynı tohum, adım adım kilitli)
  results   : sonuç ve ödül kararı
  initials  : skor tablosuna giren ziyaretçi 3 harf girer
  board     : skor tablosu (yeni kayıt vurgulu), sonra başlığa dönülür

Kiosk modu (config.stand.kiosk / --kiosk): tam ekran, fare gizli, çıkış için ESC
basılı tutulur. Operatör tuşları: F1 yardım, F2 rakip sırası, F3 dil, F4 ses,
F9 (iki kez) skor tablosunu sıfırla, F11 tam ekran (kiosk dışında).
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import pygame

from .agents.base import DISPLAY_NAMES, Agent, AgentKind
from .agents.human import HumanAgent
from .audio import Sounds
from .config import GameConfig
from .leaderboard import Leaderboard, PrizeDecision, decide_prize
from .neuron_panel import NeuronPanel
from .recording import Recording, new_recording
from .render import BG, DIM, HUMAN_COLOR, KIND_BADGE, OPPONENT_COLOR, TEXT, Renderer
from .sensing import build_state
from .texts import ALPHABETS, fmt_int, get_texts
from .world import World

MAX_STEPS_PER_FRAME = 5  # yavaş karelerde simülasyonun sarmala girmesini önler
GOLD = (255, 210, 60)
GREEN = (120, 255, 140)
RED = (255, 110, 110)

START_KEYS = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_UP, pygame.K_w)
CONFIRM_KEYS = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER)
LEFT_KEYS = (pygame.K_LEFT, pygame.K_a)
RIGHT_KEYS = (pygame.K_RIGHT, pygame.K_d)
UP_KEYS = (pygame.K_UP, pygame.K_w)
DOWN_KEYS = (pygame.K_DOWN, pygame.K_s)


def draw_star(surf: pygame.Surface, center: tuple, r: float, color: tuple) -> None:
    """Beş köşeli yıldız (yazı tipinde ★ karakteri olmadığı için çizilir)."""
    cx, cy = center
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    pygame.draw.polygon(surf, color, pts)


def kind_name(kind: str, lang: str) -> str:
    try:
        return DISPLAY_NAMES.get(lang, DISPLAY_NAMES["en"])[AgentKind(kind)]
    except ValueError:
        return kind


class App:
    def __init__(self, cfg: GameConfig, opponents: Agent | list[Agent], fixed_seed: int | None = None):
        self.cfg = cfg
        self.st = cfg.stand
        self.kiosk = cfg.stand.kiosk
        self.opponents = [opponents] if isinstance(opponents, Agent) else list(opponents)
        self.sel = 0
        self.fixed_seed = fixed_seed
        self.lang = cfg.display.language
        self.tx = get_texts(self.lang)

        pygame.init()
        pygame.display.set_caption("Beat the Fly")
        self.fullscreen = cfg.display.fullscreen or self.kiosk
        self.screen = self._make_screen()
        if self.kiosk:
            pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen, cfg, self.tx, side_panel=self.st.neuron_panel)
        self.panel = NeuronPanel(self.renderer) if self.st.neuron_panel else None
        self.sound_on = cfg.display.sound
        self.sounds = Sounds(cfg.display.sound, cfg.display.volume, cfg.swatter.loom_s)
        self.human = HumanAgent(pygame.key.get_pressed)
        self.board = Leaderboard(self.st.leaderboard_path, self.st.leaderboard_size, self.st.leaderboard_scope)

        self.state = "title"
        self.state_t = 0.0
        self.idle_t = 0.0
        self.worlds: list[World] = []
        self.recs: list[Recording] = []
        self.seed = 0
        self.last_tick_sound = -1
        self.decision: PrizeDecision | None = None
        self.entry_id: int | None = None
        self.highlight_id: int | None = None
        self.human_active = False
        self.letters = [0, 0, 0]
        self.letter_pos = 0
        self.letters_touched = False
        self.esc_since: float | None = None
        self.show_help = False
        self.toast_msg, self.toast_until = "", 0.0
        self.reset_armed_until = 0.0

    @property
    def opponent(self) -> Agent:
        return self.opponents[self.sel]

    def _make_screen(self) -> pygame.Surface:
        d = self.cfg.display
        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode((d.width, d.height))

    def _relayout(self) -> None:
        self.renderer.screen = self.screen
        self.renderer.layout()
        if self.panel:
            self.panel.relayout()

    def toast(self, msg: str, secs: float = 3.0) -> None:
        self.toast_msg, self.toast_until = msg, time.monotonic() + secs

    def play_sound(self, name: str) -> None:
        if self.sound_on:
            self.sounds.play(name)

    def set_lang(self, lang: str) -> None:
        self.lang = lang
        self.tx = get_texts(lang)
        self.renderer.tx = self.tx
        if self.panel and self.panel.agent is not None:
            self.panel.tx, self.panel.lang = self.tx, lang
            self.panel.relayout()
        self.toast(self.tx["lang_name"])

    # -- tur yönetimi ---------------------------------------------------------

    def _pick_seed(self) -> int:
        # Kayıttan tekrar yapan rakip, kaydın tohumunu dayatır
        pick = getattr(self.opponent, "pick_seed", None)
        if pick is not None:
            return pick()
        if self.fixed_seed is not None:
            return self.fixed_seed
        return random.randrange(1, 10**9)

    def start_round(self) -> None:
        self.seed = self._pick_seed()
        self.worlds = [World(self.cfg, self.seed), World(self.cfg, self.seed)]
        self.human.reset(self.seed, self.cfg)
        self.opponent.reset(self.seed, self.cfg)
        self.recs = [
            new_recording(self.seed, self.human.describe(), self.cfg),
            new_recording(self.seed, self.opponent.describe(), self.cfg),
        ]
        self.renderer.reset_fx()
        if self.panel:
            self.panel.reset(self.opponent, self.tx, self.lang)
        self.acc = 0.0
        self.human_active = False
        self.decision = None
        self.entry_id = None
        self.set_state("countdown")

    def set_state(self, s: str) -> None:
        self.state = s
        self.state_t = 0.0
        self.last_tick_sound = -1

    def step_worlds(self) -> None:
        agents = (self.human, self.opponent)
        kinds: set[str] = set()
        for i, (world, agent) in enumerate(zip(self.worlds, agents)):
            action = agent.get_action(build_state(world)).clamped()
            if i == 0 and (action.turn != 0.0 or action.forward > 0.0):
                self.human_active = True
            tel = agent.telemetry() if i == 1 else None
            self.recs[i].add(action, tel)
            if i == 1 and self.panel:
                self.panel.push(tel)
            world.step(action)
            self.renderer.on_events(i, world.events)
            for e in world.events:
                kinds.add(e.kind if i == 0 or e.kind != "fruit" else "fruit_other")
        # Aynı anda iki tarafta olan olaylar için sesi bir kez çal
        for kind, sound in (("slam", "slam"), ("swatter_spawn", "whoosh"), ("fruit", "fruit"),
                            ("fruit_other", "fruit_other")):
            if kind in kinds:
                self.play_sound(sound)

    def finish_round(self) -> None:
        for rec, world in zip(self.recs, self.worlds):
            rec.final_score, rec.fruits, rec.hits = world.score, world.fruits_collected, world.hits
        rc = self.cfg.recording
        if rc.enabled:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for rec, who in zip(self.recs, ("human", "opponent")):
                # Kayıttan tekrarın tekrarını kaydetmenin anlamı yok
                if rec.agent.get("kind") == AgentKind.REPLAY.value:
                    continue
                try:
                    rec.save(Path(rc.directory) / f"{stamp}_seed{self.seed}_{who}_{rec.agent['name']}.json.gz")
                except OSError as e:
                    print(f"Could not save recording: {e}")
        h, o = self.worlds[0].score, self.worlds[1].score
        opp = self.opponent
        self.decision = decide_prize(self.cfg.prize, h, o, opp.effective_kind, self.board.prizes(today_only=True))
        # Ziyaretçi hiç tuşa basmadıysa (ör. başlatıp gitti) tur skor tablosuna yazılmaz
        if self.human_active:
            self.entry_id = self.board.add(h, opp.effective_kind, opp.display_name("en"), o,
                                           self.decision.outcome, self.decision.prize, self.seed)
        self.play_sound("win" if self.decision.outcome == "win" else "lose")
        self.set_state("results")

    def after_results(self) -> None:
        if self.entry_id is not None and self.st.initials and self.board.qualifies(self.entry_id):
            self.letters, self.letter_pos, self.letters_touched = [0, 0, 0], 0, False
            self.set_state("initials")
        elif self.entry_id is not None:
            self.highlight_id = self.entry_id
            self.set_state("board")
        else:
            self.set_state("title")

    def confirm_initials(self) -> None:
        alpha = ALPHABETS.get(self.lang, ALPHABETS["en"])
        initials = "".join(alpha[i] for i in self.letters) if self.letters_touched else "???"
        self.board.set_initials(self.entry_id, initials)
        self.highlight_id = self.entry_id
        self.set_state("board")

    # -- girdi ------------------------------------------------------------------------

    def on_key(self, k: int) -> None:
        s = self.state
        if s == "title":
            if k in START_KEYS:
                if len(self.opponents) > 1:
                    self.set_state("choose")
                else:
                    self.start_round()
        elif s == "choose":
            if k in LEFT_KEYS:
                self.sel = (self.sel - 1) % len(self.opponents)
            elif k in RIGHT_KEYS:
                self.sel = (self.sel + 1) % len(self.opponents)
            elif k in START_KEYS:
                self.start_round()
        elif s == "results":
            if k in CONFIRM_KEYS and self.state_t > 2.0:
                self.after_results()
        elif s == "initials":
            n = len(ALPHABETS.get(self.lang, ALPHABETS["en"]))
            if k in UP_KEYS:
                self.letters[self.letter_pos] = (self.letters[self.letter_pos] + 1) % n
                self.letters_touched = True
            elif k in DOWN_KEYS:
                self.letters[self.letter_pos] = (self.letters[self.letter_pos] - 1) % n
                self.letters_touched = True
            elif k in LEFT_KEYS:
                self.letter_pos = max(0, self.letter_pos - 1)
            elif k in RIGHT_KEYS:
                self.letter_pos = min(2, self.letter_pos + 1)
            elif k in CONFIRM_KEYS:
                self.confirm_initials()
        elif s == "board":
            if k in START_KEYS:
                self.highlight_id = None
                self.set_state("title")

    def on_operator_key(self, k: int) -> bool:
        """Operatör tuşları; işlendiyse True."""
        now = time.monotonic()
        if k == pygame.K_F1:
            self.show_help = not self.show_help
        elif k == pygame.K_F2:
            if self.state not in ("countdown", "play") and len(self.opponents) > 1:
                self.sel = (self.sel + 1) % len(self.opponents)
                self.toast(self.tx["opponent_first"].format(name=self.opponent.display_name(self.lang)))
        elif k == pygame.K_F3:
            self.set_lang("tr" if self.lang == "en" else "en")
        elif k == pygame.K_F4:
            self.sound_on = not self.sound_on
            self.toast(self.tx["sound_on" if self.sound_on else "sound_off"])
        elif k == pygame.K_F9:
            if now < self.reset_armed_until:
                backup = self.board.reset()
                self.reset_armed_until = 0.0
                self.toast(self.tx["reset_done"].format(path=backup or "-"), 6.0)
            else:
                self.reset_armed_until = now + 3.0
                self.toast(self.tx["reset_confirm"], 3.0)
        elif k == pygame.K_F11 and not self.kiosk:
            self.fullscreen = not self.fullscreen
            self.screen = self._make_screen()
            self._relayout()
        else:
            return False
        return True

    # -- ana döngü ------------------------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            dt = min(self.clock.tick(self.cfg.display.fps) / 1000.0, 0.1)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    self.idle_t = 0.0
                    if ev.key == pygame.K_ESCAPE:
                        if not self.kiosk:
                            running = False
                        else:
                            # Kiosk: basılı tutulursa çık; kısa basış tur dışında başlığa döndürür
                            self.esc_since = time.monotonic()
                            if self.state not in ("countdown", "play"):
                                self.set_state("title")
                    elif not self.on_operator_key(ev.key):
                        self.on_key(ev.key)
                elif ev.type == pygame.KEYUP and ev.key == pygame.K_ESCAPE:
                    self.esc_since = None
            if self.exit_requested():
                running = False
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def exit_requested(self) -> bool:
        """Kiosk: ESC yeterince uzun basılı tutuldu mu?"""
        return self.esc_since is not None and time.monotonic() - self.esc_since >= self.st.exit_hold_s

    def update(self, dt: float) -> None:
        self.state_t += dt
        self.idle_t += dt
        self.renderer.update_fx(dt)
        if self.panel:
            self.panel.update(dt)
        s = self.state
        if s == "choose" and self.idle_t > 20.0:
            self.set_state("title")
        elif s == "countdown":
            n = int(self.cfg.round.countdown_s - self.state_t) + 1
            if n != self.last_tick_sound and n > 0:
                self.play_sound("tick")
                self.last_tick_sound = n
            if self.state_t >= self.cfg.round.countdown_s:
                self.play_sound("go")
                self.set_state("play")
        elif s == "play":
            self.acc += dt
            step = self.worlds[0].dt
            n = 0
            while self.acc >= step and n < MAX_STEPS_PER_FRAME and not self.worlds[0].done:
                self.step_worlds()
                self.acc -= step
                n += 1
            if n == MAX_STEPS_PER_FRAME:
                self.acc = 0.0
            if self.worlds[0].done:
                self.finish_round()
        elif s == "results":
            limit = self.cfg.display.results_s * (2 if self.decision and self.decision.prize else 1)
            if self.state_t > limit and self.idle_t > 3.0:
                self.after_results()
        elif s == "initials":
            if self.idle_t > self.st.idle_return_s:
                self.confirm_initials()
        elif s == "board":
            if self.state_t > max(10.0, self.st.attract_page_s * 1.5) and self.idle_t > 5.0:
                self.highlight_id = None
                self.set_state("title")

    # -- çizim ------------------------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(BG)
        s = self.state
        if s == "title":
            pages = self._attract_pages()
            page = pages[int(self.state_t / self.st.attract_page_s) % len(pages)]
            {"title": self.draw_title, "how": self.draw_how, "board": self.draw_board}[page]()
        elif s == "choose":
            self.draw_choose()
        elif s == "initials":
            self.draw_initials()
        elif s == "board":
            self.draw_board()
        else:
            self.draw_game()
        self.draw_overlays()

    def _attract_pages(self) -> list[str]:
        pages = ["title"]
        if any(o.panel_info() for o in self.opponents):
            pages.append("how")
        if self.board.top():
            pages.append("board")
        return pages

    def draw_game(self) -> None:
        r = self.renderer
        now = time.perf_counter()
        colors = (HUMAN_COLOR, OPPONENT_COLOR)
        agents = (self.human, self.opponent)
        names = (self.tx["you"], self.opponent.display_name(self.lang))
        for i, world in enumerate(self.worlds):
            r.draw_panel(i, world, colors[i], now)
            r.draw_header(i, names[i], world.score, colors[i], world)
            r.draw_badge(i, agents[i].label(self.lang), agents[i].kind, agents[i].ui_note(self.lang))
        r.draw_timer(self.worlds[0].time_left)
        if self.panel:
            self.panel.draw()
        if self.state == "countdown":
            n = int(self.cfg.round.countdown_s - self.state_t) + 1
            r.dim(90)
            r.draw_center_text(str(max(1, n)))
        elif self.state == "play" and self.state_t < 0.6:
            r.draw_center_text(self.tx["go"], color=GREEN)
        elif self.state == "results":
            self.draw_results()

    # -- sayfalar ---------------------------------------------------------------------

    def _prize_lines(self) -> list[str]:
        pc = self.cfg.prize
        if not pc.enabled:
            return []
        eligible = [o for o in self.opponents if o.effective_kind in pc.eligible]
        if not eligible:
            return []
        lines = [self.tx["title_prize"].format(name=eligible[0].display_name(self.lang))]
        if pc.min_score > 0 or pc.margin > 1:
            lines.append(self.tx["title_prize_rule"].format(min=pc.min_score, margin=pc.margin))
        return lines

    def draw_title(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        r.draw_center_text(tx["title"], "huge", GOLD, dy=-0.28)
        r.blit_text(tx["subtitle"].format(secs=int(self.cfg.round.duration_s)), "small", TEXT,
                    center=(W // 2, int(H * 0.37)))
        y = int(H * 0.46)
        if len(self.opponents) == 1:
            r.blit_text(f"{tx['opponent']}:", "small", DIM, center=(W // 2, y))
            r.blit_text(self.opponent.label(self.lang), "mid", OPPONENT_COLOR, center=(W // 2, y + int(H * 0.055)))
            note = self.opponent.ui_note(self.lang)
            if note:
                r.blit_text(note, "small", DIM, center=(W // 2, y + int(H * 0.105)))
        else:
            names = "   /   ".join(o.display_name(self.lang) for o in self.opponents)
            r.blit_text(tx["opponent_pick"], "small", DIM, center=(W // 2, y))
            r.blit_text(names, "mid", OPPONENT_COLOR, center=(W // 2, y + int(H * 0.055)))
        y = int(H * 0.64)
        for i, line in enumerate(self._prize_lines()):
            pulse = 0.75 + 0.25 * math.sin(self.state_t * 4) if i == 0 else 1.0
            col = tuple(int(c * pulse) for c in GOLD) if i == 0 else DIM
            r.blit_text(line, "mid" if i == 0 else "small", col, center=(W // 2, y))
            y += int(H * 0.05)
        r.blit_text(tx["controls"], "small", TEXT, center=(W // 2, int(H * 0.78)))
        if int(self.state_t * 2) % 2 == 0:
            r.blit_text(tx["press_start"], "mid", GREEN, center=(W // 2, int(H * 0.87)))

    def draw_how(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        info = next((o.panel_info() for o in self.opponents if o.panel_info()), {}) or {}
        n = fmt_int(info.get("neurons", 138639), self.lang)
        r.draw_center_text(tx["how_title"], "big", GOLD, dy=-0.36)
        y = int(H * 0.26)
        for line in tx["how_lines"]:
            r.blit_text(line.format(neurons=n), "small", TEXT, center=(W // 2, y))
            y += int(H * 0.058)
        if int(self.state_t * 2) % 2 == 0:
            r.blit_text(tx["press_start"], "mid", GREEN, center=(W // 2, int(H * 0.87)))

    def draw_choose(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        r.draw_center_text(tx["choose_title"], "big", GOLD, dy=-0.36)
        n = len(self.opponents)
        gap = int(W * 0.03)
        cw = int(min(W * 0.4, (W - (n + 1) * gap) / n))
        ch = int(H * 0.5)
        x = (W - n * cw - (n - 1) * gap) // 2
        stats = self.board.stats()
        for i, o in enumerate(self.opponents):
            rect = pygame.Rect(x + i * (cw + gap), int(H * 0.22), cw, ch)
            sel = i == self.sel
            pygame.draw.rect(self.screen, (34, 38, 52) if sel else (24, 27, 38), rect, border_radius=14)
            pygame.draw.rect(self.screen, OPPONENT_COLOR if sel else (60, 65, 85), rect, 4 if sel else 2,
                             border_radius=14)
            cx, y = rect.centerx, rect.y + int(H * 0.05)
            r.blit_text(o.display_name(self.lang), "mid", TEXT if sel else DIM, center=(cx, y))
            y += int(H * 0.065)
            badge = r.text(o.label(self.lang), "small", (15, 15, 20))
            box = badge.get_rect(center=(cx, y)).inflate(14, 8)
            pygame.draw.rect(self.screen, KIND_BADGE.get(o.kind, DIM), box, border_radius=6)
            self.screen.blit(badge, badge.get_rect(center=box.center))
            y += int(H * 0.06)
            desc = tx.get(f"desc_{o.kind.value}")
            if desc:
                for line in r.wrap(desc, "small", cw - 40):
                    r.blit_text(line, "small", TEXT if sel else DIM, center=(cx, y))
                    y += int(H * 0.04)
                y += int(H * 0.01)
            note = o.ui_note(self.lang)
            if note:
                for line in r.wrap(note, "tiny", cw - 30, sep=" · "):
                    r.blit_text(line, "tiny", DIM, center=(cx, y))
                    y += int(H * 0.03)
            st = stats.get(o.effective_kind)
            if st:
                r.blit_text(tx["avg_here"].format(x=f"{st['opp_avg']:.1f}"), "small", DIM,
                            center=(cx, rect.bottom - int(H * 0.15)))
                r.blit_text(tx["record_line"].format(w=st["wins"], n=st["rounds"]), "small", TEXT,
                            center=(cx, rect.bottom - int(H * 0.11)))
            if self.cfg.prize.enabled and o.effective_kind in self.cfg.prize.eligible:
                tag = r.text(tx["prize_tag"], "mid", (20, 16, 0))
                tb = tag.get_rect(center=(cx, rect.bottom - int(H * 0.045))).inflate(20, 8)
                pygame.draw.rect(self.screen, GOLD, tb, border_radius=8)
                self.screen.blit(tag, tag.get_rect(center=tb.center))
        r.blit_text(tx["choose_hint"], "mid", GREEN, center=(W // 2, int(H * 0.84)))

    def draw_results(self) -> None:
        r, tx = self.renderer, self.tx
        if self.state_t < 1.2:
            r.draw_center_text(tx["time_up"], color=GOLD)
            return
        r.dim(185)
        W, H = r.W, r.H
        h, o = self.worlds[0], self.worlds[1]
        d = self.decision
        head, col = {"win": (tx["you_win"], GREEN), "tie": (tx["tie"], GOLD),
                     "loss": (tx["you_lose"], RED)}[d.outcome]
        r.draw_center_text(head, "huge", col, dy=-0.24)
        r.blit_text(f"{h.score}  :  {o.score}", "big", TEXT, center=(W // 2, int(H * 0.43)))
        eligible = [x for x in self.opponents if x.effective_kind in self.cfg.prize.eligible]
        pc = self.cfg.prize
        lines: list[tuple[str, tuple, str]] = []
        if d.prize:
            pulse = 0.7 + 0.3 * math.sin(self.state_t * 6)
            lines = [(tx["prize"], tuple(int(c * pulse) for c in GOLD), "big"), (tx["prize_show"], TEXT, "mid")]
        elif d.outcome == "win":
            if d.reason == "not_eligible" and pc.enabled and eligible:
                lines = [(tx["prize_not_eligible"].format(name=eligible[0].display_name(self.lang)), GOLD, "mid")]
            elif d.reason == "below_threshold":
                lines = [(tx["prize_below"].format(min=pc.min_score, margin=pc.margin), GOLD, "mid")]
            elif d.reason == "no_stock":
                lines = [(tx["prize_no_stock"], GOLD, "mid")]
        else:
            lines = [(tx["no_prize"], col, "mid")]
        y = int(H * 0.56)
        for text, c, font in lines:
            r.blit_text(text, font, c, center=(W // 2, y))
            y += int(H * (0.09 if font == "big" else 0.06))
        for i, w in enumerate((h, o)):
            # Arenanın alt kısmında (kararmış), altındaki etiketlerle çakışmasın
            r.blit_text(f"{w.fruits_collected} {tx['fruits']}  /  {w.hits} {tx['hits']}", "small", DIM,
                        midbottom=(r.panels[i].centerx, r.panels[i].bottom - int(H * 0.015)))
        if self.state_t > 2.0:
            r.blit_text(tx["continue"], "small", TEXT, center=(W // 2, int(H * 0.84)))
        r.blit_text(f"{tx['seed']}: {self.seed}", "tiny", DIM, bottomright=(W - 10, H - 6))

    def draw_initials(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        alpha = ALPHABETS.get(self.lang, ALPHABETS["en"])
        r.draw_center_text(tx["enter_initials"], "mid", GOLD, dy=-0.33)
        score = self.worlds[0].score if self.worlds else 0
        rank = self.board.rank(self.entry_id) if self.entry_id else None
        r.blit_text(f"{score}", "huge", TEXT, center=(W // 2, int(H * 0.33)))
        if rank:
            r.blit_text(f"#{rank}", "mid", DIM, center=(W // 2, int(H * 0.44)))
        size = int(H * 0.16)
        gap = int(size * 0.25)
        x0 = W // 2 - (3 * size + 2 * gap) // 2
        for i in range(3):
            rect = pygame.Rect(x0 + i * (size + gap), int(H * 0.52), size, size)
            active = i == self.letter_pos
            pygame.draw.rect(self.screen, (34, 38, 52), rect, border_radius=12)
            pygame.draw.rect(self.screen, GOLD if active else (60, 65, 85), rect, 5 if active else 2,
                             border_radius=12)
            ch = alpha[self.letters[i]] if self.letters_touched else "?"
            r.blit_text(ch, "big", TEXT, center=rect.center)
            if active and int(self.state_t * 3) % 2 == 0:
                for dy in (-1, 1):  # yukarı/aşağı okları (uç, kutudan dışarı bakar)
                    cy = rect.centery + dy * (size // 2 + 18)
                    pygame.draw.polygon(self.screen, GOLD, [(rect.centerx - 12, cy - dy * 6),
                                                            (rect.centerx + 12, cy - dy * 6),
                                                            (rect.centerx, cy + dy * 10)])
        r.blit_text(tx["initials_hint"], "small", TEXT, center=(W // 2, int(H * 0.86)))

    def draw_board(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        scope = tx["board_today"] if self.st.leaderboard_scope == "today" else tx["board_all"]
        r.draw_center_text(tx["board_title"], "big", GOLD, dy=-0.41)
        r.blit_text(scope, "small", DIM, center=(W // 2, int(H * 0.16)))
        top = self.board.top()
        stats = self.board.stats()
        if not top:
            r.blit_text(tx["board_empty"], "mid", TEXT, center=(W // 2, int(H * 0.45)))
            return
        # Ziyaretçi skorları + rakiplerin ortalama skorları (karşılaştırma satırları)
        rows = [("visitor", e.score, e) for e in top]
        for kind, s in stats.items():
            rows.append(("bench", s["opp_avg"], kind))
        rows.sort(key=lambda x: (-x[1], x[0] == "bench"))
        y = int(H * 0.21)
        row_h = int(H * min(0.052, 0.55 / max(1, len(rows))))
        col_rank, col_name, col_score, col_vs = (int(W * f) for f in (0.26, 0.31, 0.56, 0.62))
        rank = 0
        for typ, score, obj in rows:
            if typ == "visitor":
                rank += 1
                hl = obj.id == self.highlight_id
                c = GOLD if hl else TEXT
                if hl:
                    pygame.draw.rect(self.screen, (60, 52, 20), (int(W * 0.23), y - 4, int(W * 0.54), row_h),
                                     border_radius=6)
                r.blit_text(f"{rank}.", "small", c, topright=(col_rank, y))
                r.blit_text(obj.initials, "small", c, topleft=(col_name, y))
                r.blit_text(str(obj.score), "small", c, topright=(col_score, y))
                vs = f"vs {kind_name(obj.opponent_kind, self.lang)}"
                vr = r.blit_text(vs, "small", DIM if not hl else c, topleft=(col_vs, y))
                if obj.prize:
                    draw_star(self.screen, (vr.right + 22, vr.centery), row_h * 0.33, GOLD)
            else:
                name = tx["board_avg"].format(name=kind_name(obj, self.lang))
                c = OPPONENT_COLOR
                r.blit_text("·", "small", c, topright=(col_rank, y))
                r.blit_text(name, "small", c, topleft=(col_name, y))
                r.blit_text(f"{score:.1f}", "small", c, topright=(col_score, y))
            y += row_h
        y = max(y + int(H * 0.02), int(H * 0.78))
        for kind, s in stats.items():
            r.blit_text(tx["board_vs"].format(name=kind_name(kind, self.lang), w=s["wins"], l=s["losses"],
                                              t=s["ties"]), "small", TEXT, center=(W // 2, y))
            y += int(H * 0.04)
        if self.cfg.prize.enabled:
            r.blit_text(tx["board_prizes"].format(n=self.board.prizes(), t=self.board.prizes(today_only=True)),
                        "small", GOLD, center=(W // 2, y))

    @staticmethod
    def _short_path(p: str, keep: int = 3) -> str:
        parts = Path(p).parts
        return p if len(parts) <= keep else ".../" + "/".join(parts[-keep:])

    # -- üst katmanlar ------------------------------------------------------------------

    def draw_overlays(self) -> None:
        r, tx = self.renderer, self.tx
        W, H = r.W, r.H
        now = time.monotonic()
        if self.toast_msg and now < self.toast_until:
            s = r.text(self.toast_msg, "small", TEXT)
            box = s.get_rect(midbottom=(W // 2, H - 14)).inflate(24, 12)
            pygame.draw.rect(self.screen, (40, 44, 60), box, border_radius=8)
            self.screen.blit(s, s.get_rect(center=box.center))
        if self.esc_since is not None:
            left = max(0.0, self.st.exit_hold_s - (now - self.esc_since))
            r.blit_text(tx["hold_esc"].format(s=math.ceil(left)), "mid", RED, midtop=(W // 2, 12))
        if self.show_help:
            lines = [tx["help_title"], ""]
            esc = tx["esc_kiosk"].format(s=self.st.exit_hold_s) if self.kiosk else tx["esc_normal"]
            lines += [l.format(esc=esc) for l in tx["help_lines"]]
            pc = self.cfg.prize
            lines += ["", "opponents: " + ", ".join(o.display_name(self.lang) for o in self.opponents),
                      f"prize: enabled={pc.enabled} margin={pc.margin} min_score={pc.min_score} "
                      f"eligible={pc.eligible} max_per_day={pc.max_per_day}",
                      f"leaderboard: {self._short_path(self.st.leaderboard_path)} ({self.st.leaderboard_scope})",
                      f"kiosk={self.kiosk}  sound={'on' if self.sound_on else 'off'}"]
            box = pygame.Rect(int(W * 0.15), int(H * 0.12), int(W * 0.7), int(H * 0.72))
            s = pygame.Surface(box.size, pygame.SRCALPHA)
            s.fill((10, 12, 20, 235))
            self.screen.blit(s, box)
            pygame.draw.rect(self.screen, GOLD, box, 2, border_radius=8)
            y = box.y + 20
            for i, line in enumerate(lines):
                r.blit_text(line, "mid" if i == 0 else "small", GOLD if i == 0 else TEXT, topleft=(box.x + 30, y))
                y += int(H * (0.06 if i == 0 else 0.04))
