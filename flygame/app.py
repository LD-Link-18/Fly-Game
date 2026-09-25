"""Oyun uygulaması: başlık -> geri sayım -> tur -> sonuç döngüsü.

Solda ziyaretçi (klavye), sağda rakip ajan. İki taraf aynı tohumla
kurulmuş ayrı dünyalarda, adım adım kilitli (lockstep) oynar.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pygame

from .agents.base import Agent, AgentKind
from .agents.human import HumanAgent
from .audio import Sounds
from .config import GameConfig
from .recording import Recording, new_recording
from .render import BG, DIM, HUMAN_COLOR, OPPONENT_COLOR, TEXT, Renderer
from .sensing import build_state
from .texts import get_texts
from .world import World

MAX_STEPS_PER_FRAME = 5  # yavaş karelerde simülasyonun sarmala girmesini önler


class App:
    def __init__(self, cfg: GameConfig, opponent: Agent, fixed_seed: int | None = None):
        self.cfg = cfg
        self.opponent = opponent
        self.fixed_seed = fixed_seed
        self.lang = cfg.display.language
        self.tx = get_texts(self.lang)

        pygame.init()
        pygame.display.set_caption("Beat the Fly")
        self.fullscreen = cfg.display.fullscreen
        self.screen = self._make_screen()
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen, cfg, self.tx)
        self.sounds = Sounds(cfg.display.sound, cfg.display.volume, cfg.swatter.loom_s)
        self.human = HumanAgent(pygame.key.get_pressed)

        self.state = "title"
        self.state_t = 0.0
        self.worlds: list[World] = []
        self.recs: list[Recording] = []
        self.seed = 0
        self.last_tick_sound = -1

    def _make_screen(self) -> pygame.Surface:
        d = self.cfg.display
        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode((d.width, d.height))

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
        self.acc = 0.0
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
            self.recs[i].add(action, agent.telemetry() if i == 1 else None)
            world.step(action)
            self.renderer.on_events(i, world.events)
            for e in world.events:
                kinds.add(e.kind if i == 0 or e.kind != "fruit" else "fruit_other")
        # Aynı anda iki tarafta olan olaylar için sesi bir kez çal
        if "slam" in kinds:
            self.sounds.play("slam")
        if "swatter_spawn" in kinds:
            self.sounds.play("whoosh")
        if "fruit" in kinds:
            self.sounds.play("fruit")
        if "fruit_other" in kinds:
            self.sounds.play("fruit_other")

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
        self.sounds.play("win" if h > o else "lose")
        self.set_state("results")

    # -- ana döngü ------------------------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.cfg.display.fps) / 1000.0
            dt = min(dt, 0.1)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_F11:
                        self.fullscreen = not self.fullscreen
                        self.screen = self._make_screen()
                        self.renderer.screen = self.screen
                        self.renderer.layout()
                    elif ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                        if self.state == "title":
                            self.start_round()
                        elif self.state == "results" and self.state_t > 1.5:
                            self.set_state("title")
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def update(self, dt: float) -> None:
        self.state_t += dt
        self.renderer.update_fx(dt)
        if self.state == "countdown":
            n = int(self.cfg.round.countdown_s - self.state_t) + 1
            if n != self.last_tick_sound and n > 0:
                self.sounds.play("tick")
                self.last_tick_sound = n
            if self.state_t >= self.cfg.round.countdown_s:
                self.sounds.play("go")
                self.set_state("play")
        elif self.state == "play":
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
        elif self.state == "results":
            if self.state_t > self.cfg.display.results_s:
                self.set_state("title")

    # -- çizim ------------------------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(BG)
        r = self.renderer
        if self.state == "title":
            self.draw_title()
            return
        now = time.perf_counter()
        colors = (HUMAN_COLOR, OPPONENT_COLOR)
        names = (self.tx["you"], self.opponent.name)
        agents = (self.human, self.opponent)
        for i, world in enumerate(self.worlds):
            r.draw_panel(i, world, colors[i], now)
            r.draw_header(i, names[i], world.score, colors[i], world)
            r.draw_badge(i, agents[i].label(self.lang), agents[i].kind, agents[i].ui_note(self.lang))
        r.draw_timer(self.worlds[0].time_left)

        if self.state == "countdown":
            n = int(self.cfg.round.countdown_s - self.state_t) + 1
            r.dim(90)
            r.draw_center_text(str(max(1, n)))
        elif self.state == "play" and self.state_t < 0.6:
            r.draw_center_text(self.tx["go"], color=(120, 255, 140))
        elif self.state == "results":
            self.draw_results()

    def draw_title(self) -> None:
        r = self.renderer
        tx = self.tx
        r.draw_center_text(tx["title"], "huge", (255, 210, 60), dy=-0.25)
        r.blit_text(tx["subtitle"].format(secs=int(self.cfg.round.duration_s)), "small", TEXT,
                    center=(r.W // 2, int(r.H * 0.40)))
        r.blit_text(f"{tx['opponent']}:", "small", DIM, center=(r.W // 2, int(r.H * 0.50)))
        r.blit_text(self.opponent.label(self.lang), "mid", OPPONENT_COLOR, center=(r.W // 2, int(r.H * 0.56)))
        note = self.opponent.ui_note(self.lang)
        if note:
            r.blit_text(note, "small", DIM, center=(r.W // 2, int(r.H * 0.61)))
        r.blit_text(tx["controls"], "small", TEXT, center=(r.W // 2, int(r.H * 0.70)))
        if int(self.state_t * 2) % 2 == 0:
            r.blit_text(tx["press_start"], "mid", (120, 255, 140), center=(r.W // 2, int(r.H * 0.82)))

    def draw_results(self) -> None:
        r = self.renderer
        tx = self.tx
        if self.state_t < 1.2:
            r.draw_center_text(tx["time_up"], color=(255, 210, 60))
            return
        r.dim(170)
        h, o = self.worlds[0], self.worlds[1]
        if h.score > o.score:
            head, col, sub = tx["you_win"], (120, 255, 140), tx["prize"]
        elif h.score < o.score:
            head, col, sub = tx["you_lose"], (255, 110, 110), tx["no_prize"]
        else:
            head, col, sub = tx["tie"], (255, 210, 60), tx["no_prize"]
        r.draw_center_text(head, "huge", col, dy=-0.2)
        r.blit_text(f"{h.score}  :  {o.score}", "big", TEXT, center=(r.W // 2, int(r.H * 0.47)))
        r.blit_text(sub, "mid", col, center=(r.W // 2, int(r.H * 0.60)))
        for i, w in enumerate((h, o)):
            cx = r.panels[i].centerx
            r.blit_text(f"{w.fruits_collected} {tx['fruits']}  /  {w.hits} {tx['hits']}", "small", DIM,
                        center=(cx, int(r.H * 0.70)))
        r.blit_text(tx["continue"], "small", TEXT, center=(r.W // 2, int(r.H * 0.82)))
        r.blit_text(f"{tx['seed']}: {self.seed}", "tiny", DIM, bottomright=(r.W - 10, r.H - 6))
