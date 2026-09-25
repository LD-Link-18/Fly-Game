"""Faz 5 testleri: ödül kuralı, skor tablosu, stant akışı (ekransız), kiosk tuşları."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flygame.config import GameConfig, PrizeConfig
from flygame.leaderboard import Leaderboard, decide_prize


class TestPrizeRule(unittest.TestCase):
    def test_cases(self):
        pc = PrizeConfig()  # margin 1, min 0, eligible fly_brain
        self.assertTrue(decide_prize(pc, 20, 19, "fly_brain", 0).prize)
        d = decide_prize(pc, 20, 20, "fly_brain", 0)
        self.assertEqual((d.outcome, d.prize), ("tie", False))
        d = decide_prize(pc, 25, 10, "heuristic", 0)
        self.assertEqual((d.outcome, d.prize, d.reason), ("win", False, "not_eligible"))
        self.assertEqual(decide_prize(pc, 5, 9, "fly_brain", 0).outcome, "loss")

    def test_threshold_and_stock(self):
        pc = PrizeConfig(margin=3, min_score=15, max_per_day=2)
        self.assertEqual(decide_prize(pc, 20, 18, "fly_brain", 0).reason, "below_threshold")  # fark 2 < 3
        self.assertEqual(decide_prize(pc, 14, 5, "fly_brain", 0).reason, "below_threshold")   # 14 < 15
        self.assertTrue(decide_prize(pc, 20, 16, "fly_brain", 1).prize)
        self.assertEqual(decide_prize(pc, 20, 16, "fly_brain", 2).reason, "no_stock")
        self.assertTrue(decide_prize(PrizeConfig(margin=0), 10, 10, "fly_brain", 0).prize)  # margin 0: beraberlik yeter
        self.assertEqual(decide_prize(PrizeConfig(enabled=False), 30, 1, "fly_brain", 0).reason, "disabled")


class TestLeaderboard(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "board.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_ranking_persistence_and_stats(self):
        b = Leaderboard(self.path, size=3)
        ids = [b.add(s, "fly_brain", "FLY BRAIN", 20, "win" if s > 20 else "loss", s > 20, 1)
               for s in (10, 30, 25, 5)]
        b.set_initials(ids[1], "ABC")
        self.assertEqual([e.score for e in b.top()], [30, 25, 10])
        self.assertEqual(b.rank(ids[1]), 1)
        self.assertTrue(b.qualifies(ids[2]))
        self.assertFalse(b.qualifies(ids[3]))       # 4. sırada, liste 3 kişilik
        again = Leaderboard(self.path, size=3)      # dosyadan yeniden yükle
        self.assertEqual(again.top()[0].initials, "ABC")
        st = again.stats()["fly_brain"]
        self.assertEqual((st["rounds"], st["wins"], st["losses"], st["prizes"]), (4, 2, 2, 2))
        self.assertEqual(json.loads(self.path.read_text())["version"], 1)

    def test_corrupt_file_is_backed_up(self):
        self.path.write_text("{not json")
        b = Leaderboard(self.path)
        self.assertEqual(b.entries, [])
        self.assertEqual(len(list(Path(self.dir.name).glob("board.json.corrupt-*"))), 1)

    def test_reset_keeps_backup_and_today_scope(self):
        b = Leaderboard(self.path, scope="today")
        b.add(12, "heuristic", "HEURISTIC BOT", 30, "loss", False, 1)
        b.entries[0].day = "2000-01-01"            # eski gün: "bugün" kapsamında görünmez
        self.assertEqual(b.top(), [])
        backup = b.reset()
        self.assertTrue(backup.exists())
        self.assertEqual(Leaderboard(self.path).entries, [])


class TestStandFlow(unittest.TestCase):
    """Oyun akışı, ekransız (SDL dummy); rakip betikli bot (GPU gerekmez)."""

    def make_app(self, opponents=("scripted",), active_player=True):
        import pygame  # noqa: F401

        from flygame.agents import make_agent
        from flygame.agents.scripted import ScriptedAgent
        from flygame.app import App

        cfg = GameConfig()
        cfg.display.sound = False
        cfg.recording.enabled = False
        cfg.round.duration_s = 5.0
        cfg.stand.leaderboard_path = str(Path(self.dir.name) / "board.json")
        cfg.prize.eligible = ["scripted"]   # testte betikli bot ödül veren rakip
        app = App(cfg, [make_agent(o, cfg) for o in opponents], fixed_seed=3)
        if active_player:
            bot = ScriptedAgent()
            app.human.get_action = lambda st: bot.get_action(st)
        return app

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        import pygame

        pygame.quit()
        self.dir.cleanup()

    def play_round(self, app):
        import pygame

        app.on_key(pygame.K_SPACE)
        app.update(app.cfg.round.countdown_s + 0.1)
        app.draw()
        while app.state == "play":
            app.update(1 / 60)
        app.draw()
        app.update(2.5)
        app.draw()

    def test_title_screen_updates_before_any_round(self):
        # Gerileme testi: ilk turdan önce başlık ekranında update/draw çökmemeli
        app = self.make_app(("scripted", "heuristic"))
        for _ in range(3):
            app.update(app.st.attract_page_s)   # tüm çekici sayfalardan geç
            app.draw()

    def test_full_round_to_leaderboard(self):
        import pygame

        app = self.make_app()
        app.draw()
        self.play_round(app)
        self.assertEqual(app.state, "results")
        self.assertIsNotNone(app.entry_id)
        app.on_key(pygame.K_SPACE)
        self.assertEqual(app.state, "initials")
        for k in (pygame.K_UP, pygame.K_RIGHT, pygame.K_UP, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN):
            app.on_key(k)
        app.draw()
        app.on_key(pygame.K_SPACE)
        self.assertEqual(app.state, "board")
        app.draw()
        self.assertEqual(app.board.top()[0].initials, "BCZ")
        app.on_key(pygame.K_SPACE)
        self.assertEqual(app.state, "title")

    def test_idle_round_not_recorded(self):
        app = self.make_app(active_player=False)
        self.play_round(app)
        self.assertIsNone(app.entry_id)
        self.assertEqual(app.board.entries, [])

    def test_choose_opponent(self):
        import pygame

        app = self.make_app(("scripted", "heuristic"))
        app.on_key(pygame.K_SPACE)
        self.assertEqual(app.state, "choose")
        app.draw()
        app.on_key(pygame.K_RIGHT)
        app.on_key(pygame.K_SPACE)
        self.assertEqual(app.state, "countdown")
        self.assertEqual(app.opponent.effective_kind, "heuristic")

    def test_kiosk_exit_hold_and_reset(self):
        import pygame

        app = self.make_app()
        app.kiosk = True
        app.esc_since = time.monotonic() - 0.5
        self.assertFalse(app.exit_requested())
        app.esc_since = time.monotonic() - (app.st.exit_hold_s + 0.1)
        self.assertTrue(app.exit_requested())
        app.board.add(9, "scripted", "SCRIPTED BOT", 3, "win", True, 1)
        app.on_operator_key(pygame.K_F9)           # ilk basış: onay ister
        self.assertEqual(len(app.board.entries), 1)
        app.on_operator_key(pygame.K_F9)           # ikinci basış: sıfırlar
        self.assertEqual(app.board.entries, [])
        app.on_operator_key(pygame.K_F1)
        app.draw()                                  # yardım katmanı çizilebiliyor


class TestReplayPanelInfo(unittest.TestCase):
    def test_recorded_panel_is_marked_recorded(self):
        from flygame.agents.replay import ReplayAgent
        from flygame.recording import new_recording

        rec = new_recording(5, {"name": "flywire783-lif", "kind": "fly_brain",
                                "panel": {"groups": [{"en": "x", "tr": "x", "n": 2, "color": [1, 2, 3]}],
                                          "neurons": 10, "connections": 20}}, GameConfig())
        with tempfile.TemporaryDirectory() as d:
            agent = ReplayAgent(rec.save(Path(d) / "r.json.gz"))
            info = agent.panel_info()
        self.assertFalse(info["live"])
        self.assertEqual(agent.effective_kind, "fly_brain")
        self.assertIn("replay", agent.display_name("en"))


class FakeBrainAgent:
    """Beyin haritası paneli için sahte rakip: rol listeleri ve 'fired' telemetrisi verir."""

    def __init__(self, info):
        from flygame.agents.base import AgentKind

        self.kind = AgentKind.FLY_BRAIN
        self.info = info

    def panel_info(self):
        return self.info

    def display_name(self, lang="en"):
        return "FLY BRAIN"


class TestBrainMapPanel(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pygame

        from flygame import neuron_panel
        from flygame.render import Renderer
        from flygame.texts import TEXTS

        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        cfg = GameConfig()
        cfg.brain.data_dir = "fake-map-data"
        rng = np.random.default_rng(0)
        pos = np.column_stack([rng.uniform(90, 900, 5000), rng.uniform(50, 440, 5000),
                               rng.uniform(0, 280, 5000)]).astype(np.float32)
        pos[7] = np.nan                                       # konumu olmayan nöron
        neuron_panel._POSITIONS["fake-map-data"] = pos
        self.renderer = Renderer(self.screen, cfg, TEXTS["en"], side_panel=True)
        self.panel = neuron_panel.NeuronPanel(self.renderer)
        self.tx = TEXTS["en"]

    def tearDown(self):
        import pygame

        from flygame import neuron_panel

        neuron_panel._POSITIONS.pop("fake-map-data", None)
        pygame.quit()

    def test_spikes_light_up_map(self):
        import numpy as np

        info = {"neurons": 5000, "connections": 9, "live": True,
                "map": {"space": "flywire783", "fruit": [1, 2, 3], "loom": [4, 5], "decision": [6, 8]}}
        self.panel.reset(FakeBrainAgent(info), self.tx, "en")
        self.assertEqual(len(self.panel.views), 2)            # arkadan + üstten
        view = self.panel.views[0][1]
        before = view.render()
        for _ in range(5):
            self.panel.push({"spikes": 4, "ms": 16.7, "turn": -0.5, "escape": True,
                             "fired": np.array([1, 4, 6, 7, 100, 99999], np.int32)})  # 7: NaN, 99999: aralık dışı
            self.panel.update(1 / 60)
            self.panel.draw()
        self.assertTrue(view.hot and view.heat.max() > 0)
        self.assertIn(6, self.panel.pings)                    # karar nöronu halkası
        g = view.heat[view.gx[1], view.gy[1]]
        self.assertGreater(g[1], g[0])                         # meyve nöronu yeşil yanar
        for _ in range(120):                                   # girdi kesilince söner
            self.panel.update(1 / 60)
        self.assertFalse(view.hot)
        self.panel.draw()
        self.assertEqual(before.get_size(), view.surf.get_size())

    def test_old_recording_and_missing_data(self):
        from flygame import neuron_panel

        # Harita verisi olmayan eski kayıt ve indirilmemiş FlyWire verisi: panel çökmez, açıklar
        self.panel.reset(FakeBrainAgent({"groups": [], "neurons": 1, "live": False}), self.tx, "en")
        self.assertEqual(self.panel.map_error, "old")
        self.panel.draw()
        neuron_panel._POSITIONS["fake-map-data"] = None
        self.panel.reset(FakeBrainAgent({"neurons": 1, "live": True, "map": {"fruit": []}}), self.tx, "en")
        self.assertEqual(self.panel.map_error, "missing")
        self.panel.push({"fired": [1, 2], "ms": 16.7})
        self.panel.update(0.1)
        self.panel.draw()


if __name__ == "__main__":
    unittest.main()
