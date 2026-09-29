"""Çekirdek testler: belirlenimcilik, kayıt/tekrar, duyusal yön kuralları, ayarlar.

Çalıştırma: python -m unittest discover tests
"""

import math
import tempfile
import unittest
from pathlib import Path

from flygame.actions import Action
from flygame.agents.base import Agent, AgentKind
from flygame.agents.replay import ReplayAgent
from flygame.agents.heuristic import HeuristicAgent
from flygame.agents.scripted import ScriptedAgent
from flygame.config import GameConfig, config_from_dict, load_config
from flygame.recording import Recording
from flygame.runner import run_round
from flygame.schedule import make_schedule
from flygame.sensing import build_state
from flygame.world import Fruit, Swatter, World


class ConstantAgent(Agent):
    kind = AgentKind.SCRIPTED
    name = "constant"

    def __init__(self, action):
        self.action = action

    def get_action(self, state):
        return self.action


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_result(self):
        cfg = GameConfig()
        a = run_round(ScriptedAgent(), 11, cfg)
        b = run_round(ScriptedAgent(), 11, cfg)
        self.assertEqual((a.score, a.fruits, a.hits), (b.score, b.fruits, b.hits))

    def test_schedule_independent_of_player(self):
        # Meyve deseni oyuncunun hareketinden bağımsız olmalı
        cfg = GameConfig()
        w1, w2 = World(cfg, 5), World(cfg, 5)
        seen1, seen2 = [], []
        for _ in range(w1.total_ticks):
            w1.step(Action(0, 0))
            w2.step(Action(1, 1))
            seen1 += [(e.x, e.y) for e in w1.events if e.kind in ("fruit", "fruit_expired")]
            seen2 += [(e.x, e.y) for e in w2.events if e.kind in ("fruit", "fruit_expired")]
        sched = make_schedule(5, cfg)
        spawned = {(f.x, f.y) for f in sched.fruits}
        self.assertTrue(set(seen1) <= spawned and set(seen2) <= spawned)
        self.assertEqual([s.t for s in w1.schedule.swatters], [s.t for s in w2.schedule.swatters])


class TestReplay(unittest.TestCase):
    def test_roundtrip(self):
        cfg = GameConfig()
        res = run_round(ScriptedAgent(), 42, cfg, record=True)
        with tempfile.TemporaryDirectory() as d:
            path = res.recording.save(Path(d) / "r.json.gz")
            rec = Recording.load(path)
            self.assertEqual(rec.final_score, res.score)
            agent = ReplayAgent(path)
            self.assertEqual(agent.pick_seed(), 42)
            again = run_round(agent, 42, cfg)
        self.assertEqual(again.score, res.score)
        self.assertEqual(agent.kind, AgentKind.REPLAY)
        self.assertIn("scripted", agent.label("en"))

    def test_fired_spikes_roundtrip(self):
        # Beyin haritası verisi ("fired") kayıtta bit matrisine paketlenir, yüklerken geri açılır
        import numpy as np

        rec = Recording(seed=1, agent={"name": "x", "kind": "fly_brain"}, config={}, config_hash="h")
        rng = np.random.default_rng(0)
        fired = [np.sort(rng.choice(138639, int(rng.integers(0, 300)), replace=False)).astype(np.int32)
                 for _ in range(40)]
        for f in fired:
            rec.add(Action(0, 0), {"spikes": len(f), "fired": f})
        rec.add(Action(0, 0), None)
        with tempfile.TemporaryDirectory() as d:
            back = Recording.load(rec.save(Path(d) / "r.json.gz"))
        for f, t in zip(fired, back.telemetry):
            np.testing.assert_array_equal(t["fired"], f)
            self.assertEqual(t["spikes"], len(f))
        self.assertIsNone(back.telemetry[-1])
        self.assertIn("fired", rec.telemetry[0])   # kaydetmek bellekteki kaydı değiştirmez

    def test_wrong_seed_rejected(self):
        cfg = GameConfig()
        res = run_round(ScriptedAgent(), 3, cfg, record=True)
        with tempfile.TemporaryDirectory() as d:
            agent = ReplayAgent(res.recording.save(Path(d) / "r.json.gz"))
            with self.assertRaises(ValueError):
                agent.reset(4, cfg)


class TestSensing(unittest.TestCase):
    def _world(self):
        cfg = GameConfig()
        cfg.fruit.initial_count = 0
        w = World(cfg, 1)
        w.fruits.clear()
        w.swatters.clear()
        w.fly.x, w.fly.y, w.fly.heading = 400, 400, -math.pi / 2  # yukarı bakıyor
        return w

    def test_fruit_on_right(self):
        # Yukarı bakan sineğin sağı ekranda +x yönüdür
        w = self._world()
        w.fruits.append(Fruit(1, 550, 380, 0, 0.0))
        s = build_state(w).sensory
        self.assertGreater(s.nearest_fruit_bearing, 0)
        self.assertGreater(s.odor_right, s.odor_left)
        self.assertGreater(s.fruit_right, s.fruit_left)

    def test_loom_on_left_and_growing(self):
        w = self._world()
        w.swatters.append(Swatter(1, 250, 400, 85.0, 0.0, 1.3))
        w.t = 1.0
        s = build_state(w).sensory
        self.assertGreater(s.loom_left, s.loom_right)
        early = self._world()
        early.swatters.append(Swatter(1, 250, 400, 85.0, 0.0, 1.3))
        early.t = 0.3
        self.assertGreater(s.loom_left, build_state(early).sensory.loom_left)

    def test_turn_right_increases_heading(self):
        w = self._world()
        h0 = w.fly.heading
        w.step(Action(1.0, 0.0))
        self.assertGreater(w.fly.heading, h0)


class TestHeuristic(unittest.TestCase):
    def test_dodges_swatter_overhead(self):
        # Sineklik tam sineğin üstünde belirir; ajan çarpmadan kaçmalı
        cfg = GameConfig()
        w = World(cfg, 1)
        w.fruits.clear()
        w.swatters.append(Swatter(99, w.fly.x, w.fly.y, cfg.swatter.radius, 0.0, cfg.swatter.loom_s))
        agent = HeuristicAgent()
        agent.reset(1, cfg)
        while w.t < cfg.swatter.loom_s + 0.1:
            w.step(agent.get_action(build_state(w)))
        self.assertEqual(w.hits, 0)

    def test_beats_scripted_on_average(self):
        cfg = GameConfig()
        seeds = range(1, 9)
        h = sum(run_round(HeuristicAgent(), s, cfg).score for s in seeds)
        b = sum(run_round(ScriptedAgent(), s, cfg).score for s in seeds)
        self.assertGreater(h, b)

    def test_handicap_lowers_score(self):
        cfg = GameConfig()
        weak = GameConfig()
        weak.heuristic.speed_factor = 0.5
        weak.heuristic.reaction_s = 0.9
        seeds = range(1, 6)
        full = sum(run_round(HeuristicAgent(), s, cfg).score for s in seeds)
        slow = sum(run_round(HeuristicAgent(), s, weak).score for s in seeds)
        self.assertGreater(full, slow)


class TestConfig(unittest.TestCase):
    def test_override_and_typo(self):
        cfg = config_from_dict({"swatter": {"loom_s": 1}})
        self.assertEqual(cfg.swatter.loom_s, 1.0)
        with self.assertRaises(ValueError):
            config_from_dict({"swatter": {"loom": 1.0}})

    def test_display_does_not_change_gameplay_hash(self):
        a, b = GameConfig(), GameConfig()
        b.display.language = "tr"
        self.assertEqual(a.gameplay_hash(), b.gameplay_hash())
        b.fly.max_speed = 1.0
        self.assertNotEqual(a.gameplay_hash(), b.gameplay_hash())

    def test_example_configs_load(self):
        for p in Path(__file__).resolve().parent.parent.glob("configs/*.toml"):
            load_config(p)


if __name__ == "__main__":
    unittest.main()
