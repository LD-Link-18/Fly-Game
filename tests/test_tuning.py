"""Faz 4 testleri: kodlayıcı/kod çözücü, arama yardımcıları, karıştırılmış ağ kontrolü."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from flygame.config import GameConfig, load_config

try:
    import torch
except ImportError:
    torch = None


def tiny_conn():
    from flygame.brain.connectome import Connectome

    # 8 nöron: her girdi/çıktı grubundan birer tane (sol/sağ)
    types = ["LC10a", "LC10a", "LPLC2", "LPLC2", "DNa02", "DNa02", "DNp01", "DNp01"]
    sides = ["left", "right"] * 4
    return Connectome(
        root_ids=np.arange(8, dtype=np.int64),
        rowptr=np.array([0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.int64),
        post=np.array([4, 5, 5, 4], dtype=np.int32),
        weight=np.ones(4, dtype=np.float32),
        cell_type=np.array(types), hb_type=np.array([""] * 8), side=np.array(sides),
    )


class TestInterface(unittest.TestCase):
    def setUp(self):
        from flygame.brain.interface import BrainIO, params_from_config

        self.cfg = GameConfig()
        self.cfg.brain.steer_types = ["DNa02"]
        self.cfg.brain.loom_types = ["LPLC2"]
        self.cfg.brain.front_types = []    # isteğe bağlı gruplar: küçük ağda yok
        self.cfg.brain.lateral_types = []
        self.io = BrainIO(tiny_conn(), self.cfg.brain)
        self.P = params_from_config(self.cfg.brain, batch=2)

    def test_encode_contrast(self):
        sens = np.array([[1.0, 0.6, 0.0, 0.0], [1.0, 0.6, 0.0, 0.0]])
        self.P["fruit_contrast"] = np.array([0.0, 1.0])
        _, g = self.io.encode(sens, self.P)
        gain = self.cfg.brain.fruit_gain_hz
        np.testing.assert_allclose(g[0, :2], [gain * 1.0, gain * 0.6])
        np.testing.assert_allclose(g[1, :2], [gain * 0.4, 0.0], atol=1e-9)  # yalnızca fark kalır

    def test_decode_turn_sign_and_escape(self):
        from flygame.brain.interface import DecoderState

        st = DecoderState.new(2, self.P["cruise_forward"])
        # okuma sırası: steer_L, steer_R, escape_L, escape_R (her grupta 1 nöron)
        counts = np.array([[0, 10, 0, 0], [10, 0, 10, 0]], dtype=float)
        self.P["rate_window_ms"] = np.array([1e-6, 1e-6])  # yumuşatma yok
        self.io.decode(counts, 100.0, self.P, st)
        self.assertGreater(st.turn[0], 0)   # sağ DN aktif -> sağa dön
        self.assertLess(st.turn[1], 0)      # sol DN aktif -> sola dön
        self.assertFalse(st.escaping[0])
        self.assertTrue(st.escaping[1])     # dev lif 100 Hz > eşik
        self.assertEqual(st.forward[1], self.cfg.brain.escape_forward)

    def test_right_scale_balances(self):
        from flygame.brain.interface import DecoderState

        st = DecoderState.new(1, self.P["cruise_forward"][:1])
        P = {k: v[:1] for k, v in self.P.items()}
        P["rate_window_ms"] = np.array([1e-6])
        P["steer_right_scale"] = np.array([2.0])
        self.io.decode(np.array([[20, 10, 0, 0]], dtype=float), 100.0, P, st)
        self.assertEqual(st.turn[0], 0.0)   # 10 Hz * 2 = 20 Hz: dengede


class TestRetinotopicEncoding(unittest.TestCase):
    def setUp(self):
        from flygame.brain.interface import BrainIO, params_from_config

        cfg = GameConfig()
        cfg.brain.steer_types = ["DNa02"]
        cfg.brain.loom_types = ["LPLC2"]
        cfg.brain.front_types = []
        cfg.brain.lateral_types = []
        self.io = BrainIO(tiny_conn(), cfg.brain)
        # Retinotopik kodlamayı yapay alıcı alanlarla dene: sol LC10a -60°, sağ LC10a +20°
        self.io.encoding = "retinotopic"
        self.io.fruit_az = np.array([-60.0, 20.0])
        self.P = params_from_config(cfg.brain)
        self.P["fruit_rf_deg"] = np.array([20.0])
        self.cfg = cfg

    def state_with_fruit(self, bearing_deg: float):
        import math

        from flygame.world import Fruit, World
        from flygame.sensing import build_state

        w = World(self.cfg, 1)
        w.fruits.clear()
        w.swatters.clear()
        w.fly.x, w.fly.y, w.fly.heading = 400.0, 400.0, 0.0   # +x yönüne bakıyor
        b = math.radians(bearing_deg)
        w.fruits.append(Fruit(1, 400 + 100 * math.cos(b), 400 + 100 * math.sin(b), 0, 0.0))
        return build_state(w)

    def test_neuron_facing_the_fruit_is_driven_most(self):
        r_left, _ = self.io.encode_states([self.state_with_fruit(-60.0)], self.P)
        r_right, _ = self.io.encode_states([self.state_with_fruit(20.0)], self.P)
        self.assertGreater(r_left[0, 0], 10 * r_left[0, 1])     # meyve -60°: sol nöron
        self.assertGreater(r_right[0, 1], 10 * r_right[0, 0])   # meyve +20°: sağ nöron

    def test_drive_falls_off_with_angle(self):
        near, _ = self.io.encode_states([self.state_with_fruit(15.0)], self.P)
        far, _ = self.io.encode_states([self.state_with_fruit(80.0)], self.P)
        self.assertGreater(near[0, 1], far[0, 1])


class TestPopulationReadout(unittest.TestCase):
    def test_lateral_weight_adds_turn_and_front_sets_speed(self):
        from flygame.brain.connectome import Connectome
        from flygame.brain.interface import BrainIO, DecoderState, params_from_config

        types = ["LC10a", "LC10a", "LPLC2", "LPLC2", "DNa02", "DNa02", "DNp01", "DNp01",
                 "DNa03", "DNa03", "DNae002", "DNae002"]
        n = len(types)
        conn = Connectome(np.arange(n), np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32),
                          np.array(types), np.array([""] * n), np.array(["left", "right"] * (n // 2)))
        cfg = GameConfig()
        cfg.brain.steer_types, cfg.brain.loom_types = ["DNa02"], ["LPLC2"]
        cfg.brain.front_types, cfg.brain.lateral_types = ["DNa03"], ["DNae002"]
        io = BrainIO(conn, cfg.brain)
        P = params_from_config(cfg.brain)
        P["rate_window_ms"] = np.array([1e-6])
        P["turn_deadzone_hz"] = np.array([0.0])
        # okuma sırası: steer L,R, escape L,R, front L,R, lateral L,R
        counts = np.array([[0, 0, 0, 0, 5, 5, 0, 10]], dtype=float)
        st = DecoderState.new(1, P["cruise_forward"])
        io.decode(counts, 100.0, P, st)
        self.assertEqual(st.turn[0], 0.0)                      # ağırlıklar 0: eski kod çözücü
        P["turn_lat_weight"] = np.array([1.0])
        P["speed_front_gain"] = np.array([0.005])
        io.decode(counts, 100.0, P, st)
        self.assertGreater(st.turn[0], 0.0)                    # sağ yan DN -> sağa dön
        self.assertGreater(st.forward[0], cfg.brain.cruise_forward * (1 - cfg.brain.turn_slowdown))


class TestSearchHelpers(unittest.TestCase):
    def test_unit_roundtrip_and_bounds(self):
        from flygame.brain.interface import TUNABLE
        from flygame.brain.tuning import _from_unit, _to_unit, default_params

        base = default_params(GameConfig())
        back = _from_unit(_to_unit(base))
        for k in TUNABLE:
            self.assertAlmostEqual(back[k], base[k], delta=abs(base[k]) * 1e-3 + 1e-9)
        wild = _from_unit(np.full(len(TUNABLE), 5.0))  # sınır dışı -> kırpılır
        for k, (lo, hi, _) in TUNABLE.items():
            self.assertLessEqual(wild[k], hi * 1.0001)

    def test_tuned_toml_loads_as_config(self):
        from flygame.brain.tuning import default_params, params_toml

        text = params_toml(default_params(GameConfig()), ["test"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.toml"
            p.write_text(text)
            cfg = load_config(p)
        self.assertEqual(cfg.brain.turn_gain, GameConfig().brain.turn_gain)

    def test_every_tunable_is_a_config_field(self):
        from flygame.brain.interface import check_config_has_tunables

        check_config_has_tunables()


class TestShuffleControl(unittest.TestCase):
    def test_degrees_preserved_wiring_changed(self):
        from flygame.brain.connectome import Connectome, shuffled

        rng = np.random.default_rng(0)
        n, e = 50, 400
        pre = np.sort(rng.integers(0, n, e))
        rowptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))])
        post = rng.integers(0, n, e).astype(np.int32)
        c = Connectome(np.arange(n), rowptr, post, rng.random(e).astype(np.float32),
                       np.array([""] * n), np.array([""] * n), np.array([""] * n))
        s = shuffled(c, 1)
        np.testing.assert_array_equal(s.rowptr, c.rowptr)          # çıkış dereceleri
        np.testing.assert_array_equal(np.bincount(s.post, minlength=n), np.bincount(c.post, minlength=n))
        np.testing.assert_array_equal(s.weight, c.weight)
        self.assertFalse(np.array_equal(s.post, c.post))


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs CUDA")
class TestTritonBackend(unittest.TestCase):
    @staticmethod
    def net():
        # Rastgele ama sabit küçük bir ağ
        from flygame.brain.connectome import Connectome

        rng = np.random.default_rng(3)
        n, e = 2000, 40000
        pre = np.sort(rng.integers(0, n, e))
        rowptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))])
        post = rng.integers(0, n, e).astype(np.int32)
        w = rng.choice([-3.0, 1.0, 2.0, 5.0], e).astype(np.float32)
        return Connectome(np.arange(n), rowptr, post, w, np.array([""] * n), np.array([""] * n),
                          np.array([""] * n))

    def test_track_fired_member0_only(self):
        # Beyin haritası: iki arka uçta da yalnızca ilk üyenin ateşleyen nöronları listelenir
        from flygame.brain.lif_torch import LIFBrain

        c = self.net()
        inputs = np.arange(40)
        for backend in ("torch", "triton"):
            b = LIFBrain(c, inputs, np.arange(c.n_neurons), device="cuda", backend=backend, batch=2,
                         use_cuda_graph=False, track_fired=True)
            r = np.zeros((2, len(inputs)), np.float32)
            r[0, :20] = 10000.0
            r[1, 20:] = 10000.0
            b.reset(0)
            b.set_input_rates(r)
            counts = b.run(20)
            np.testing.assert_array_equal(b.last_fired, np.flatnonzero(counts[0] > 0), err_msg=backend)
            self.assertFalse(np.array_equal(counts[0] > 0, counts[1] > 0))

    def test_matches_torch_and_is_deterministic(self):
        from flygame.brain.lif_torch import LIFBrain

        # Deterministik girdi (her adım bir girdi spike'ı)
        c = self.net()
        inputs = np.arange(40)

        def spikes(backend, batch=1, member=0):
            b = LIFBrain(c, inputs, np.arange(40, 60), device="cuda", backend=backend, batch=batch,
                         use_cuda_graph=False)
            r = np.zeros((batch, len(inputs)), np.float32)
            r[member] = 10000.0
            b.reset(0)
            b.set_input_rates(r)
            counts = b.run(20)
            return counts[member], b.last_spikes[member]

        ct, st = spikes("torch")
        cx, sx = spikes("triton")
        cx2, _ = spikes("triton")
        cb, _ = spikes("triton", batch=3, member=1)
        self.assertEqual(st, sx)
        np.testing.assert_array_equal(ct, cx)
        np.testing.assert_array_equal(cx, cx2)
        np.testing.assert_array_equal(cx, cb)


if __name__ == "__main__":
    unittest.main()
