"""Sinek beyni rakibi: tüm FlyWire v783 beyninin canlı LIF simülasyonu.

Her oyun adımında:
  1) Kodlama: sensory.fruit_left/right -> sol/sağ LC10a (küçük nesne algılayıcı)
     nöronlarına, sensory.loom_left/right -> sol/sağ LPLC2+LC4 (yaklaşma
     algılayıcı) nöronlarına Poisson girdi hızı. Göz ve optik lob hesabı
     ATLANIR: bu sinyalleri oyun hesaplar ve doğrudan bu nöronlara verilir.
  2) Simülasyon: beyin, oyun zamanına yetişecek kadar (1.8 ms'lik bloklar) ilerler.
  3) Kod çözme: sağ ve sol DNa01/DNa02 (aynı tarafa dönüşü başlatan inen nöronlar)
     ateşleme farkı -> dönüş; dev lif (DNp01) -> kaçış hızı.
     İleri seyir hızı ELLE belirlenmiş sabittir (bkz. config.brain.cruise_forward).

Hangi nöronların seçileceği, kazançlar ve eşikler config.brain içindedir.
Eşlemeler elle seçilmiştir (Eon Systems çalışmasındaki gibi); öğrenilmemiştir.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..actions import Action
from ..config import BrainConfig, GameConfig
from ..sensing import GameState
from .base import Agent, AgentKind

SIDES = ("left", "right")


class FlyBrainAgent(Agent):
    kind = AgentKind.FLY_BRAIN
    name = "flywire783-lif"

    def __init__(self, cfg: GameConfig):
        from ..brain.connectome import load_connectome
        from ..brain.lif_torch import LIFBrain, LIFParams

        self.bc: BrainConfig = cfg.brain
        bc = self.bc
        conn = load_connectome(bc.data_dir)
        self.n_neurons, self.n_connections = conn.n_neurons, conn.n_connections

        def group(types: list[str], side: str) -> np.ndarray:
            idx = np.concatenate([conn.find(t, side) for t in types]) if types else np.zeros(0, np.int64)
            if len(idx) == 0:
                raise ValueError(f"No neurons of types {types} on side {side} in the connectome")
            return np.unique(idx)

        # Girdi grupları: (kanal, taraf) -> nöron indeksleri
        self.in_groups = {(ch, s): group(types, s)
                          for ch, types in (("fruit", bc.fruit_types), ("loom", bc.loom_types))
                          for s in SIDES}
        # Aynı nöron birden fazla girdi grubunda olmasın (hızlar karışmasın)
        seen: set[int] = set()
        for key, idx in self.in_groups.items():
            dup = seen.intersection(idx.tolist())
            if dup:
                raise ValueError(f"Input group {key} overlaps another group ({len(dup)} neurons)")
            seen.update(idx.tolist())
        input_idx = np.concatenate(list(self.in_groups.values()))
        self._in_slices = {}
        pos = 0
        for key, idx in self.in_groups.items():
            self._in_slices[key] = slice(pos, pos + len(idx))
            pos += len(idx)

        # Çıktı grupları
        self.out_groups = {(ch, s): group(types, s)
                           for ch, types in (("steer", bc.steer_types), ("escape", bc.escape_types))
                           for s in SIDES}
        readout_idx = np.concatenate(list(self.out_groups.values()))
        self._out_slices = {}
        pos = 0
        for key, idx in self.out_groups.items():
            self._out_slices[key] = slice(pos, pos + len(idx))
            pos += len(idx)

        self.brain = LIFBrain(conn, input_idx, readout_idx, device=bc.device,
                              params=LIFParams(dt=bc.dt_ms))
        self._rates_in = np.zeros(len(input_idx), dtype=np.float32)
        # Isınma: CUDA grafiği yakalama ve derleme burada olsun, oyun sırasında takılmasın
        self.brain.run(2)
        self.reset(0, cfg)

    def reset(self, seed: int, cfg: GameConfig) -> None:
        self.brain.reset(seed)
        self.game_ms = 0.0
        self.out_rates = {k: 0.0 for k in self.out_groups}
        self.in_rates = {k: 0.0 for k in self.in_groups}
        self.last_turn = 0.0
        self.last_forward = self.bc.cruise_forward
        self.escaping = False
        self.wall_s = 0.0
        self.sim_ms = 0.0

    # -- kodlama / kod çözme --------------------------------------------------------

    def _encode(self, state: GameState) -> None:
        s, bc = state.sensory, self.bc
        vals = {
            ("fruit", "left"): min(bc.fruit_max_hz, bc.fruit_gain_hz * s.fruit_left),
            ("fruit", "right"): min(bc.fruit_max_hz, bc.fruit_gain_hz * s.fruit_right),
            ("loom", "left"): min(bc.loom_max_hz, bc.loom_gain_hz * s.loom_left),
            ("loom", "right"): min(bc.loom_max_hz, bc.loom_gain_hz * s.loom_right),
        }
        for key, rate in vals.items():
            self._rates_in[self._in_slices[key]] = rate
        self.in_rates = vals
        self.brain.set_input_rates(self._rates_in)

    def _decode(self, counts: np.ndarray, window_ms: float) -> Action:
        bc = self.bc
        # Grup başına ortalama ateşleme hızı (Hz), üstel ortalamayla yumuşatılmış
        a = 1.0 - math.exp(-window_ms / max(1e-6, bc.rate_window_ms))
        for key, sl in self._out_slices.items():
            hz = counts[sl].mean() / (window_ms / 1000.0)
            self.out_rates[key] += a * (hz - self.out_rates[key])
        diff = self.out_rates[("steer", "right")] - self.out_rates[("steer", "left")]
        if abs(diff) < bc.turn_deadzone_hz:
            diff = 0.0
        turn = max(-1.0, min(1.0, bc.turn_gain * diff))
        esc = max(self.out_rates[("escape", "left")], self.out_rates[("escape", "right")])
        self.escaping = bool(esc >= bc.escape_threshold_hz)
        forward = bc.escape_forward if self.escaping else bc.cruise_forward
        self.last_turn, self.last_forward = turn, forward
        return Action(turn, forward)

    # -- ana döngü --------------------------------------------------------------------

    def get_action(self, state: GameState) -> Action:
        t0 = time.perf_counter()
        self._encode(state)
        # Beyni oyun zamanına yetiştir (bu adımın süresi kadar)
        self.game_ms += state.raw.dt * 1000.0
        block = self.brain.block_ms
        n_blocks = int((self.game_ms - self.brain.time_ms) / block + 1e-9)
        if n_blocks <= 0:
            self.wall_s += time.perf_counter() - t0
            return Action(self.last_turn, self.last_forward)
        counts = self.brain.run(n_blocks)
        action = self._decode(counts, n_blocks * block)
        self.sim_ms += n_blocks * block
        self.wall_s += time.perf_counter() - t0
        return action

    @property
    def realtime_factor(self) -> float:
        # > 1 ise beyin gerçek zamandan hızlı çalışıyor
        return (self.sim_ms / 1000.0) / self.wall_s if self.wall_s > 0 else float("inf")

    def telemetry(self) -> dict:
        r = lambda x: round(float(x), 1)  # noqa: E731 - kayıt dosyası küçük kalsın
        return {
            "in": {f"{c}_{s[0].upper()}": r(v) for (c, s), v in self.in_rates.items()},
            "out": {f"{c}_{s[0].upper()}": r(v) for (c, s), v in self.out_rates.items()},
            "spikes": int(self.brain.last_spikes),
            "escape": bool(self.escaping),
        }

    def ui_note(self, lang: str = "en") -> str:
        # Dürüstlük: beynin neyi kontrol ettiği, neyin elle sabitlendiği arayüzde yazılır
        n = f"{self.n_neurons:,}".replace(",", "." if lang == "tr" else ",")
        if lang == "tr":
            return f"{n} nöron (FlyWire v783) · dönüş ve kaçış: beyin · seyir hızı: elle sabit"
        return f"{n} neurons (FlyWire v783) · turning & escape: brain · cruise speed: hand-set"

    def describe(self) -> dict:
        d = super().describe()
        d["brain"] = {
            "connectome": "FlyWire v783 (Shiu et al. 2024 LIF)",
            "neurons": self.n_neurons,
            "connections": self.n_connections,
            "config": dict(self.bc.__dict__),
        }
        return d
