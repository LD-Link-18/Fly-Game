"""Sinek beyni rakibi: tüm FlyWire v783 beyninin canlı LIF simülasyonu.

Her oyun adımında:
  1) Kodlama: meyve görüşü (bkz. BrainIO.features) -> sol/sağ LC10a (küçük nesne algılayıcı)
     nöronlarına, sensory.loom_left/right -> sol/sağ LPLC2+LC4 (yaklaşma
     algılayıcı) nöronlarına Poisson girdi hızı. Göz ve optik lob hesabı
     ATLANIR: bu sinyalleri oyun hesaplar ve doğrudan bu nöronlara verilir.
  2) Simülasyon: beyin, oyun zamanına yetişecek kadar (1.8 ms'lik bloklar) ilerler.
  3) Kod çözme: sağ ve sol DNa01/DNa02 (aynı tarafa dönüşü başlatan inen nöronlar)
     ateşleme farkı -> dönüş; dev lif (DNp01) -> kaçış hızı.
     İleri hız ELLE belirlenmiş bir kuraldır: sabit seyir hızı, beyin keskin dönüş
     komutu verdikçe yavaşlar (bkz. config.brain.cruise_forward, turn_slowdown).

Hangi nöronların seçileceği, kazançlar ve eşikler config.brain içindedir.
Eşlemeler elle seçilmiştir (Eon Systems çalışmasındaki gibi); öğrenilmemiştir.
"""

from __future__ import annotations

import time

import numpy as np

from ..actions import Action
from ..config import BrainConfig, GameConfig
from ..sensing import GameState
from .base import Agent, AgentKind

NO_SPIKES = np.zeros(0, dtype=np.int32)
MAP_SPACE = "flywire783"  # spike indeksleri bu nöron sırasına göredir (Completeness_783.csv)


class FlyBrainAgent(Agent):
    kind = AgentKind.FLY_BRAIN
    name = "flywire783-lif"

    def __init__(self, cfg: GameConfig):
        from ..brain.connectome import load_connectome
        from ..brain.interface import BrainIO
        from ..brain.lif_torch import LIFBrain, LIFParams

        self.bc: BrainConfig = cfg.brain
        conn = load_connectome(self.bc.data_dir)
        self.n_neurons, self.n_connections = conn.n_neurons, conn.n_connections
        # Kodlayıcı/kod çözücü ayarlama aracıyla ortaktır (bkz. brain/interface.py)
        self.io = BrainIO(conn, self.bc)
        # track_fired: her adımda tüm beyinde ateşleyen nöronlar beyin haritası için
        # okunur (kararı etkilemez; yalnızca gösterim)
        self.brain = LIFBrain(conn, self.io.input_idx, self.io.readout_idx, device=self.bc.device,
                              params=LIFParams(dt=self.bc.dt_ms), track_fired=True)
        # Isınma: çekirdek derleme ve spike dağıtım yolunun ilk kullanımı burada olsun,
        # oyun sırasında takılmasın (girdi olmadan spike olmaz, dağıtım yolu çalışmaz)
        self.brain.set_input_rates(np.full(len(self.io.input_idx), 150.0, np.float32))
        self.brain.run(20)
        self.reset(0, cfg)

    def reset(self, seed: int, cfg: GameConfig) -> None:
        from ..brain.interface import DecoderState, params_from_config

        self.bc = cfg.brain
        self.P = params_from_config(self.bc)
        self.dec = DecoderState.new(1, self.P["cruise_forward"])
        self.in_rates = np.zeros(4)
        self.fired = NO_SPIKES
        self.tick_ms = 0.0
        self.brain.reset(seed)
        self.game_ms = 0.0
        self.wall_s = 0.0
        self.sim_ms = 0.0

    def get_action(self, state: GameState) -> Action:
        t0 = time.perf_counter()
        rates, g = self.io.encode_states([state], self.P)
        self.in_rates = g[0]
        self.brain.set_input_rates(rates)
        # Beyni oyun zamanına yetiştir (bu adımın süresi kadar)
        self.game_ms += state.raw.dt * 1000.0
        block = self.brain.block_ms
        n_blocks = int((self.game_ms - self.brain.time_ms) / block + 1e-9)
        self.fired, self.tick_ms = NO_SPIKES, 0.0
        if n_blocks > 0:
            counts = self.brain.run(n_blocks)
            self.io.decode(counts, n_blocks * block, self.P, self.dec)
            self.sim_ms += n_blocks * block
            self.tick_ms = n_blocks * block
            self.fired = self.brain.last_fired  # bu adımda ateşleyen tüm nöronlar (harita)
        self.wall_s += time.perf_counter() - t0
        return Action(float(self.dec.turn[0]), float(self.dec.forward[0]))

    @property
    def realtime_factor(self) -> float:
        # > 1 ise beyin gerçek zamandan hızlı çalışıyor
        return (self.sim_ms / 1000.0) / self.wall_s if self.wall_s > 0 else float("inf")

    def telemetry(self) -> dict:
        from ..brain.interface import IN_KEYS, OUT_KEYS

        r = lambda x: round(float(x), 1)  # noqa: E731 - kayıt dosyası küçük kalsın
        return {
            "in": {f"{c}_{sd[0].upper()}": r(v) for (c, sd), v in zip(IN_KEYS, self.in_rates)},
            "out": {f"{c}_{sd[0].upper()}": r(v) for (c, sd), v in zip(OUT_KEYS, self.dec.out_rates[0])},
            "spikes": int(self.brain.last_spikes[0]),
            "ms": round(self.tick_ms, 2),
            "escape": bool(self.dec.escaping[0]),
            "turn": round(float(self.dec.turn[0]), 3),
            "fwd": round(float(self.dec.forward[0]), 3),
            "fired": self.fired,  # kayıtta sıkıştırılarak saklanır (bkz. recording.py)
        }

    # -- nöron paneli (beyin haritası) ---------------------------------------------------

    def panel_info(self) -> dict:
        # Haritada renklendirilecek nöronlar (model indeksleri): girdi verilen göz nöronları
        # ve kararın okunduğu inen nöronlar. Diğer tüm nöronlar da ateşledikçe haritada yanar.
        io = self.io
        roles = {
            "fruit": np.concatenate([io.in_groups[("fruit", sd)] for sd in ("left", "right")]),
            "loom": np.concatenate([io.in_groups[("loom", sd)] for sd in ("left", "right")]),
            "decision": io.readout_idx,
        }
        return {"neurons": self.n_neurons, "connections": self.n_connections, "live": True,
                "map": {"space": MAP_SPACE, **{k: v.astype(int).tolist() for k, v in roles.items()}}}

    def ui_note(self, lang: str = "en") -> str:
        # Dürüstlük: beynin neyi kontrol ettiği, neyin elle belirlendiği arayüzde yazılır
        n = f"{self.n_neurons:,}".replace(",", "." if lang == "tr" else ",")
        brain_speed = self.bc.speed_front_gain != 0.0
        if lang == "tr":
            speed = "hız: elle taban + beyin" if brain_speed else "hız: elle kural"
            return f"{n} nöron (FlyWire v783) · dönüş ve kaçış: beyin · {speed}"
        speed = "speed: hand-set base + brain" if brain_speed else "speed: hand-set rule"
        return f"{n} neurons (FlyWire v783) · turning & escape: brain · {speed}"

    def describe(self) -> dict:
        d = super().describe()
        d["brain"] = {
            "connectome": "FlyWire v783 (Shiu et al. 2024 LIF)",
            "neurons": self.n_neurons,
            "connections": self.n_connections,
            "config": dict(self.bc.__dict__),
        }
        d["panel"] = {k: v for k, v in self.panel_info().items() if k != "live"}
        return d
