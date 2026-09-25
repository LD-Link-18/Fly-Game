"""Oyun <-> beyin arayüzü: kodlayıcı (duyusal -> Poisson hızları) ve kod çözücü
(inen nöron hızları -> eylem). Tek beyin (canlı oyun) ve toplu beyin (ayarlama)
AYNI kodu kullanır; böylece ayarlanan parametreler oyunda aynı davranır.

Sayısal parametreler (kazançlar, eşikler) üye başına ayrı olabilir: her biri
(B,) boyutlu bir dizi olarak verilir. Nöron seçimi (hangi tipler) toplu
değerlendirmede tüm üyeler için ortaktır.

İki meyve kodlaması vardır (config.brain.fruit_encoding):
  "hemifield"   : sol/sağ LC10a gruplarının hepsi aynı hızla uyarılır (yalnızca sol/sağ bilgisi).
  "retinotopic" : her LC10a nöronu, alıcı alanının yönüne (brain/retinotopy.py) meyvenin
                  ne kadar yakın olduğuna göre uyarılır: beyin meyvenin NE KADAR solda/sağda
                  olduğunu da alır.
Kod çözücü, dönüş için DNa01/DNa02'ye ek olarak yan/arka alana duyarlı DN'leri
(lateral_types) ve ileri hız için önde nesneye duyarlı DN'leri (front_types) okur
(bkz. prob sonuçları, README). Ek ağırlıklar 0 iken eski kod çözücüyle aynıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..config import BrainConfig
from .connectome import Connectome

SIDES = ("left", "right")

# Ayarlanabilir sayısal parametreler: ad -> (alt sınır, üst sınır, ölçek)
# "log" ölçekli olanlar çarpımsal olarak, "lin" olanlar toplamsal olarak aranır.
TUNABLE = {
    "fruit_gain_hz": (10.0, 800.0, "log"),
    "fruit_max_hz": (40.0, 800.0, "log"),
    "fruit_contrast": (0.0, 1.0, "lin"),
    "fruit_length": (20.0, 400.0, "log"),
    "fruit_side_width": (0.05, 1.5, "log"),
    "loom_gain_hz": (10.0, 600.0, "log"),
    "loom_max_hz": (40.0, 400.0, "log"),
    "turn_gain": (0.002, 0.2, "log"),
    "turn_deadzone_hz": (0.0, 30.0, "lin"),
    "steer_right_scale": (0.3, 4.0, "log"),
    "escape_threshold_hz": (5.0, 200.0, "log"),
    "rate_window_ms": (10.0, 300.0, "log"),
    "cruise_forward": (0.2, 1.0, "lin"),
    "escape_forward": (0.2, 1.0, "lin"),
    "turn_slowdown": (0.0, 0.9, "lin"),
    "fruit_rf_deg": (5.0, 60.0, "log"),
    "turn_lat_weight": (-2.0, 2.0, "lin"),
    "turn_front_weight": (-2.0, 2.0, "lin"),
    "speed_front_gain": (-0.02, 0.02, "lin"),
}
HEMIFIELD_ONLY = ("fruit_contrast", "fruit_side_width")
RETINOTOPIC_ONLY = ("fruit_rf_deg",)


def active_tunables(bc: BrainConfig) -> list[str]:
    """Seçili kodlamada anlamı olan (aranacak) parametreler."""
    skip = HEMIFIELD_ONLY if bc.fruit_encoding == "retinotopic" else RETINOTOPIC_ONLY
    return [k for k in TUNABLE if k not in skip]


def params_from_config(bc: BrainConfig, batch: int = 1) -> dict[str, np.ndarray]:
    return {k: np.full(batch, float(getattr(bc, k))) for k in TUNABLE}


def stack_params(param_dicts: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {k: np.array([float(d[k]) for d in param_dicts]) for k in TUNABLE}


@dataclass
class DecoderState:
    out_rates: np.ndarray   # (B, len(OUT_KEYS)) grup hızları (Hz, yumuşatılmış)
    turn: np.ndarray        # (B,) son dönüş komutu
    forward: np.ndarray     # (B,) son ileri komutu
    escaping: np.ndarray    # (B,) bool

    @classmethod
    def new(cls, batch: int, cruise: np.ndarray) -> "DecoderState":
        return cls(np.zeros((batch, len(OUT_KEYS))), np.zeros(batch), np.array(cruise, dtype=float),
                   np.zeros(batch, bool))


OUT_KEYS = (("steer", "left"), ("steer", "right"), ("escape", "left"), ("escape", "right"),
            ("front", "left"), ("front", "right"), ("lateral", "left"), ("lateral", "right"))
OPTIONAL_OUT = ("front", "lateral")  # boş liste verilebilir (okunmaz)
IN_KEYS = (("fruit", "left"), ("fruit", "right"), ("loom", "left"), ("loom", "right"))


class BrainIO:
    def __init__(self, conn: Connectome, bc: BrainConfig):
        def group(types: list[str], side: str, optional: bool = False) -> np.ndarray:
            idx = np.concatenate([conn.find(t, side) for t in types]) if types else np.zeros(0, np.int64)
            if len(idx) == 0 and not (optional and not types):
                raise ValueError(f"No neurons of types {types} on side {side} in the connectome")
            return np.unique(idx).astype(np.int64)

        types = {"fruit": bc.fruit_types, "loom": bc.loom_types, "steer": bc.steer_types,
                 "escape": bc.escape_types, "front": bc.front_types, "lateral": bc.lateral_types}
        self.encoding = bc.fruit_encoding
        if self.encoding not in ("hemifield", "retinotopic"):
            raise ValueError(f"brain.fruit_encoding must be 'hemifield' or 'retinotopic', got {self.encoding!r}")
        self.in_groups = {key: group(types[key[0]], key[1]) for key in IN_KEYS}
        self.out_groups = {key: group(types[key[0]], key[1], key[0] in OPTIONAL_OUT) for key in OUT_KEYS}
        for name, groups in (("input", self.in_groups), ("readout", self.out_groups)):
            seen: set[int] = set()
            for key, idx in groups.items():
                dup = seen.intersection(idx.tolist())
                if dup:
                    raise ValueError(f"{name} group {key} overlaps another group ({len(dup)} neurons)")
                seen.update(idx.tolist())
        self.input_idx = np.concatenate([self.in_groups[k] for k in IN_KEYS])
        self.readout_idx = np.concatenate([self.out_groups[k] for k in OUT_KEYS])
        self.in_slices = self._slices(self.in_groups, IN_KEYS)
        self.out_slices = self._slices(self.out_groups, OUT_KEYS)
        self.fruit_az = None
        if self.encoding == "retinotopic":
            self.fruit_az = self._fruit_azimuths(conn, bc)

    def _fruit_azimuths(self, conn: Connectome, bc: BrainConfig) -> np.ndarray:
        # Meyve girdi nöronlarının (önce sol, sonra sağ grup) alıcı alan yönleri (derece)
        from . import retinotopy

        az_of: dict[int, float] = {}
        for t in bc.fruit_types:
            r = retinotopy.load(conn, t, bc.data_dir, bc.eye_front_deg, bc.eye_back_deg)
            for side in SIDES:
                az_of.update(zip(r[side]["idx"].tolist(), r[side]["azimuth"].tolist()))
        idx = np.concatenate([self.in_groups[("fruit", "left")], self.in_groups[("fruit", "right")]])
        return np.array([az_of.get(int(n), np.nan) for n in idx])

    @staticmethod
    def _slices(groups, keys):
        out, pos = [], 0
        for k in keys:
            out.append(slice(pos, pos + len(groups[k])))
            pos += len(groups[k])
        return out

    # -- kodlayıcı ---------------------------------------------------------------------

    @staticmethod
    def features(states, P: dict[str, np.ndarray]) -> np.ndarray:
        """Oyun durumlarından (B, 4) duyusal özellik: fruit_L, fruit_R, loom_L, loom_R.

        Meyve "görüşü" burada, sineğin kendi bakış açısından hesaplanır (göz ve optik
        lob simüle edilmez): her meyvenin belirginliği exp(-mesafe / fruit_length);
        kısa uzunluk yakındaki meyveyi baskın yapar (tek hedefe odaklanma). Sol/sağ
        payı tüm görüş alanında tanh(yön / fruit_side_width) ile ayrılır: arkada
        soldaki meyve de net biçimde "sol" sayılır.
        Yaklaşma (loom) sinyali oyunun duyusal hesabından aynen alınır.
        """
        out = np.zeros((len(states), 4))
        for i, st in enumerate(states):
            L, w = P["fruit_length"][i], P["fruit_side_width"][i]
            for f in st.raw.fruits:
                sal = np.exp(-f.distance / L)
                wr = 0.5 * (1.0 + np.tanh(f.bearing / w))
                out[i, 0] += sal * (1.0 - wr)
                out[i, 1] += sal * wr
            out[i, 2] = st.sensory.loom_left
            out[i, 3] = st.sensory.loom_right
        return out

    def encode_states(self, states, P: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Oyun durumlarından (B, M) girdi hızları ve (B, 4) grup ortalamaları (telemetri)."""
        if self.encoding == "hemifield":
            return self.encode(self.features(states, P), P)
        B = len(states)
        rates = np.zeros((B, len(self.input_idx)), dtype=np.float32)
        g = np.zeros((B, 4))
        nf = len(self.fruit_az)
        valid = ~np.isnan(self.fruit_az)
        az = np.where(valid, self.fruit_az, 0.0)
        for i, st in enumerate(states):
            fr = st.raw.fruits
            if fr:
                b = np.degrees([f.bearing for f in fr])
                sal = np.exp(-np.array([f.distance for f in fr]) / P["fruit_length"][i])
                d = (b[:, None] - az[None, :] + 180.0) % 360.0 - 180.0       # açı farkı, -180..180
                drive = (sal[:, None] * np.exp(-0.5 * (d / P["fruit_rf_deg"][i]) ** 2)).sum(axis=0)
                r = np.minimum(P["fruit_max_hz"][i], P["fruit_gain_hz"][i] * drive) * valid
                rates[i, :nf] = r
                nl = self.in_slices[0].stop
                g[i, 0], g[i, 1] = r[:nl].mean(), r[nl:].mean()
            ll = min(P["loom_max_hz"][i], P["loom_gain_hz"][i] * st.sensory.loom_left)
            lr = min(P["loom_max_hz"][i], P["loom_gain_hz"][i] * st.sensory.loom_right)
            rates[i, self.in_slices[2]] = ll
            rates[i, self.in_slices[3]] = lr
            g[i, 2], g[i, 3] = ll, lr
        return rates, g

    def encode(self, sens: np.ndarray, P: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """sens: (B, 4) = fruit_left, fruit_right, loom_left, loom_right.

        Döndürür: (B, M) girdi nöronu hızları ve (B, 4) grup hızları (telemetri için).
        """
        fl, fr, ll, lr = sens.T
        # Kontrast: iki tarafın ortak kısmını çıkar (yalnızca fark kalsın); 0 = kapalı
        common = P["fruit_contrast"] * np.minimum(fl, fr)
        fl, fr = fl - common, fr - common
        g = np.stack([
            np.minimum(P["fruit_max_hz"], P["fruit_gain_hz"] * fl),
            np.minimum(P["fruit_max_hz"], P["fruit_gain_hz"] * fr),
            np.minimum(P["loom_max_hz"], P["loom_gain_hz"] * ll),
            np.minimum(P["loom_max_hz"], P["loom_gain_hz"] * lr),
        ], axis=1)
        rates = np.empty((sens.shape[0], len(self.input_idx)), dtype=np.float32)
        for j, sl in enumerate(self.in_slices):
            rates[:, sl] = g[:, j:j + 1]
        return rates, g

    # -- kod çözücü --------------------------------------------------------------------

    def decode(self, counts: np.ndarray, window_ms: float, P: dict[str, np.ndarray], st: DecoderState) -> None:
        """counts: (B, R) bu pencerede okuma nöronlarının spike sayıları. st yerinde güncellenir."""
        hz = np.stack([counts[:, sl].mean(axis=1) if sl.stop > sl.start else np.zeros(len(counts))
                       for sl in self.out_slices], axis=1) / (window_ms / 1000.0)
        a = 1.0 - np.exp(-window_ms / np.maximum(1e-6, P["rate_window_ms"]))
        st.out_rates += a[:, None] * (hz - st.out_rates)
        sL, sR, eL, eR, fL, fR, lL, lR = st.out_rates.T
        # Dönüş: DNa01/02 farkı + yan/arka alan DN'leri + önde nesne DN'leri (ağırlıklı)
        diff = (sR * P["steer_right_scale"] - sL) + P["turn_lat_weight"] * (lR - lL) \
            + P["turn_front_weight"] * (fR - fL)
        diff = np.where(np.abs(diff) < P["turn_deadzone_hz"], 0.0, diff)
        st.turn = np.clip(P["turn_gain"] * diff, -1.0, 1.0)
        st.escaping = np.maximum(eL, eR) >= P["escape_threshold_hz"]
        # İleri hız: elle belirlenen seyir hızı + önde nesne DN'lerinin etkisi (beyin);
        # keskin dönüşte yavaşlama ELLE belirlenmiş bir kuraldır
        base = np.clip(P["cruise_forward"] + P["speed_front_gain"] * (fL + fR) / 2.0, 0.0, 1.0)
        speed = np.where(st.escaping, P["escape_forward"], base)
        st.forward = speed * (1.0 - P["turn_slowdown"] * np.abs(st.turn))


def check_config_has_tunables() -> None:
    # Her ayarlanabilir parametrenin BrainConfig'te bir alanı olmalı
    names = {f.name for f in fields(BrainConfig)}
    missing = [k for k in TUNABLE if k not in names]
    if missing:
        raise RuntimeError(f"BrainConfig lacks tunable fields: {missing}")

