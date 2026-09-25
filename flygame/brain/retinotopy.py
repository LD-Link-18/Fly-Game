"""Görsel projeksiyon nöronlarının (ör. LC10a) alıcı alan (receptive field) yönleri.

Her LC nöronunun görsel alanda nereye baktığı, FlyWire'daki sütun (column)
atamalarından tahmin edilir:

1. FlyWire Codex "column_assignment" tablosu, 31 sütunsal hücre tipini (Tm1,
   Tm20, T2a, ...) gözün altıgen sütun ızgarasına (p, q) yerleştirir; iki göz için.
   Ters/düz yön kuralı Zhao et al. (2022) ile aynıdır: +p ön-üst, +q arka-üst.
   Buradan h = q - p (arka yönü) ve v = p + q (üst yönü) elde edilir.
2. Bir LC nöronunun merkezi = sütunu bilinen presinaptik ortaklarının (h, v)
   değerlerinin sinaps sayısıyla ağırlıklı ortalaması.
3. h, o satırın (v) ön ve arka kenarına göre 0..1'e normalize edilir ve
   gözün yatay görüş alanına (varsayılan -12° .. 155°, sağ göz; sol göz ayna)
   doğrusal olarak eşlenir. Bu bir YAKLAŞIMDIR: gerçek göz haritası doğrusal
   değildir ve ön/arka kenar açıları literatürden yuvarlak değerlerdir.

Yön kuralı oyunla aynıdır: derece, pozitif = sineğin SAĞI.
Sütunu bilinen hiç girdisi olmayan nöronların yönü NaN'dır (uyarılmazlar).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .connectome import Connectome

COLUMNS_FILE = "column_assignment.csv.gz"


def _load_columns(data_dir: Path):
    import pandas as pd  # yalnızca önbellek oluştururken gerekli

    path = data_dir / COLUMNS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python -m flygame brain-download")
    col = pd.read_csv(path)
    col["h"] = col.q - col.p
    col["v"] = col.p + col.q
    return col


def _row_extents(col, hemisphere: str):
    # Her satır (v) için gözün ön (h_min) ve arka (h_max) kenarı, Mi1 sütunlarından
    g = col[(col.type == "Mi1") & (col.hemisphere == hemisphere)]
    rows = g.groupby("v").h.agg(["min", "max"]).sort_index()
    return rows.index.to_numpy(float), rows["min"].to_numpy(float), rows["max"].to_numpy(float)


def estimate(conn: Connectome, cell_type: str, data_dir: str | Path,
             front_deg: float, back_deg: float) -> dict[str, dict[str, np.ndarray]]:
    """Her taraf için: {'idx': nöron indeksleri, 'azimuth': derece, 'elev': -1..1, 'n_syn': sinaps}."""
    data_dir = Path(data_dir)
    col = _load_columns(data_dir)
    index = {int(r): i for i, r in enumerate(conn.root_ids)}
    col = col[col.root_id.isin(index.keys())]
    H = np.full(conn.n_neurons, np.nan)
    V = np.full(conn.n_neurons, np.nan)
    ii = col.root_id.map(index).to_numpy(int)
    H[ii] = col.h.to_numpy(float)
    V[ii] = col.v.to_numpy(float)

    pre = np.repeat(np.arange(conn.n_neurons), np.diff(conn.rowptr))
    post = conn.post.astype(np.int64)
    w = np.abs(conn.weight).astype(float)
    out = {}
    for side in ("left", "right"):
        idx = np.unique(conn.find(cell_type, side))
        sel = np.isin(post, idx) & ~np.isnan(H[pre])
        pos = np.searchsorted(idx, post[sel])
        ws = np.bincount(pos, weights=w[sel], minlength=len(idx))
        with np.errstate(invalid="ignore", divide="ignore"):
            hc = np.bincount(pos, weights=w[sel] * H[pre[sel]], minlength=len(idx)) / ws
            vc = np.bincount(pos, weights=w[sel] * V[pre[sel]], minlength=len(idx)) / ws
        # Satıra göre ön/arka normalizasyonu (göz kenarı satırdan satıra değişir)
        rv, rmin, rmax = _row_extents(col, side)
        hmin = np.interp(vc, rv, rmin)
        hmax = np.interp(vc, rv, rmax)
        u = np.clip((hc - hmin) / np.maximum(1e-6, hmax - hmin), 0.0, 1.0)
        az = front_deg + u * (back_deg - front_deg)   # sağ göz
        if side == "left":
            az = -az                                   # sol göz: ayna
        elev = np.clip(vc / max(abs(rv.min()), abs(rv.max())), -1.0, 1.0)
        out[side] = {"idx": idx, "azimuth": az, "elev": elev, "n_syn": ws}
    return out


def load(conn: Connectome, cell_type: str, data_dir: str | Path,
         front_deg: float, back_deg: float) -> dict[str, dict[str, np.ndarray]]:
    """Önbellekli estimate()."""
    data_dir = Path(data_dir)
    safe = cell_type.replace(":", "_")
    path = data_dir / f"retinotopy_{safe}_{front_deg:g}_{back_deg:g}.npz"
    if path.exists():
        d = np.load(path)
        return {s: {k: d[f"{s}_{k}"] for k in ("idx", "azimuth", "elev", "n_syn")} for s in ("left", "right")}
    out = estimate(conn, cell_type, data_dir, front_deg, back_deg)
    np.savez(path, **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    return out
