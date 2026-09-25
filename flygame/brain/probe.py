"""Yanallaşma (lateralization) probu: sol/sağ duyusal uyarıma inen nöron yanıtları.

Her duyusal grubu (ör. sol LC10a) dinlenme durumundan 150 Hz Poisson girdiyle
400 ms uyarır ve okuma nöronlarının (sol/sağ) ortalama ateşleme hızını yazar.
Seçilen nöronlar ve GPU modeli doğru mu, hızlıca kontrol etmek için kullanılır.

Brian2 referansı (Shiu et al. modeli, aynı veri, dt = 0.1 ms, 400 ms, 150 Hz,
10 deneme ortalaması ± std; FlyWire tip adları). GPU aktarımı bu değerlerle
istatistiksel olarak uyumludur ve deterministik girdide spike'ı spike'ına aynıdır:
  LPLC2+LC4 sol  -> DNa01 R 41 ± 4,   DNa02 R  9 ± 3,   DNp01 L 168 ± 3 / R 108 ± 3
  LPLC2+LC4 sağ  -> DNa01 L 39 ± 4,   DNa02 L 26 ± 4,   DNp01 L  98 ± 3 / R 187 ± 4
  LC10a sol      -> DNa02 L 133 ± 6,  DNa01 L 18 ± 3
  LC10a sağ      -> DNa02 R  84 ± 9,  DNa01 R  9 ± 2
"""

from __future__ import annotations

import time

import numpy as np

from .connectome import load_connectome
from .lif_torch import LIFBrain, LIFParams

DEFAULT_STIM = [["LPLC2", "LC4"], ["LC10a"], ["LC16"]]
DEFAULT_READ = ["DNa01", "hb:DNa01", "DNa02", "DNp01", "MDN", "DNp09"]


def run_probe(data_dir: str, device: str, stim_sets=DEFAULT_STIM, read_types=DEFAULT_READ,
              rate_hz: float = 150.0, dur_ms: float = 400.0, dt_ms: float = 0.1, seed: int = 0) -> None:
    conn = load_connectome(data_dir)
    sides = ("left", "right")
    groups = {(tuple(ts), s): np.unique(np.concatenate([conn.find(t, s) for t in ts]))
              for ts in stim_sets for s in sides}
    input_idx = np.unique(np.concatenate(list(groups.values())))
    pos = {n: i for i, n in enumerate(input_idx)}
    reads = {(t, s): conn.find(t, s) for t in read_types for s in sides}
    readout_idx = np.concatenate(list(reads.values()))
    brain = LIFBrain(conn, input_idx, readout_idx, device=device, params=LIFParams(dt=dt_ms), seed=seed)
    n_blocks = int(round(dur_ms / brain.block_ms))

    cols = [f"{t}_{s[0].upper()}" for t, s in reads]
    print(f"{'stimulus (' + str(int(rate_hz)) + ' Hz)':24}" + "".join(f"{c:>11}" for c in cols) + "   wall")
    conds = [("baseline", None)] + [(f"{'+'.join(ts)} {s}", (tuple(ts), s)) for ts in stim_sets for s in sides]
    for name, key in conds:
        brain.reset(seed)
        rates = np.zeros(len(input_idx), np.float32)
        if key is not None:
            rates[[pos[n] for n in groups[key]]] = rate_hz
        brain.set_input_rates(rates)
        t0 = time.perf_counter()
        counts = brain.run(n_blocks)[0]
        wall = time.perf_counter() - t0
        secs = n_blocks * brain.block_ms / 1000.0
        out, i = [], 0
        for k, idx in reads.items():
            out.append(counts[i:i + len(idx)].mean() / secs if len(idx) else float("nan"))
            i += len(idx)
        print(f"{name:24}" + "".join(f"{v:11.1f}" for v in out) + f"   {wall:.2f}s ({secs / wall:.2f}x realtime)")
