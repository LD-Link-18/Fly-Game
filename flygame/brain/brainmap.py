"""Beyin haritası verisi: modeldeki her nöronun beyindeki konumu (FlyWire v783).

Kaynak: neuron_annotations.tsv (Schlegel et al. 2024), "pos_x/pos_y/pos_z" sütunları.
Bu, FlyWire'ın her nöron için verdiği temsilî noktadır (nöronun üzerinde bir nokta,
çoğunlukla ana dalında); birimi 4 x 4 x 40 nm vokseldir. Burada mikrometreye çevrilir.

Eksenler (veriden doğrulandı):
  x: sineğin solundan sağına (sol taraftaki nöronların x'i küçük)
  y: sırttan karına (yukarıdan aşağıya)
  z: önden arkaya (anten lobu önde, mantar cisimciği gövdeleri arkada)

Satır sırası modeldeki nöron sırasıdır (Completeness_783.csv); yani simülasyonun
spike indeksleri doğrudan bu diziye bakar. Yalnızca numpy + csv kullanılır: torch
ve pandas kurulu olmayan (yalnızca kayıt oynatan) bir stant bilgisayarında da çalışır.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

MAP_CACHE = "brain_map_783.npz"
VOXEL_UM = np.array([0.004, 0.004, 0.040])  # FlyWire voksel boyutu (µm)


def build_map_cache(data_dir: Path) -> Path:
    with open(data_dir / "Completeness_783.csv", newline="") as f:
        rows = csv.reader(f)
        next(rows)
        root_ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    order = {rid: i for i, rid in enumerate(root_ids.tolist())}
    pos = np.full((len(root_ids), 3), np.nan, dtype=np.float32)
    with open(data_dir / "neuron_annotations.tsv", newline="") as f:
        rows = csv.reader(f, delimiter="\t")
        head = next(rows)
        ci = [head.index(c) for c in ("root_id", "pos_x", "pos_y", "pos_z")]
        for r in rows:
            i = order.get(int(r[ci[0]]))
            if i is not None and np.isnan(pos[i, 0]):
                try:
                    pos[i] = [float(r[c]) for c in ci[1:]]
                except ValueError:
                    pass  # konumu olmayan birkaç nöron: haritada gösterilmez
    pos *= VOXEL_UM.astype(np.float32)
    path = data_dir / MAP_CACHE
    np.savez(path, pos_um=pos, root_ids=root_ids)
    return path


def load_positions(data_dir: Path | str) -> np.ndarray:
    """(N, 3) nöron konumları (µm), model sırasıyla; konumu bilinmeyenler NaN.

    Veri yoksa FileNotFoundError (çağıran taraf "harita verisi yok" gösterir)."""
    data_dir = Path(data_dir)
    path = data_dir / MAP_CACHE
    if not path.exists():
        missing = [f for f in ("Completeness_783.csv", "neuron_annotations.tsv") if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"FlyWire data missing in {data_dir}: {', '.join(missing)}")
        build_map_cache(data_dir)
    return np.load(path)["pos_um"]
