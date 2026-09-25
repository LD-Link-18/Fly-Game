"""FlyWire v783 konnektom verisinin yüklenmesi ve önbelleğe alınması.

Ham veri (bkz. README, "Fly brain" bölümü):
  data/flywire/Completeness_783.csv       - modeldeki nöronlar (Shiu et al. 2024)
  data/flywire/Connectivity_783.parquet   - bağlantılar ve işaretli sinaps sayıları
  data/flywire/neuron_annotations.tsv     - hücre tipleri ve taraf (Schlegel et al. 2024)

Ham dosyaların okunması pandas + parquet motoru gerektirir ve ~20 sn sürer.
Bu yüzden ilk seferde sıkıştırılmamış bir .npz önbelleği yazılır; oyun
sırasında yalnızca numpy ile bu önbellek okunur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/flywire")
CACHE_NAME = "connectome_783_cache_v2.npz"


@dataclass
class Connectome:
    root_ids: np.ndarray   # (N,) int64, FlyWire kimlikleri; model indeksi = satır sırası
    rowptr: np.ndarray     # (N+1,) int64, presinaptik nörona göre CSR satır başlangıçları
    post: np.ndarray       # (E,) int32, postsinaptik indeks (pre'ye göre sıralı)
    weight: np.ndarray     # (E,) float32, işaretli sinaps sayısı ("Excitatory x Connectivity")
    cell_type: np.ndarray  # (N,) str, FlyWire konsensüs tipi; bilinmiyorsa ""
    hb_type: np.ndarray    # (N,) str, hemibrain tipi (bazı DN isimleri FlyWire'dan farklı!)
    side: np.ndarray       # (N,) str, "left" / "right" / ""

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_connections(self) -> int:
        return len(self.post)

    def find(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Hücre tipine (ve isteğe bağlı tarafa) göre model indekslerini döndür.

        Varsayılan FlyWire tipidir. "hb:" öneki hemibrain tipini seçer; örn.
        FlyWire "DNa01" = hemibrain "VES006", hemibrain "DNa01" = FlyWire "DNae001".
        """
        if cell_type.startswith("hb:"):
            m = self.hb_type == cell_type[3:]
        else:
            m = self.cell_type == cell_type
        if side is not None:
            m &= self.side == side
        return np.flatnonzero(m)


def build_cache(data_dir: Path = DATA_DIR) -> Path:
    import pandas as pd  # yalnızca önbellek oluştururken gerekli

    comp = pd.read_csv(data_dir / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(
        data_dir / "Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    ann = pd.read_csv(
        data_dir / "neuron_annotations.tsv", sep="\t", low_memory=False,
        usecols=["root_id", "cell_type", "hemibrain_type", "side"],
    )
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = len(root_ids)

    # İki isimlendirme ayrı saklanır: FlyWire tipi ve hemibrain tipi
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(root_ids)
    ctype = ann["cell_type"].fillna("").astype(str).to_numpy()
    hbtype = ann["hemibrain_type"].fillna("").astype(str).to_numpy()
    side = ann["side"].fillna("").astype(str).to_numpy()

    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = con["Postsynaptic_Index"].to_numpy(np.int32)[order]
    weight = con["Excitatory x Connectivity"].to_numpy(np.float32)[order]
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(rowptr, pre + 1, 1)
    rowptr = np.cumsum(rowptr)

    path = data_dir / CACHE_NAME
    np.savez(path, root_ids=root_ids, rowptr=rowptr, post=post, weight=weight,
             cell_type=ctype.astype("U"), hb_type=hbtype.astype("U"), side=side.astype("U"))
    return path


def load_connectome(data_dir: Path | str = DATA_DIR) -> Connectome:
    data_dir = Path(data_dir)
    path = data_dir / CACHE_NAME
    if not path.exists():
        missing = [f for f in ("Completeness_783.csv", "Connectivity_783.parquet", "neuron_annotations.tsv")
                   if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"FlyWire data missing in {data_dir}: {', '.join(missing)}. "
                "See README section 'Fly brain' for download commands."
            )
        print("Building connectome cache (one time, ~30 s)...")
        build_cache(data_dir)
    d = np.load(path)
    return Connectome(d["root_ids"], d["rowptr"], d["post"], d["weight"], d["cell_type"], d["hb_type"], d["side"])
