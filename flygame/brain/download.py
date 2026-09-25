"""FlyWire v783 verisini indir (~136 MB), sabit sürüm ve SHA-256 doğrulamasıyla.

Kaynaklar:
  - Shiu et al. 2024 modeli: github.com/philshiu/Drosophila_brain_model
  - FlyWire hücre tipleri (Schlegel et al. 2024): github.com/flyconnectome/flywire_annotations
Dosyalar belirli commit'lere sabitlenmiştir; böylece veri sessizce değişmez.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANN = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597"

FILES = {
    "Completeness_783.csv": (f"{SHIU}/Completeness_783.csv",
                             "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    "Connectivity_783.parquet": (f"{SHIU}/Connectivity_783.parquet",
                                 "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    "neuron_annotations.tsv": (f"{ANN}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
                               "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FILES.items():
        path = data_dir / name
        if path.exists() and _sha256(path) == sha:
            print(f"ok       {name}")
            continue
        print(f"download {name} ...", flush=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink()
            raise RuntimeError(f"{name}: checksum mismatch (got {got})")
        tmp.replace(path)
        print(f"ok       {name}")
