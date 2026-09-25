"""Ayarlama (brain-tune) ilerlemesi: günlük dosyasından yüzde ve kalan süre tahmini.

Çalışan aramayı etkilemez; yalnızca günlüğü (runs/tuning/*.log) okur.

  python -m flygame tune-status                       # en yeni günlük
  watch -n 30 .venv/bin/python -m flygame tune-status  # 30 sn'de bir kendini yeniler
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

HEADER = re.compile(r"(\d+) generations x (\d+) candidates x (\d+) seeds \(batch (\d+)\).*hold-out seeds ([\d,\-]+)")
GEN = re.compile(r"^gen\s+(\d+):.*best so far\s+([\d.]+) over (\d+) rounds.*\|\s+(\d+)s")


def _n_seeds(spec: str) -> int:
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += int(b) - int(a) + 1 if b else 1
    return n


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _search_running() -> bool:
    # Başka bir süreçte "brain-tune" çalışıyor mu? (/proc taraması; Linux)
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if b"flygame" in cmd and b"brain-tune" in cmd and b"tune-status" not in cmd:
            return True
    return False


def status(log: Path) -> str:
    lines = log.read_text().splitlines()
    head = next((m for m in map(HEADER.search, lines) if m), None)
    if head is None:
        return f"{log}: no search header yet (the run is starting)"
    G, pop, spg, batch = (int(head.group(i)) for i in range(1, 5))
    n_hold = _n_seeds(head.group(5))
    gens = [m for m in map(GEN.match, lines) if m]
    times = [int(m.group(4)) for m in gens]
    done = len(gens)
    finished = any(l.startswith("Wrote") for l in lines)
    in_control = any(l.startswith("Control") for l in lines)
    in_holdout = any(l.startswith("Hold-out") for l in lines)

    # İş birimi: bir toplu GPU turu ("batch round")
    per_gen = math.ceil(pop * spg / batch)
    hold_b = math.ceil(2 * n_hold / batch)      # varsayılan + ayarlı, ayrılmış tohumlarda
    ctrl_b = math.ceil(n_hold / batch)          # karıştırılmış bağlantı ağı kontrolü
    avg_gen = sum(times[-3:]) / len(times[-3:]) if times else None
    per_batch = avg_gen / per_gen if avg_gen else None
    since_last = time.time() - log.stat().st_mtime
    elapsed = sum(times) + (0 if finished else since_last)

    out = [f"Tuning run: {log}"]
    if gens:
        best, n = gens[-1].group(2), gens[-1].group(3)
        out.append(f"  generation {done}/{G} done ({100 * done / G:.0f}% of the search) · best so far {best} ({n} rounds)")
    if finished:
        out.append("  FINISHED. Results:")
        out += [f"    {l}" for l in lines if re.match(r"^(fly brain|tuned params|heuristic|scripted|Wrote)", l)]
        return "\n".join(out)
    if per_batch is None:
        out.append(f"  first generation running for {_fmt(since_last)} (no time estimate until it finishes)")
        return "\n".join(out)

    if in_control:
        phase, remaining = "final: shuffled-wiring control", ctrl_b * per_batch - since_last
    elif in_holdout:
        phase, remaining = "final: hold-out comparison", (hold_b + ctrl_b) * per_batch - since_last
    else:
        phase = f"search, generation {done + 1}/{G}"
        remaining = (G - done) * avg_gen - since_last + (hold_b + ctrl_b) * per_batch
    remaining = max(remaining, 60.0)  # tahmin aşıldıysa "az kaldı" göster
    pct = 100 * elapsed / (elapsed + remaining)
    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
    finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
    out.append(f"  now: {phase}")
    out.append(f"  [{bar}] ~{pct:.0f}% · elapsed {_fmt(elapsed)} · remaining ~{_fmt(remaining)} · "
               f"expected finish ~{finish}  (estimate from the last generations' speed)")
    if not _search_running():
        out.append("  WARNING: no brain-tune process is running; the search may have stopped or crashed.")
    return "\n".join(out)


def latest_log(directory: str | Path = "runs/tuning") -> Path | None:
    logs = sorted(Path(directory).glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None
