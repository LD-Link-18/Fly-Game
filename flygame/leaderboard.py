"""Stant skor tablosu ve ödül kuralı.

Her ziyaretçi turu bir kayıt olarak JSON dosyasında tutulur (varsayılan
runs/stand/leaderboard.json). Dosya her değişiklikte atomik olarak yazılır
(önce geçici dosya, sonra yeniden adlandırma), böylece elektrik kesilse veya
oyun çökse bile bozuk kalmaz. Bozuk bir dosya bulunursa yedeklenir ve boş
tabloyla devam edilir.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PrizeConfig


@dataclass(frozen=True)
class PrizeDecision:
    outcome: str   # "win" | "tie" | "loss"
    prize: bool
    reason: str    # "prize" | "not_eligible" | "below_threshold" | "no_stock" | "disabled" | "no_win"


def decide_prize(pc: PrizeConfig, visitor: int, opponent: int, opponent_kind: str,
                 prizes_today: int) -> PrizeDecision:
    """Ödül kuralı (bkz. config.PrizeConfig)."""
    outcome = "win" if visitor > opponent else "tie" if visitor == opponent else "loss"
    if not pc.enabled:
        return PrizeDecision(outcome, False, "disabled")
    if opponent_kind not in pc.eligible:
        return PrizeDecision(outcome, False, "not_eligible")
    if visitor < opponent + pc.margin or visitor < pc.min_score:
        return PrizeDecision(outcome, False, "below_threshold" if outcome == "win" else "no_win")
    if pc.max_per_day and prizes_today >= pc.max_per_day:
        return PrizeDecision(outcome, False, "no_stock")
    return PrizeDecision(outcome, True, "prize")


@dataclass
class Entry:
    id: int
    time: float
    day: str
    initials: str
    score: int
    opponent_kind: str
    opponent_name: str
    opponent_score: int
    outcome: str
    prize: bool
    seed: int


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class Leaderboard:
    def __init__(self, path: str | Path, size: int = 10, scope: str = "all"):
        self.path = Path(path)
        self.size = size
        self.scope = scope
        self.entries: list[Entry] = self._load()

    # -- dosya ---------------------------------------------------------------------

    def _load(self) -> list[Entry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [Entry(**e) for e in data["entries"]]
        except (ValueError, KeyError, TypeError) as e:
            # Bozuk dosyayı silme; yedekle ve boş tabloyla devam et
            backup = self.path.with_name(f"{self.path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
            self.path.replace(backup)
            print(f"Leaderboard file was unreadable ({e}); moved to {backup}")
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": [asdict(e) for e in self.entries]},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- değişiklikler ----------------------------------------------------------------

    def add(self, score: int, opponent_kind: str, opponent_name: str, opponent_score: int,
            outcome: str, prize: bool, seed: int, initials: str = "???") -> int:
        eid = max((e.id for e in self.entries), default=0) + 1
        self.entries.append(Entry(eid, time.time(), _today(), initials, int(score), opponent_kind,
                                  opponent_name, int(opponent_score), outcome, bool(prize), int(seed)))
        self._save()
        return eid

    def set_initials(self, eid: int, initials: str) -> None:
        for e in self.entries:
            if e.id == eid:
                e.initials = initials
        self._save()

    def reset(self) -> Path | None:
        """Tabloyu sıfırla; eski dosya zaman damgalı bir yedeğe taşınır (silinmez)."""
        backup = None
        if self.path.exists():
            backup = self.path.with_name(f"{self.path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            self.path.replace(backup)
        self.entries = []
        return backup

    # -- sorgular ---------------------------------------------------------------------

    def _scoped(self) -> list[Entry]:
        if self.scope == "today":
            today = _today()
            return [e for e in self.entries if e.day == today]
        return list(self.entries)

    def _ranked(self) -> list[Entry]:
        # Yüksek skor önce; eşitlikte önce yapan önde
        return sorted(self._scoped(), key=lambda e: (-e.score, e.time))

    def top(self) -> list[Entry]:
        return self._ranked()[:self.size]

    def rank(self, eid: int) -> int | None:
        for i, e in enumerate(self._ranked()):
            if e.id == eid:
                return i + 1
        return None

    def qualifies(self, eid: int) -> bool:
        """Bu kayıt ilk `size` içinde mi (ve 0'dan büyük mü)?"""
        r = self.rank(eid)
        e = next((e for e in self.entries if e.id == eid), None)
        return r is not None and r <= self.size and e is not None and e.score > 0

    def stats(self) -> dict[str, dict]:
        """Rakip türü başına: tur, ziyaretçi galibiyet/yenilgi/beraberlik, ortalamalar."""
        out: dict[str, dict] = {}
        for e in self._scoped():
            s = out.setdefault(e.opponent_kind, {"rounds": 0, "wins": 0, "losses": 0, "ties": 0,
                                                 "opp_sum": 0, "vis_sum": 0, "prizes": 0})
            s["rounds"] += 1
            s["wins"] += e.outcome == "win"
            s["losses"] += e.outcome == "loss"
            s["ties"] += e.outcome == "tie"
            s["opp_sum"] += e.opponent_score
            s["vis_sum"] += e.score
            s["prizes"] += e.prize
        for s in out.values():
            s["opp_avg"] = s["opp_sum"] / s["rounds"]
            s["vis_avg"] = s["vis_sum"] / s["rounds"]
        return out

    def prizes(self, today_only: bool = False) -> int:
        today = _today()
        return sum(e.prize for e in self.entries if not today_only or e.day == today)
