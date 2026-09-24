"""Sezgisel rakip (Faz 2): meyve arar, yaklaşan sinekliklerden kaçar.

SİNEK BEYNİ DEĞİLDİR. Elle yazılmış kurallar ve kısa vadeli bir planlayıcı;
sinek beyni yavaş/başarısız olursa güvenilir yedek rakip olarak kullanılır.
Arayüzde "HEURISTIC BOT" olarak etiketlenir.

Nasıl çalışır:
  1) Hedef seçimi: dönüş süresi dahil tahmini varış süresi en kısa olan ve
     vardığımızda hâlâ duracak meyve. Titremeyi önlemek için mevcut hedef,
     yenisi belirgin şekilde daha iyi değilse korunur.
  2) Tehdit yoksa: hedefe dön ve ilerle.
  3) Çarpmamış bir sineklik varsa: birkaç aday hareketi (hedefe devam, 16 kaçış
     yönü, fren) oyunun kendi hareket fiziğiyle ileriye doğru simüle et, her
     sinekliğin çarpma anında nerede olacağımıza bak ve en düşük maliyetli
     adayı seç. Plan birkaç adımda bir yenilenir.

Ajan yalnızca ham durumu (raw) ve oyunun herkese açık ayarlarını kullanır;
gelecekteki sinekliklerin zamanını (tur desenini) bilmez.

Zorluk: config.heuristic (speed_factor, reaction_s) ile zayıflatılabilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..actions import Action
from ..config import GameConfig
from ..sensing import GameState, RawState
from ..world import FlyBody, move_fly, wrap_angle
from .base import Agent, AgentKind


@dataclass
class HeuristicParams:
    # Yönlendirme
    turn_gain: float = 3.5            # açı hatası (rad) -> dönüş komutu
    slow_turn_deg: float = 55.0       # hedef bu açıdan fazla yandaysa yavaşla
    slow_forward: float = 0.35        # yavaşlarken ileri hız
    switch_ratio: float = 0.75        # yeni hedef, eskisinin bu oranı kadar sürede ulaşılırsa hedef değiştir
    fruit_expiry_slack_s: float = 0.2  # varıştan sonra en az bu kadar ömrü kalan meyveler
    # Kaçınma planlayıcısı
    headings: int = 16                # denenen kaçış yönü sayısı
    replan_ticks: int = 4             # tehlike varken kaç adımda bir yeniden plan
    plan_dt_ticks: int = 2            # planlamada kaç oyun adımı birleştirilir (hız için)
    safety_margin: float = 14.0       # vuruş yarıçapına eklenen güvenlik payı
    hit_cost: float = 100.0
    near_cost: float = 1.5            # güvenlik payı içindeki her birim için maliyet
    fruit_reward: float = 6.0         # plan sırasında toplanan meyve başına ödül
    eta_cost: float = 2.0             # plan sonunda hedefe kalan süre (sn) başına maliyet
    wall_cost: float = 3.0            # plan sonunda duvara yakınlık maliyeti
    switch_cost: float = 0.5          # önceki plandan farklı bir aday seçme maliyeti


class HeuristicAgent(Agent):
    kind = AgentKind.HEURISTIC
    name = "heuristic"

    def __init__(self, params: HeuristicParams | None = None):
        self.p = params or HeuristicParams()
        self.cfg: GameConfig | None = None
        self._target: int | None = None
        self._plan: tuple | None = None
        self._plan_age = 0

    def reset(self, seed: int, cfg: GameConfig) -> None:
        self.cfg = cfg
        self._target = None
        self._plan = None
        self._plan_age = 0

    # -- yardımcılar ------------------------------------------------------------

    def _eta(self, fx: float, fy: float, heading: float, x: float, y: float) -> float:
        # Dönüş + düz uçuş süresiyle kaba varış tahmini
        fc = self.cfg.fly
        d = math.hypot(x - fx, y - fy)
        b = abs(wrap_angle(math.atan2(y - fy, x - fx) - heading))
        return b / math.radians(fc.turn_rate_deg) + d / fc.max_speed

    def _pick_target(self, raw: RawState) -> None:
        f = raw.fly
        best_id, best_eta, cur_eta = None, math.inf, math.inf
        for fr in raw.fruits:
            eta = self._eta(f.x, f.y, f.heading, fr.x, fr.y)
            if eta + self.p.fruit_expiry_slack_s > fr.time_left:
                continue
            # Çarpmak üzere olan bir sinekliğin altındaki meyveyi ertele
            for s in raw.swatters:
                if not s.impacted and math.hypot(fr.x - s.x, fr.y - s.y) < s.radius and eta < s.time_to_impact + 0.2:
                    eta += 1.0
            if fr.id == self._target:
                cur_eta = eta
            if eta < best_eta:
                best_id, best_eta = fr.id, eta
        if self._target is None or cur_eta == math.inf or best_eta < cur_eta * self.p.switch_ratio:
            self._target = best_id

    def _target_pos(self, raw: RawState) -> tuple[float, float] | None:
        for fr in raw.fruits:
            if fr.id == self._target:
                return fr.x, fr.y
        return None

    def _steer(self, fly: FlyBody, tx: float, ty: float) -> tuple[float, float]:
        err = wrap_angle(math.atan2(ty - fly.y, tx - fly.x) - fly.heading)
        turn = max(-1.0, min(1.0, self.p.turn_gain * err))
        forward = 1.0 if abs(err) < math.radians(self.p.slow_turn_deg) else self.p.slow_forward
        return turn, forward

    def _steer_heading(self, fly: FlyBody, h: float) -> float:
        return max(-1.0, min(1.0, self.p.turn_gain * wrap_angle(h - fly.heading)))

    # -- planlayıcı -----------------------------------------------------------------

    def _candidates(self, raw: RawState):
        # ("seek",) hedefe devam; ("head", açı) o yöne tam hız; ("brake",) dur
        cands = [("seek",), ("brake",)]
        for i in range(self.p.headings):
            cands.append(("head", 2 * math.pi * i / self.p.headings))
        return cands

    def _control(self, cand, fly: FlyBody, target) -> tuple[float, float]:
        turn, fwd = self._raw_control(cand, fly, target)
        return turn, fwd * self.cfg.heuristic.speed_factor

    def _raw_control(self, cand, fly: FlyBody, target) -> tuple[float, float]:
        if cand[0] == "seek":
            if target is None:
                return 0.0, 0.0
            return self._steer(fly, *target)
        if cand[0] == "brake":
            return 0.0, 0.0
        return self._steer_heading(fly, cand[1]), 1.0

    def _rollout_cost(self, cand, raw: RawState, threats, target) -> float:
        cfg, p = self.cfg, self.p
        f = raw.fly
        fly = FlyBody(f.x, f.y, f.heading, f.vx, f.vy)
        dt = raw.dt * p.plan_dt_ticks
        horizon = max(t.time_to_impact for t in threats) + dt
        stun = f.stun_left
        reach2 = (cfg.fly.radius + cfg.fruit.radius) ** 2
        eaten: set[int] = set()
        pending = sorted(threats, key=lambda s: s.time_to_impact)
        cost = 0.0
        t = 0.0
        while t < horizon:
            if stun > 0:
                turn, fwd = 0.0, 0.0
                stun -= dt
            else:
                turn, fwd = self._control(cand, fly, target)
            move_fly(fly, turn, fwd, cfg.fly, cfg.arena, dt)
            t += dt
            for fr in raw.fruits:
                if fr.id not in eaten and fr.time_left > t and (fr.x - fly.x) ** 2 + (fr.y - fly.y) ** 2 <= reach2:
                    eaten.add(fr.id)
                    cost -= p.fruit_reward
            while pending and pending[0].time_to_impact <= t:
                s = pending.pop(0)
                d = math.hypot(s.x - fly.x, s.y - fly.y)
                if d <= s.radius:
                    cost += p.hit_cost
                    stun = max(stun, cfg.swatter.stun_s)
                elif d < s.radius + p.safety_margin:
                    cost += p.near_cost * (s.radius + p.safety_margin - d)
        # Plan sonunda: hedefe (ya da en yakın meyveye) ne kadar uzak kaldık?
        best = math.inf
        for fr in raw.fruits:
            if fr.id not in eaten and fr.time_left > t:
                best = min(best, self._eta(fly.x, fly.y, fly.heading, fr.x, fr.y))
        if best < math.inf:
            cost += p.eta_cost * best
        # Duvar dibinde bitirmek kaçış seçeneklerini azaltır
        m = cfg.swatter.radius
        edge = min(fly.x, fly.y, cfg.arena.width - fly.x, cfg.arena.height - fly.y)
        if edge < m:
            cost += p.wall_cost * (1 - edge / m)
        if self._plan is not None and cand != self._plan:
            cost += p.switch_cost
        return cost

    # -- ana karar --------------------------------------------------------------------

    def get_action(self, state: GameState) -> Action:
        raw = state.raw
        f = raw.fly
        fly = FlyBody(f.x, f.y, f.heading, f.vx, f.vy)
        self._pick_target(raw)
        target = self._target_pos(raw)
        # Tepki gecikmesi: yeni beliren gölgeler henüz "fark edilmedi"
        hc = self.cfg.heuristic
        loom_s = self.cfg.swatter.loom_s
        threats = [s for s in raw.swatters if not s.impacted and s.progress * loom_s >= hc.reaction_s]

        if not threats:
            self._plan = None
            if target is None:
                # Meyve yok: arenanın ortasına doğru dolaş
                turn, fwd = self._steer(fly, raw.arena_w / 2, raw.arena_h / 2)
            else:
                turn, fwd = self._steer(fly, *target)
            return Action(turn, fwd * hc.speed_factor)

        if self._plan is None or self._plan_age >= self.p.replan_ticks:
            self._plan = min(self._candidates(raw), key=lambda c: self._rollout_cost(c, raw, threats, target))
            self._plan_age = 0
        self._plan_age += 1
        return Action(*self._control(self._plan, fly, target))
