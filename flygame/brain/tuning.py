"""Sinek beyni ayarlama: çok tohumda puan dağılımı ve evrimsel arama.

Birçok (parametre, tohum) işi tek bir toplu GPU beyninde aynı anda oynatılır:
her üyenin kendi dünyası, kendi beyin durumu ve kendi kodlayıcı/kod çözücü
parametreleri vardır. Kodlayıcı/kod çözücü canlı oyundakiyle aynı koddur
(brain/interface.py).

Evrimsel arama yalnızca kodlayıcı/kod çözücünün sayısal parametrelerini
değiştirir (kazançlar, eşikler, seyir hızı). Bağlantı ağına (konnektom) ve
nöron modeline dokunulmaz.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..actions import Action
from ..config import GameConfig
from ..sensing import build_state
from ..world import World
from .connectome import load_connectome, shuffled
from .interface import TUNABLE, BrainIO, DecoderState, active_tunables, params_from_config, stack_params
from .lif_torch import LIFBrain, LIFParams


@dataclass
class EvalResult:
    seed: int
    score: int
    fruits: int
    hits: int


class BatchEvaluator:
    def __init__(self, cfg: GameConfig, batch: int = 32, shuffle_seed: int | None = None,
                 device: str | None = None):
        self.cfg = cfg
        conn = load_connectome(cfg.brain.data_dir)
        if shuffle_seed is not None:
            conn = shuffled(conn, shuffle_seed)
        self.shuffled = shuffle_seed is not None
        self.io = BrainIO(conn, cfg.brain)
        self.B = batch
        self.brain = LIFBrain(conn, self.io.input_idx, self.io.readout_idx,
                              device=device or cfg.brain.device, batch=batch,
                              params=LIFParams(dt=cfg.brain.dt_ms))
        self.brain.run(1)  # çekirdek derlemesi
        self.brain_seconds = 0.0
        self.wall_seconds = 0.0

    def evaluate(self, jobs: list[tuple[dict, int]], progress=None) -> list[EvalResult]:
        """jobs: (parametre sözlüğü, oyun tohumu) listesi. Sıra korunur."""
        out: list[EvalResult] = []
        for start in range(0, len(jobs), self.B):
            chunk = jobs[start:start + self.B]
            padded = chunk + [chunk[-1]] * (self.B - len(chunk))  # sabit boyut: yeniden derleme yok
            t0 = time.perf_counter()
            res = self._run_batch(padded)
            self.wall_seconds += time.perf_counter() - t0
            out.extend(res[:len(chunk)])
            if progress:
                progress(len(out), len(jobs))
        return out

    def _run_batch(self, jobs: list[tuple[dict, int]]) -> list[EvalResult]:
        cfg, io, brain = self.cfg, self.io, self.brain
        P = stack_params([p for p, _ in jobs])
        worlds = [World(cfg, seed) for _, seed in jobs]
        # Beyin gürültüsü (Poisson girdisi) için tekrarlanabilir tohum
        brain.reset(1 + sum(seed * (i + 1) for i, (_, seed) in enumerate(jobs)) % (2**31))
        dec = DecoderState.new(self.B, P["cruise_forward"])
        block = brain.block_ms
        dt_ms = worlds[0].dt * 1000.0
        game_ms = 0.0
        while not worlds[0].done:
            rates, _ = io.encode_states([build_state(w) for w in worlds], P)
            brain.set_input_rates(rates)
            game_ms += dt_ms
            n = int((game_ms - brain.time_ms) / block + 1e-9)
            if n > 0:
                io.decode(brain.run(n), n * block, P, dec)
                self.brain_seconds += n * block / 1000.0 * self.B
            for i, w in enumerate(worlds):
                w.step(Action(float(dec.turn[i]), float(dec.forward[i])))
        return [EvalResult(seed, w.score, w.fruits_collected, w.hits) for (_, seed), w in zip(jobs, worlds)]


# -- evrimsel arama --------------------------------------------------------------------

def _to_unit(params: dict, names: list[str] | None = None) -> np.ndarray:
    u = []
    for k in names or list(TUNABLE):
        lo, hi, scale = TUNABLE[k]
        x = min(max(float(params[k]), lo), hi)
        u.append(math.log(x / lo) / math.log(hi / lo) if scale == "log" else (x - lo) / (hi - lo))
    return np.array(u)


def _from_unit(u: np.ndarray, names: list[str] | None = None, base: dict | None = None) -> dict:
    out = dict(base or {})
    for k, x in zip(names or list(TUNABLE), np.clip(u, 0.0, 1.0)):
        lo, hi, scale = TUNABLE[k]
        val = lo * (hi / lo) ** x if scale == "log" else lo + x * (hi - lo)
        out[k] = float(f"{val:.4g}")  # okunabilir, yuvarlanmış değerler
    return out


@dataclass
class Candidate:
    params: dict
    scores: list

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else -math.inf


def evolve(evaluator: BatchEvaluator, base: dict, generations: int, pop: int, seeds_per_gen: int,
           sigma: float = 0.15, sigma_decay: float = 0.9, search_seed: int = 0,
           seed_offset: int = 10_000, log=print, names: list[str] | None = None,
           on_generation=None) -> tuple[Candidate, list[dict]]:
    """Seçkinci (elitist) evrim stratejisi.

    Her nesilde tüm adaylar aynı, yeni tohumlarda oynar (ortak rastgele sayılar).
    Seçkinler bir sonraki nesle taşınır ve yeniden değerlendirilir; sıralama bir
    adayın o ana kadarki TÜM puanlarının ortalamasıyla yapılır (şanslı tek tura
    karşı dirençli).
    """
    rng = np.random.default_rng(search_seed)
    names = names or active_tunables(evaluator.cfg.brain)  # yalnızca seçili kodlamada anlamlı olanlar
    n_elite = max(2, pop // 4)
    population = [Candidate(dict(base), [])]
    while len(population) < pop:
        u = _to_unit(base, names) + rng.normal(0, sigma * 2, len(names))
        population.append(Candidate(_from_unit(u, names, base), []))
    history = []
    for gen in range(generations):
        seeds = [seed_offset + gen * seeds_per_gen + j for j in range(seeds_per_gen)]
        jobs = [(c.params, s) for c in population for s in seeds]
        t0 = time.perf_counter()
        res = evaluator.evaluate(jobs)
        for i, c in enumerate(population):
            c.scores.extend(r.score for r in res[i * seeds_per_gen:(i + 1) * seeds_per_gen])
        population.sort(key=lambda c: c.mean, reverse=True)
        gen_scores = [np.mean([r.score for r in res[i * seeds_per_gen:(i + 1) * seeds_per_gen]])
                      for i in range(len(population))]
        best = population[0]
        history.append({"generation": gen, "seeds": seeds, "sigma": sigma,
                        "best_mean": best.mean, "best_n": len(best.scores), "best_params": best.params,
                        "gen_mean": float(np.mean(gen_scores)), "gen_max": float(np.max(gen_scores)),
                        "wall_s": time.perf_counter() - t0})
        log(f"gen {gen:2d}: this-gen mean {np.mean(gen_scores):5.2f} max {np.max(gen_scores):5.2f} | "
            f"best so far {best.mean:5.2f} over {len(best.scores)} rounds | sigma {sigma:.3f} | "
            f"{time.perf_counter() - t0:5.0f}s")
        if on_generation:
            on_generation(gen, best, history)  # ör. ara kayıt (checkpoint)
        # Sonraki nesil: seçkinler + onlardan mutasyon/çaprazlama ile çocuklar
        elites = population[:n_elite]
        children = []
        while len(elites) + len(children) < pop:
            a = elites[min(int(rng.exponential(n_elite / 2)), n_elite - 1)]
            ua = _to_unit(a.params, names)
            if rng.random() < 0.5:
                ub = _to_unit(elites[rng.integers(n_elite)].params, names)
                ua = np.where(rng.random(len(ua)) < 0.5, ua, ub)
            children.append(Candidate(_from_unit(ua + rng.normal(0, sigma, len(ua)), names, a.params), []))
        population = elites + children
        sigma *= sigma_decay
    population.sort(key=lambda c: c.mean, reverse=True)
    return population[0], history


def summarize(scores: list[int]) -> str:
    a = np.array(scores, dtype=float)
    q = np.percentile(a, [10, 50, 90])
    return (f"mean {a.mean():5.2f}  sd {a.std(ddof=1) if len(a) > 1 else 0:4.2f}  "
            f"min {a.min():3.0f}  p10 {q[0]:4.1f}  median {q[1]:4.1f}  p90 {q[2]:4.1f}  max {a.max():3.0f}  (n={len(a)})")


def histogram(scores: list[int], width: int = 40) -> str:
    a = np.array(scores)
    lo, hi = int(a.min()), int(a.max())
    counts = np.bincount(a - lo, minlength=hi - lo + 1)
    scale = width / max(1, counts.max())
    return "\n".join(f"  {lo + i:3d} | {'#' * int(round(c * scale))} {c if c else ''}" for i, c in enumerate(counts))


def params_toml(params: dict, header_lines: list[str], encoding: str | None = None) -> str:
    lines = [f"# {h}" for h in header_lines] + ["", "[brain]"]
    if encoding:
        lines.append(f'fruit_encoding = "{encoding}"')
    for k in TUNABLE:
        lines.append(f"{k} = {float(params[k])!r}")
    return "\n".join(lines) + "\n"


def default_params(cfg: GameConfig) -> dict:
    return {k: float(v[0]) for k, v in params_from_config(cfg.brain).items()}
