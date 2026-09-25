"""Komut satırı girişi.

  python -m flygame play   [--opponent scripted] [--seed N] [--config f.toml] [--fullscreen] [--lang tr]
  python -m flygame record --agent scripted --seeds 1-20 [--out runs/scripted]
  python -m flygame verify runs/x.json.gz
  python -m flygame brain-download         # FlyWire verisini indir (~136 MB)
  python -m flygame brain-probe            # sinek beyni: sol/sağ uyarım -> DN yanıtları
  python -m flygame brain-eval --seeds 1-32 # sinek beyni puan dağılımı
  python -m flygame brain-tune              # kodlayıcı/kod çözücü evrimsel arama
  python -m flygame tune-status             # çalışan aramanın ilerlemesi ve kalan süresi
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

from .agents import make_agent
from .config import config_from_dict, load_config
from .recording import Recording
from .runner import run_round


def parse_seeds(spec: str) -> list[int]:
    # "5" -> [5], "1-10" -> [1..10], "1,4,9" -> [1,4,9]
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_play(args) -> None:
    cfg = load_config(args.config)
    if args.fullscreen:
        cfg.display.fullscreen = True
    if args.lang:
        cfg.display.language = args.lang
    if args.no_sound:
        cfg.display.sound = False
    from .app import App

    App(cfg, make_agent(args.opponent, cfg), args.seed).run()


def cmd_record(args) -> None:
    cfg = load_config(args.config)
    out = Path(args.out or f"runs/{args.agent.replace(':', '_').replace('/', '_')}")
    scores = []
    agent = make_agent(args.agent, cfg)  # bir kez oluştur (sinek beyni yüklemesi pahalı)
    for seed in parse_seeds(args.seeds):
        res = run_round(agent, seed, cfg, record=True)
        path = res.recording.save(out / f"seed{seed}_{agent.name}.json.gz")
        scores.append(res.score)
        print(f"seed {seed:>6}  score {res.score:>3}  fruit {res.fruits:>3}  hits {res.hits:>2}  "
              f"{res.realtime_factor:8.1f}x realtime  -> {path}")
    if len(scores) > 1:
        print(f"mean {statistics.mean(scores):.2f}  sd {statistics.stdev(scores):.2f}  "
              f"min {min(scores)}  max {max(scores)}")


def cmd_verify(args) -> None:
    # Kaydı, kaydedildiği ayarlarla yeniden oynat ve skorun tuttuğunu kontrol et
    from .agents.replay import ReplayAgent

    ok = True
    for p in args.paths:
        rec = Recording.load(p)
        cfg = config_from_dict(rec.config)
        agent = ReplayAgent(p)
        res = run_round(agent, agent.pick_seed(), cfg)
        match = res.score == rec.final_score
        ok &= match
        print(f"{'OK  ' if match else 'FAIL'} {p}: recorded {rec.final_score}, replayed {res.score}")
        cur = load_config(args.config)
        if cur.gameplay_hash() != rec.config_hash:
            print("     note: current config differs from the one this run was recorded with")
    sys.exit(0 if ok else 1)


def cmd_brain_probe(args) -> None:
    from .brain.probe import run_probe

    cfg = load_config(args.config)
    run_probe(cfg.brain.data_dir, args.device or cfg.brain.device, dt_ms=cfg.brain.dt_ms)


def cmd_brain_download(args) -> None:
    from .brain.download import download_all

    cfg = load_config(args.config)
    download_all(cfg.brain.data_dir)
    from .brain.connectome import load_connectome

    c = load_connectome(cfg.brain.data_dir)
    print(f"connectome ready: {c.n_neurons} neurons, {c.n_connections} connections")


def _progress(done: int, total: int) -> None:
    print(f"  ... {done}/{total} rounds", flush=True)


def _baselines(seeds: list[int], cfg) -> dict[str, list[int]]:
    from .agents.heuristic import HeuristicAgent
    from .agents.scripted import ScriptedAgent

    return {name: [run_round(a, s, cfg).score for s in seeds]
            for name, a in (("heuristic bot", HeuristicAgent()), ("scripted bot", ScriptedAgent()))}


def cmd_brain_eval(args) -> None:
    from .brain.tuning import BatchEvaluator, default_params, histogram, summarize

    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    ev = BatchEvaluator(cfg, batch=min(args.batch, len(seeds)), shuffle_seed=args.shuffle_seed)
    what = f"SHUFFLED connectome (control, shuffle seed {args.shuffle_seed})" if ev.shuffled else "FlyWire v783 connectome"
    print(f"Fly brain on {len(seeds)} seeds, {what}, batch {ev.B}")
    res = ev.evaluate([(default_params(cfg), s) for s in seeds], progress=_progress)
    scores = [r.score for r in res]
    print(f"\nscore  {summarize(scores)}")
    print(f"fruit mean {np.mean([r.fruits for r in res]):.2f}   hits/round {np.mean([r.hits for r in res]):.2f}")
    print(histogram(scores))
    print(f"\nthroughput: {ev.brain_seconds / ev.wall_seconds:.1f} brain-seconds per second")
    if args.compare:
        for name, sc in _baselines(seeds, cfg).items():
            print(f"{name:14} {summarize(sc)}")


def cmd_brain_tune(args) -> None:
    import json
    import time as _time

    import torch

    from .brain.tuning import BatchEvaluator, default_params, evolve, histogram, params_toml, summarize

    cfg = load_config(args.config)
    holdout = parse_seeds(args.holdout)
    batch = min(args.batch, args.pop * args.seeds_per_gen)
    ev = BatchEvaluator(cfg, batch=batch)
    base = default_params(cfg)
    stamp = _time.strftime('%Y%m%d-%H%M%S')
    checkpoint = Path("runs/tuning") / f"tune_{stamp}_checkpoint.toml"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evolutionary search: {args.generations} generations x {args.pop} candidates x "
          f"{args.seeds_per_gen} seeds (batch {batch}); training seeds start at {args.seed_offset}, "
          f"hold-out seeds {args.holdout}; best-so-far checkpoint: {checkpoint}")

    def save_checkpoint(gen, best, history):
        # Arama durdurulursa bile o ana kadarki en iyi ayarlar kaybolmasın
        checkpoint.write_text(params_toml(best.params, [
            f"ARA KAYIT (checkpoint): nesil {gen + 1}/{args.generations}; eğitim ortalaması "
            f"{best.mean:.2f} ({len(best.scores)} tur). Ayrılmış tohumlarda henüz test edilmedi.",
            f"Kullanım: python -m flygame play --opponent flybrain --config {checkpoint}",
        ], cfg.brain.fruit_encoding))

    best, history = evolve(ev, base, args.generations, args.pop, args.seeds_per_gen,
                           sigma=args.sigma, search_seed=args.search_seed, seed_offset=args.seed_offset,
                           on_generation=save_checkpoint)

    print("\nHold-out evaluation (seeds never used during the search):")
    res = ev.evaluate([(base, s) for s in holdout] + [(best.params, s) for s in holdout], progress=_progress)
    d_scores = [r.score for r in res[:len(holdout)]]
    t_scores = [r.score for r in res[len(holdout):]]
    del ev
    torch.cuda.empty_cache()
    print("Control: tuned parameters on a SHUFFLED connectome (same degrees and weights, wiring destroyed)")
    ev_s = BatchEvaluator(cfg, batch=batch, shuffle_seed=1)
    s_scores = [r.score for r in ev_s.evaluate([(best.params, s) for s in holdout], progress=_progress)]
    base_scores = _baselines(holdout, cfg)

    rows = [("fly brain, default params", d_scores), ("fly brain, TUNED params", t_scores),
            ("tuned params, shuffled wiring", s_scores)] + list(base_scores.items())
    print()
    for name, sc in rows:
        print(f"{name:30} {summarize(sc)}")
    print("\nTuned fly brain score histogram (hold-out):")
    print(histogram(t_scores))

    header = [
        "Sinek beyni için evrimsel arama ile ayarlanmış kodlayıcı/kod çözücü parametreleri.",
        f"Oluşturma: {_time.strftime('%Y-%m-%d %H:%M')}, python -m flygame brain-tune",
        f"Arama: {args.generations} nesil x {args.pop} aday x {args.seeds_per_gen} tohum, "
        f"meyve kodlaması: {cfg.brain.fruit_encoding}",
        f"Ayrılmış tohumlar ({args.holdout}): varsayılan ort. {np.mean(d_scores):.2f}, "
        f"ayarlı ort. {np.mean(t_scores):.2f}, karıştırılmış bağlantı ağı ort. {np.mean(s_scores):.2f}",
        f"Kullanım: python -m flygame play --opponent flybrain --config {args.out}",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(params_toml(best.params, header, cfg.brain.fruit_encoding))
    log_path = Path("runs/tuning") / f"tune_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"args": vars(args) | {"func": None}, "history": history,
                                    "best": {"params": best.params, "train_scores": best.scores},
                                    "holdout": {name: sc for name, sc in rows}}, indent=1, default=str))
    print(f"\nWrote {args.out} and {log_path}")


def cmd_tune_status(args) -> None:
    from .tune_status import latest_log, status

    log = Path(args.log) if args.log else latest_log()
    if log is None or not log.exists():
        print("No tuning log found in runs/tuning/")
        sys.exit(1)
    print(status(log))


def main() -> None:
    ap = argparse.ArgumentParser(prog="flygame")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="run the split-screen game")
    p.add_argument("--opponent", default="scripted", help="scripted | heuristic | flybrain | replay:<file-or-dir>")
    p.add_argument("--seed", type=int, default=None, help="fixed seed for every round")
    p.add_argument("--config", default=None, help="TOML file with setting overrides")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--lang", choices=["en", "tr"], default=None)
    p.add_argument("--no-sound", action="store_true")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("record", help="run an agent headless on seeds and save recordings")
    p.add_argument("--agent", required=True)
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("verify", help="check that recordings replay to the same score")
    p.add_argument("paths", nargs="+")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("brain-download", help="download FlyWire v783 data (~136 MB) and build the cache")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_brain_download)

    p = sub.add_parser("brain-probe", help="check left/right sensory -> descending neuron responses")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None, help="override brain.device (cuda/cpu)")
    p.set_defaults(func=cmd_brain_probe)

    p = sub.add_parser("brain-eval", help="run the fly brain on many seeds and report the score distribution")
    p.add_argument("--seeds", default="1-32")
    p.add_argument("--config", default=None)
    p.add_argument("--batch", type=int, default=32, help="brains simulated in parallel on the GPU")
    p.add_argument("--shuffle-seed", type=int, default=None, help="control: use a shuffled connectome")
    p.add_argument("--compare", action="store_true", help="also run the heuristic and scripted bots")
    p.set_defaults(func=cmd_brain_eval)

    p = sub.add_parser("brain-tune", help="evolutionary search over encoder/decoder parameters")
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--pop", type=int, default=16, help="candidates per generation")
    p.add_argument("--seeds-per-gen", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.15, help="mutation size (fraction of each range)")
    p.add_argument("--holdout", default="1-24", help="seeds for the final comparison (not used in search)")
    p.add_argument("--seed-offset", type=int, default=10_000, help="first training seed")
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="configs/brain_tuned.toml")
    p.set_defaults(func=cmd_brain_tune)

    p = sub.add_parser("tune-status", help="show progress and time left of a running brain-tune")
    p.add_argument("--log", default=None, help="log file (default: newest in runs/tuning/)")
    p.set_defaults(func=cmd_tune_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
