"""Komut satırı girişi.

  python -m flygame play   [--opponent scripted] [--seed N] [--config f.toml] [--fullscreen] [--lang tr]
  python -m flygame record --agent scripted --seeds 1-20 [--out runs/scripted]
  python -m flygame verify runs/x.json.gz
  python -m flygame brain-download         # FlyWire verisini indir (~136 MB)
  python -m flygame brain-probe            # sinek beyni: sol/sağ uyarım -> DN yanıtları
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
