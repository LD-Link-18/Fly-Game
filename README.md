# Beat the Fly

Split-screen stand game: a visitor plays a fly (arrow keys) against a computer
opponent on the same seeded fruit/swatter pattern. The goal is for the opponent
to eventually be a connectome-based fruit-fly brain simulation.

Every opponent is labeled on screen with what it really is (scripted bot,
heuristic bot, live fly-brain simulation, or recorded replay).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m flygame play                          # vs scripted bot
.venv/bin/python -m flygame play --opponent heuristic     # vs heuristic bot (seeks fruit, dodges)
.venv/bin/python -m flygame play --lang tr --fullscreen
.venv/bin/python -m flygame play --config configs/hard.toml
.venv/bin/python -m flygame play --opponent replay:runs/  # ghost replays
```

Keys: arrows (or WASD) to play, SPACE to start/continue, F11 to toggle fullscreen, ESC to quit.

## Opponents

| `--opponent` | What it is |
|---|---|
| `scripted` | heads for the nearest fruit, never dodges |
| `heuristic` | picks fruit by arrival time, dodges swatters with a short look-ahead planner; weaken it with `[heuristic] speed_factor` / `reaction_s` in the config |
| `flybrain` | **live whole-brain simulation** of the FlyWire v783 connectome (see below) |
| `replay:<file-or-dir>` | replays recorded runs (labelled as replays on screen) |

## Fly brain

A PyTorch/GPU port of the leaky integrate-and-fire whole-brain model of
[Shiu et al. 2024](https://github.com/philshiu/Drosophila_brain_model):
138,639 neurons and 15.1M connections from FlyWire v783, same equations,
parameters and 0.1 ms time step. The port matches the Brian2 reference
spike-for-spike on deterministic input and statistically on Poisson input.
On an RTX 4070 Laptop GPU it runs at about 1.4x real time inside the game.

```bash
.venv/bin/pip install -r requirements-brain.txt           # ~3 GB (PyTorch + CUDA)
.venv/bin/python -m flygame brain-download                # FlyWire data, ~136 MB, checksummed
.venv/bin/python -m flygame brain-probe                   # left/right stimulus -> DN responses
.venv/bin/python -m flygame play --opponent flybrain
```

How the game talks to the brain (all choices, gains and thresholds are in `[brain]` in the config):

- **In:** fruit seen on the left/right drives left/right **LC10a** (small-object visual
  neurons); a looming swatter on the left/right drives **LPLC2 + LC4** (looming detectors).
  The game computes these visual signals itself; the eye and optic lobe are *not* simulated.
- **Out:** right-minus-left firing of **DNa01/DNa02** (descending neurons that start
  same-side turns) sets turning; the giant fiber **DNp01** triggers an escape dash.
- **Hand-set, and labelled as such on screen:** the cruising speed (no forward-walking
  DN signal came out of the probe) and the input/output mappings themselves.
- What comes from the connectome: object on the left makes left DNa02 fire (turn toward),
  looming on the left makes right DNa01/DNa02 and the giant fiber fire (turn away, escape).

## Headless tools

```bash
# Run an agent on many seeds, save recordings, print score stats
.venv/bin/python -m flygame record --agent scripted --seeds 1-50
# Check that recordings still replay to the same score
.venv/bin/python -m flygame verify runs/scripted/*.json.gz
# Tests
.venv/bin/python -m unittest discover tests
```

## Layout

| File | Purpose |
|---|---|
| `flygame/config.py` | all difficulty knobs (override with a TOML file, see `configs/`) |
| `flygame/schedule.py` | seed → fruit/swatter pattern, generated before the round |
| `flygame/world.py` | deterministic simulation (no pygame) |
| `flygame/sensing.py` | `GameState` = `raw` + fly-style `sensory` signals |
| `flygame/agents/` | `Agent.get_action(state) -> Action` implementations |
| `flygame/recording.py` | recordings (seed + config + per-tick actions + optional telemetry) |
| `flygame/runner.py` | headless round runner |
| `flygame/app.py`, `render.py`, `audio.py` | pygame front-end |
