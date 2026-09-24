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
| `replay:<file-or-dir>` | replays recorded runs (labelled as replays on screen) |

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
