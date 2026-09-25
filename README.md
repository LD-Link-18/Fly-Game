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
A fused Triton kernel runs one brain at about 2.2x real time inside the game on an
RTX 4070 Laptop GPU, or ~32 brains in parallel at ~14 brain-seconds per second for tuning.
Results are deterministic: same seed and inputs, same spikes.

```bash
.venv/bin/pip install -r requirements-brain.txt           # ~3 GB (PyTorch + CUDA)
.venv/bin/python -m flygame brain-download                # FlyWire data, ~137 MB, checksummed
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

### Fruit vision: hemifield vs retinotopic

`[brain] fruit_encoding` picks how fruit reaches the brain:

- `hemifield`: every left LC10a neuron gets the same rate (fruit on the left), same for
  the right. The brain only learns *left or right*.
- `retinotopic`: each LC10a neuron is driven by how close the fruit is to where that neuron
  looks, so the brain also learns *how far* left or right. Receptive-field directions are
  estimated from the connectome (`flygame/brain/retinotopy.py`): FlyWire's column
  assignment (Matsliah et al. 2024) places 31 columnar cell types on the eye's hexagonal
  grid; each LC10a's center is the synapse-weighted average column of its inputs, mapped
  linearly to -12°..155° per eye. That mapping is an approximation of the real eye map.

What the connectome does with that direction information (probe, 12 LC10a neurons per
azimuth bin): DNa02 steers toward objects mostly at 45–75° and ignores objects behind
(~140°); DNa03/DNa11/DNpe023 respond to objects straight ahead; DNae002/DNg111 to objects
at the side/rear. The decoder can therefore add the side neurons to turning
(`turn_lat_weight`) and let the front neurons raise speed (`speed_front_gain`); with those
weights at 0 it is the original DNa01/DNa02 decoder.

Result so far (each encoding tuned with `brain-tune`, then compared on the same 24 hold-out
seeds 200–223): hemifield `configs/brain_tuned.toml` 21.6 ± 5.4, retinotopic
`configs/brain_tuned_retinotopic.toml` 20.4 ± 3.5. The difference is within noise; the
retinotopic fly is more consistent (worst round 15 vs 12) but does not score higher. The
search widened the receptive fields to ~48°, i.e. it preferred blurrier direction input:
the LC10a→DNa02 pursuit pathway ignores objects behind the fly, and this game rewards
turning toward fruit anywhere.

### Tuning the fly (Phase 4)

```bash
# Score distribution of the fly brain over many seeds (runs up to 32 brains at once)
.venv/bin/python -m flygame brain-eval --seeds 1-64 --compare
# Same, but on a shuffled connectome: a control for "does the wiring matter?"
.venv/bin/python -m flygame brain-eval --seeds 1-64 --shuffle-seed 1
# Evolutionary search over the encoder/decoder numbers (gains, thresholds, cruise speed);
# writes a best-so-far checkpoint to runs/tuning/ after every generation
.venv/bin/python -m flygame brain-tune --generations 12 --pop 16 --seeds-per-gen 2
# Progress and time left of a running search (or: watch -n 30 ... to keep it on screen)
.venv/bin/python -m flygame tune-status
.venv/bin/python -m flygame play --opponent flybrain --config configs/brain_tuned.toml
```

The search only changes the numbers in `[brain]` that sit *between* the game and the
brain (listed in `flygame/brain/interface.py`, `TUNABLE`); the connectome and the neuron
model are never modified. It trains on seeds 10000+, then reports default vs tuned on
hold-out seeds it never saw, the tuned settings on a shuffled connectome, and the bots.
Logs go to `runs/tuning/`.

## Running the stand

```bash
./kiosk.sh                       # fullscreen kiosk: visitors choose FLY BRAIN or HEURISTIC BOT
LANG_UI=tr ./kiosk.sh            # Turkish UI (F3 switches language at any time)
OPPONENTS=flybrain ./kiosk.sh    # only the fly brain, no choice screen
```

`kiosk.sh` runs `play --kiosk` with `configs/brain_tuned.toml` + `configs/stand.toml` and
restarts the game if it crashes. Kiosk mode: fullscreen, mouse hidden, returns to the
title screen when idle; **hold ESC for 3 s to quit**.

| Key | Operator action |
|---|---|
| F1 | help overlay with the current settings |
| F2 | change the default opponent |
| F3 | English / Turkish |
| F4 | sound on / off |
| F9 (twice) | reset the leaderboard (the old file is kept as a backup) |

Visitor flow: title screens (rotating: title, "how does the fly play?", leaderboard) →
choose opponent → 3-2-1 → 40 s round → result and prize → 3 initials if the score makes
the top 10 → leaderboard. Rounds where the visitor never touched the controls are not
recorded.

**Brain map** (right of the fly): every one of the 138,639 simulated neurons is a dot at
its real position in the FlyWire brain (`pos_x/y/z` from the FlyWire annotations,
`flygame/brain/brainmap.py`), drawn as a dim LED-style map, seen from behind (the fly's
left is on screen-left) and from above (head up, like the fly in the game). When a neuron
fires in the simulation, its dot lights up and fades within ~60 ms, so what you see is the
whole brain's spikes, not a sample or an animation. Colours: green = the eye neurons fed
with fruit (LC10a), red = the eye neurons fed with the swatter (LPLC2, LC4), orange = the
decision neurons the game reads (they also get a ring when they fire), blue = every other
neuron. Under the map, "TO THE LEGS" shows the turn command and giant-fiber escapes.
The panel says LIVE for the running simulation and RECORDED for replays; bots get a
"no brain" card. The map needs `data/flywire` (from `brain-download`); a small position
cache (`brain_map_783.npz`) is built from it on first use, without pandas or torch.

Recordings of the fly brain store every tick's spiking neurons (packed as one bit matrix
over the neurons that fired during the round, ~0.5 MB per round), so replays show the
recorded activity. Recordings made before the brain map have no map data; the panel says so.

**Prize rule** (`[prize]` in `configs/stand.toml`): a visitor wins a prize by beating an
eligible opponent (default: the fly brain, including its recorded replays) by at least
`margin` points with at least `min_score` points; `max_per_day` limits the daily stock.

**Leaderboard**: `runs/stand/leaderboard.json`, written atomically after every round. It
shows the top visitor scores, the average score of each opponent at the stand, and
visitors' win/loss counts per opponent.

No GPU at the stand? Pre-record ghost runs and replay them (labelled RECORDED, with the
recorded neuron activity):

```bash
.venv/bin/python -m flygame record --agent flybrain --config configs/brain_tuned.toml --seeds 1-50 --out runs/fly_ghosts
OPPONENTS=replay:runs/fly_ghosts,heuristic ./kiosk.sh
```

To stop the screen from blanking during the event (GNOME):
`gsettings set org.gnome.desktop.session idle-delay 0` and
`gsettings set org.gnome.desktop.screensaver lock-enabled false`.

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
