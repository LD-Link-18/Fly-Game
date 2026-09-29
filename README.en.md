# Beat the Fly (Sineği Yen)

[Türkçe](README.md) · **English**

A split-screen game built for science stands. A visitor flies a fly with the arrow keys;
on the other half of the screen a computer opponent plays the **same** fruit and swatter
pattern. After 40 seconds, whoever collected more points wins.

The headline opponent is a **live simulation of a fruit fly's (*Drosophila*) whole
brain**: the 138,639 neurons and 15.1 million connections of the FlyWire v783 connectome
run on the GPU faster than real time while the game is played. In the brain map next to
the opponent, every simulated neuron lights up at its real position the moment it fires.

Honesty rule: every opponent is labeled on screen with what it really is (scripted bot,
heuristic bot, live fly-brain simulation, recorded replay). A bot or a recording is never
presented as a "live fly brain".

## Contents

- [The game](#the-game)
- [Requirements](#requirements)
- [Install and play](#install-and-play)
- [Opponents](#opponents)
- [Fly brain](#fly-brain)
- [Stand (kiosk) mode](#stand-kiosk-mode)
- [Configuration](#configuration)
- [Recordings, headless tools and tests](#recordings-headless-tools-and-tests)
- [Layout](#layout)
- [Sources](#sources)
- [License](#license)

## The game

- A round lasts 40 seconds. The visitor plays on the left, the opponent on the right;
  both sides get the same fruit and swatter timing, generated from the same seed.
- Each fruit is worth **+1**. A swatter's shadow grows and strikes after ~1.3 s; a fly
  caught under it loses **3 points** and is stunned for 1 s. The score never goes below 0.
- Swatters mostly aim at the fly and lead its movement; to escape, fly sideways while the
  shadow grows.
- The simulation is deterministic: same config + same seed + same actions = same result.
  This is what makes replays and opponent tuning possible.

| Key | Action |
|---|---|
| ← / → (or A / D) | turn left / right |
| ↑ (or W) | fly forward (the fly stops when released) |
| SPACE | start / continue |
| F11 | toggle fullscreen (not in kiosk mode) |
| ESC | quit (hold for 3 s in kiosk mode) |

## Requirements

- **Python 3.11+** (for `tomllib`). Developed and tested with Python 3.14.
- **Game:** `pygame-ce` and `numpy`. The font module of the original `pygame` 2.6.1 does
  not work on Python 3.14, so the maintained `pygame-ce` is used (same `import pygame` API).
- **Fly-brain opponent (optional):** an NVIDIA GPU and PyTorch with CUDA (~3 GB). If
  Triton is available (it ships with PyTorch's CUDA wheels on Linux) a fused GPU kernel is
  used; otherwise it falls back to a slower pure-PyTorch path. It also runs on the CPU,
  but far too slowly for the game. `pandas` and `fastparquet` are only needed once, to
  build a cache from the raw FlyWire data. Development was done on an RTX 4070 Laptop GPU.
- On a machine without a GPU, the fly brain can be pre-recorded and replayed
  (see [Stand without a GPU](#stand-without-a-gpu)).

## Install and play

```bash
git clone https://github.com/LD-Link-18/Fly-Game.git
cd Fly-Game
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m flygame play                          # vs scripted bot
.venv/bin/python -m flygame play --opponent heuristic     # vs heuristic bot
.venv/bin/python -m flygame play --lang tr --fullscreen   # Turkish UI, fullscreen
.venv/bin/python -m flygame play --config configs/hard.toml
.venv/bin/python -m flygame play --opponent replay:runs/  # vs recorded runs ("ghosts")
.venv/bin/python -m flygame play --opponent flybrain,heuristic  # visitor picks the opponent
```

`play` options: `--opponent` (a comma list lets the visitor choose each round),
`--config` (repeatable), `--seed N` (same seed every round), `--lang en|tr`,
`--fullscreen`, `--no-sound`, `--kiosk`.

## Opponents

| `--opponent` | What it is |
|---|---|
| `scripted` | heads for the nearest fruit, never dodges |
| `heuristic` | picks fruit by arrival time, dodges swatters with a short look-ahead planner; weaken it with `[heuristic] speed_factor` / `reaction_s` |
| `flybrain` | **live whole-brain simulation** of the FlyWire v783 connectome (see below) |
| `replay:<file-or-dir>` | replays recorded runs (labeled RECORDED REPLAY on screen); with a directory, a random recording is picked each round |

## Fly brain

A PyTorch/GPU port of the leaky integrate-and-fire (LIF) whole-brain model of
[Shiu et al. 2024](https://github.com/philshiu/Drosophila_brain_model): 138,639 neurons
and 15.1M connections from FlyWire v783, same equations, parameters and 0.1 ms time step.
The port matches the Brian2 reference spike-for-spike on deterministic input and
statistically on Poisson input. A fused Triton kernel runs one brain at about 2.2x real
time inside the game on an RTX 4070 Laptop GPU, or ~32 brains in parallel at ~14
brain-seconds per second for tuning. Results are deterministic: same seed and inputs,
same spikes.

```bash
.venv/bin/pip install -r requirements-brain.txt           # ~3 GB (PyTorch + CUDA)
.venv/bin/python -m flygame brain-download                # FlyWire data, ~137 MB, SHA-256 checked
.venv/bin/python -m flygame brain-probe                   # left/right stimulus -> descending neuron responses
.venv/bin/python -m flygame play --opponent flybrain --config configs/brain_tuned.toml
```

`brain-download` fetches the data from pinned commits, verified by SHA-256, into
`data/flywire/` and builds an `.npz` cache on first use; during the game only that cache
is read, with numpy.

### How the game talks to the brain

All choices, gains and thresholds are in the `[brain]` section of the config.

- **In:** fruit seen on the left/right drives left/right **LC10a** (small-object visual
  neurons); a looming swatter on the left/right drives **LPLC2 + LC4** (looming
  detectors). The game computes these visual signals itself; the eye and optic lobe are
  *not* simulated.
- **Out:** right-minus-left firing of **DNa01/DNa02** (descending neurons that start
  same-side turns) sets turning; the giant fiber **DNp01** triggers an escape dash.
- **Hand-set, and labeled as such on screen:** the cruising speed (no forward-walking
  descending-neuron signal came out of the probe) and the input/output mappings themselves.
- **What comes from the connectome:** an object on the left makes left DNa02 fire (turn
  toward); looming on the left makes right DNa01/DNa02 and the giant fiber fire (turn
  away, escape).

### Fruit vision: hemifield vs retinotopic

`[brain] fruit_encoding` picks how fruit reaches the brain:

- `hemifield`: every left LC10a neuron gets the same rate (fruit on the left), same for
  the right. The brain only learns *left or right*.
- `retinotopic`: each LC10a neuron is driven by how close the fruit is to where that
  neuron looks, so the brain also learns *how far* left or right. Receptive-field
  directions are estimated from the connectome
  ([flygame/brain/retinotopy.py](flygame/brain/retinotopy.py)): FlyWire's column
  assignment (Matsliah et al. 2024) places 31 columnar cell types on the eye's hexagonal
  grid; each LC10a's center is the synapse-weighted average column of its inputs, mapped
  linearly to -12°..155° per eye. That mapping is an approximation of the real eye map.

What the connectome does with that direction information (probe, 12 LC10a neurons per
azimuth bin): DNa02 steers toward objects mostly at 45–75° and ignores objects behind
(~140°); DNa03/DNa11/DNpe023 respond to objects straight ahead; DNae002/DNg111 to objects
at the side/rear. The decoder can therefore add the side neurons to turning
(`turn_lat_weight`) and let the front neurons raise speed (`speed_front_gain`); with those
weights at 0 it is the original DNa01/DNa02 decoder.

Result so far (each encoding tuned with `brain-tune`, then compared on the same 24
hold-out seeds, 200–223): hemifield `configs/brain_tuned.toml` 21.6 ± 5.4, retinotopic
`configs/brain_tuned_retinotopic.toml` 20.4 ± 3.5. The difference is within noise; the
retinotopic fly is more consistent (worst round 15 vs 12) but does not score higher. The
search widened the receptive fields to ~48°, i.e. it preferred blurrier direction input:
the LC10a→DNa02 pursuit pathway ignores objects behind the fly, and this game rewards
turning toward fruit anywhere.

### Tuning the fly

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
```

The search only changes the numbers in `[brain]` that sit *between* the game and the brain
(listed in [flygame/brain/interface.py](flygame/brain/interface.py), `TUNABLE`); the
connectome and the neuron model are never modified. It trains on seeds 10000+, then
reports default vs tuned parameters on hold-out seeds it never saw, the tuned parameters
on a shuffled connectome (control), and the bots. The result is written to `--out`
(default `configs/brain_tuned.toml`), logs to `runs/tuning/`.

## Stand (kiosk) mode

```bash
./kiosk.sh                       # fullscreen kiosk: visitors choose FLY BRAIN or HEURISTIC BOT
LANG_UI=tr ./kiosk.sh            # Turkish UI (F3 switches language at any time)
OPPONENTS=flybrain ./kiosk.sh    # only the fly brain, no choice screen
```

`kiosk.sh` runs `play --kiosk` with `configs/brain_tuned.toml` + `configs/stand.toml` and
restarts the game 3 s after a crash. The config files can be changed with the `CONFIGS`
environment variable. Kiosk mode: fullscreen, mouse hidden, returns to the title screen
when idle; **hold ESC for 3 s to quit**.

| Key | Operator action |
|---|---|
| F1 | help overlay with the current settings |
| F2 | change the default opponent |
| F3 | English / Turkish |
| F4 | sound on / off |
| F9 (twice) | reset the leaderboard (the old file is kept as a backup) |

**Visitor flow:** title screens (rotating: title, "how does the fly play?", leaderboard) →
choose opponent → 3-2-1 → 40 s round → result and prize → 3 initials if the score makes
the top 10 → leaderboard. Rounds where the visitor never touched the controls are not
recorded.

**Brain map** (right of the fly): every one of the 138,639 simulated neurons is a dot at
its real position in the FlyWire brain (`pos_x/y/z` from the FlyWire annotations,
[flygame/brain/brainmap.py](flygame/brain/brainmap.py)), drawn as a dim LED-style map from
two sides: from behind (the fly's left is on screen-left) and from above (head up, like
the fly in the game). When a neuron fires in the simulation, its dot lights up and fades
within ~60 ms, so what you see is the whole brain's spikes, not a sample or an animation.
Colors: green = the eye neurons fed with fruit (LC10a), red = the eye neurons fed with the
swatter (LPLC2, LC4), orange = the decision neurons the game reads (they also get a ring
when they fire), blue = every other neuron. Under the map, "TO THE LEGS" shows the turn
command and giant-fiber escapes. The panel says LIVE for the running simulation and
RECORDED for replays; bots get a "NO BRAIN" card. The map needs `data/flywire` (from
`brain-download`); a small position cache (`brain_map_783.npz`) is built from it on first
use, without pandas or torch.

Fly-brain recordings store every tick's spiking neurons (packed as one bit matrix over the
neurons that fired during the round, ~0.5 MB per round), so replays show the recorded
activity. Recordings made before the brain map have no map data; the panel says so.

**Prize rule** (`[prize]` in `configs/stand.toml`): a visitor wins a prize by beating an
eligible opponent (default: the fly brain, including its recorded replays) by at least
`margin` points with at least `min_score` points; `max_per_day` limits the daily stock
(0 = unlimited).

**Leaderboard:** `runs/stand/leaderboard.json`, written atomically after every round
(temporary file, then rename); a corrupt file is backed up and play continues with an
empty board. It shows the top visitor scores, the average score of each opponent at the
stand, and visitors' win/loss counts per opponent. With `leaderboard_scope = "today"` the
board starts fresh every day (old entries are kept in the file).

### Stand without a GPU

Pre-record fly-brain runs and replay them (labeled RECORDED, with the recorded neuron
activity):

```bash
.venv/bin/python -m flygame record --agent flybrain --config configs/brain_tuned.toml --seeds 1-50 --out runs/fly_ghosts
OPPONENTS=replay:runs/fly_ghosts,heuristic ./kiosk.sh
```

To stop the screen from blanking during the event (GNOME):

```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

## Configuration

All settings and their defaults are dataclasses in [flygame/config.py](flygame/config.py).
To change them without touching the code, pass a TOML file that lists only the fields you
want to change:

```toml
[swatter]
loom_s = 0.9          # time from shadow to strike (smaller = faster)
interval_min_s = 1.5  # shortest gap between two swatters
```

`--config` can be given more than once; later files win. An unknown key or a value of the
wrong type is an error (typos are not silently ignored).

Sections: `[arena]`, `[round]`, `[fly]`, `[fruit]`, `[swatter]`, `[score]`, `[sensory]`,
`[heuristic]`, `[brain]`, `[display]`, `[prize]`, `[stand]`, `[recording]`.

| File | Contents |
|---|---|
| [configs/easy.toml](configs/easy.toml) | easy mode (for young children): slower, rarer swatters, faster fly |
| [configs/hard.toml](configs/hard.toml) | hard mode: faster, more frequent swatters, rarer fruit |
| [configs/brain_tuned.toml](configs/brain_tuned.toml) | fly brain tuned with `brain-tune` (hemifield encoding) |
| [configs/brain_tuned_retinotopic.toml](configs/brain_tuned_retinotopic.toml) | fly brain tuned with `brain-tune` (retinotopic encoding) |
| [configs/stand.toml](configs/stand.toml) | stand: prize rule, leaderboard, kiosk behavior |

## Recordings, headless tools and tests

In the game, both sides of every round are saved under `runs/` (`[recording]`). A
recording is a gzipped JSON file with the seed, the config and the per-tick actions (plus
optional agent telemetry).

```bash
# Run an agent headless on many seeds, save recordings, print score stats
.venv/bin/python -m flygame record --agent scripted --seeds 1-50
# Check that recordings still replay to the same score
.venv/bin/python -m flygame verify runs/scripted/*.json.gz
# Tests (fly-brain tests are skipped without torch/CUDA/data)
.venv/bin/python -m unittest discover tests
```

`--seeds` formats: `5`, `1-10`, `1,4,9`.

## Layout

| File | Purpose |
|---|---|
| [flygame/\_\_main\_\_.py](flygame/__main__.py) | command line: `play`, `record`, `verify`, `brain-*`, `tune-status` |
| [flygame/config.py](flygame/config.py) | all settings and difficulty knobs (overridden by TOML) |
| [flygame/schedule.py](flygame/schedule.py) | seed → fruit/swatter pattern, generated before the round |
| [flygame/world.py](flygame/world.py) | deterministic game simulation (no pygame) |
| [flygame/sensing.py](flygame/sensing.py) | `GameState` = `raw` + fly-style `sensory` signals |
| [flygame/actions.py](flygame/actions.py) | `Action(turn, forward)` |
| [flygame/agents/](flygame/agents/) | `Agent.get_action(state) -> Action` implementations: human, scripted, heuristic, fly brain, replay |
| [flygame/runner.py](flygame/runner.py) | headless round runner |
| [flygame/recording.py](flygame/recording.py) | recordings and replay (seed + config + per-tick actions + telemetry) |
| [flygame/app.py](flygame/app.py) | pygame app and stand state machine |
| [flygame/render.py](flygame/render.py), [audio.py](flygame/audio.py), [texts.py](flygame/texts.py) | drawing, sounds synthesized with numpy, English/Turkish UI text |
| [flygame/neuron_panel.py](flygame/neuron_panel.py) | brain map panel |
| [flygame/leaderboard.py](flygame/leaderboard.py) | leaderboard and prize rule |
| [flygame/tune_status.py](flygame/tune_status.py) | progress of a running `brain-tune` search |
| [flygame/brain/download.py](flygame/brain/download.py) | pinned download of the FlyWire data |
| [flygame/brain/connectome.py](flygame/brain/connectome.py) | connectome loading and `.npz` cache |
| [flygame/brain/lif_torch.py](flygame/brain/lif_torch.py), [lif_triton.py](flygame/brain/lif_triton.py) | LIF whole-brain model (PyTorch) and fused Triton kernel |
| [flygame/brain/interface.py](flygame/brain/interface.py) | encoder (sensory → Poisson rates) and decoder (descending neurons → action) |
| [flygame/brain/retinotopy.py](flygame/brain/retinotopy.py) | receptive-field directions of LC neurons |
| [flygame/brain/probe.py](flygame/brain/probe.py) | left/right stimulus → descending neuron response probe |
| [flygame/brain/tuning.py](flygame/brain/tuning.py) | batched GPU evaluation and evolutionary search |
| [flygame/brain/brainmap.py](flygame/brain/brainmap.py) | neuron positions in the brain |
| [kiosk.sh](kiosk.sh) | stand launcher (restarts after a crash) |
| [tests/](tests/) | core, stand, brain and tuning tests |

## Sources

- **Brain model:** Shiu et al. 2024, *A Drosophila computational brain model reveals
  sensorimotor processing*, Nature —
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Connectome:** FlyWire v783 — Dorkenwald et al. 2024, *Neuronal wiring diagram of an
  adult brain*, Nature
- **Cell types and positions:** Schlegel et al. 2024, *Whole-brain annotation and
  multi-connectome cell typing of Drosophila*, Nature —
  [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Visual column assignment:** Matsliah et al. 2024, *Neuronal parts list and wiring
  diagram for a visual system*, Nature — FlyWire Codex

The FlyWire data is not included in this repository; `brain-download` fetches it from its
sources, and it is subject to their own terms of use.

## License

[GNU Affero General Public License v3.0](LICENSE)
