# AidFOG Clinician Demo Dashboard

A standalone PyQt6 app that replays a recorded trial's fixed binary stream
(post-hysteresis output) and lets a clinician tune the cueing controller
live. Shows probability, ground truth, FSM state, and acceleration in a
scrolling timeline; sends real audio cues to the PineBuds Pro via BLE.

The goal is to characterise trade-offs between cueing strategies on the
same input — "given this patient stream, here's what FSM A does, here's
what FSM B does."

## Architecture

```
demo/
  app.py       PyQt6 main window + playback clock
  widgets.py   Four-row scrolling timeline (pyqtgraph)
  controls.py  Sliders, mode toggle, preset buttons
  counters.py  Running episode / cue / latency stats
  fsm.py       4-state and DeFOG-style cueing FSMs
  replay.py    HDF5 trial loader
  ble.py       BudsHandler subprocess bridge
  cli_smoke.py Headless validation of the data path
```

The FSM reads parameters from a shared `DemoConfig` instance on every
frame, so slider changes take effect immediately without restarting.
Changes affect future frames only — no retroactive re-simulation.

## Running on Windows

```powershell
.venv\Scripts\python.exe -m demo.app
```

Optional flags:

  `--trials DIR`         folder of `trial_*/` subfolders
                         (default: `data/project_AidFOG`)
  `--trial PATH`         start on a specific trial
  `--speed {0.5,1,2,4}`  initial playback speed
  `--no-ble`             skip BLE entirely (visual-only)
  `--ble-address MAC`    override the PineBuds MAC

Without `--no-ble`, the app spawns `BudsHandler` in a subprocess (same
module the live pipeline uses) and forwards FSM start/stop commands. If
BLE fails to connect within 8 s the banner shows a warning and the demo
degrades to visual-only without exiting.

## What the clinician sees

  1. **Probability** — model output with dashed/dotted reference lines
     at 0.7 / 0.3 and red shading over ground-truth FoG episodes.
  2. **Binary input** — the fixed post-hysteresis stream the FSM consumes.
  3. **FSM state** — IDLE / CUEING / CUEING_TAIL / REFRACTORY, with the
     CUEING rows highlighted in red where the speaker is on.

A header bar above shows running counters: episodes detected vs. total
GT, cue event count, total cue time, mean latency from GT onset to cue.

## Switching strategies

Preset buttons load canonical configs in one click:

  - **Ours (4-state)** — 1-frame entry, 500 ms tail, 1 s refractory.
  - **DeFOG (10/5 s)** — 10 s fixed cue, 5 s refractory (Zoetewei 2021).
  - **Aggressive** — long tail, short refractory.
  - **Conservative** — slower entry, no tail, longer refractory.

Or drag any slider to make a custom config. The mode radio buttons
switch between the 4-state and DeFOG-style controllers; switching mid-
trial resets the FSM to IDLE on the next frame.

## Notes

  - Sample rate is 60 Hz (matches `resources/buds.yml`).
  - Trials where neither the AI stream nor the replay IMU has valid
    data are silently filtered out of the picker.
  - On dev machines without working model output, the loader synthesises
    a plausible probability + binary from the ground-truth label and
    prints a warning to stderr. Real Windows recordings replace this.

## Quick smoke test (no GUI)

```bash
.venv/bin/python -m demo.cli_smoke data/project_AidFOG/trial_4
```

Prints per-FSM cue-event counts and total cue time. Useful for
verifying the data path without a Qt environment.
