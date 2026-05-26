"""Headless smoke test for replay + FSM.

Loads a trial, runs both FSMs on its binary stream, prints a one-line summary
per FSM. Validates the data path without needing a GUI or BLE.

    .venv/bin/python -m demo.cli_smoke data/project_AidFOG/trial_2
"""

from __future__ import annotations

import argparse

from demo.fsm import CueingFSM, DemoConfig, FSMMode
from demo.replay import SAMPLE_RATE_HZ, load_trial


def _summarise(trial, mode: FSMMode) -> str:
    config = DemoConfig(mode=mode)
    fsm = CueingFSM(config)
    n_start = n_stop = 0
    cue_frames = 0
    for b in trial.binary:
        res = fsm.step(int(b))
        if res.command and res.command.action == "start":
            n_start += 1
        elif res.command and res.command.action == "stop":
            n_stop += 1
        if res.cue_active:
            cue_frames += 1
    cue_seconds = cue_frames / SAMPLE_RATE_HZ
    return (f"{mode.value:>14s} | starts={n_start:3d} stops={n_stop:3d} "
            f"| total cue {cue_seconds:5.1f} s "
            f"({100 * cue_frames / len(trial.binary):.1f}% of trial)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir", help="Path to a trial folder containing aidfog_ai.hdf5")
    args = ap.parse_args()

    trial = load_trial(args.trial_dir)
    print(f"Trial: {trial.name}")
    print(f"  duration         {trial.duration_s:.1f} s "
          f"({trial.n_samples} samples @ {SAMPLE_RATE_HZ} Hz)")
    print(f"  GT FoG fraction  {trial.fog_fraction:.3f}")
    print(f"  binary positive  {trial.binary.mean():.3f}")
    print()
    print(_summarise(trial, FSMMode.FOUR_STATE))
    print(_summarise(trial, FSMMode.DEFOG))


if __name__ == "__main__":
    main()
