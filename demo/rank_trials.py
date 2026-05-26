"""Rank trials by how dramatically they showcase different cueing configs.

Runs every trial through both 4-state and DeFOG FSMs on its post-hysteresis
binary stream, then ranks by audible-contrast metrics:

  - Δ total cue audio (s)     larger → bigger overall difference
  - Δ event count             larger → one strategy fires far more often
  - DeFOG over-cue ratio      DeFOG_audio / FoG_duration; >2 means heavy over-cueing
  - episode count             more episodes → more chances to hear the difference

A clean demo trial scores high on Δ-audio AND has many short episodes.

Usage:
    .venv\\Scripts\\python.exe -m demo.rank_trials --trials data\\project_AidFOG
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from demo.fsm import CueingFSM, DemoConfig, FSMMode
from demo.replay import SAMPLE_RATE_HZ, list_trials, load_trial


def _simulate(binary: np.ndarray, mode: FSMMode) -> tuple[int, int]:
    """Return (cue_starts, cue_frames) for the given controller."""
    fsm = CueingFSM(DemoConfig(mode=mode))
    starts = frames = 0
    for b in binary:
        res = fsm.step(int(b))
        if res.command and res.command.action == "start":
            starts += 1
        if res.cue_active:
            frames += 1
    return starts, frames


def _episode_stats(label: np.ndarray) -> tuple[int, float]:
    """Return (n_episodes, median_duration_s) for a GT label."""
    diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    s = np.where(diff == 1)[0]
    e = np.where(diff == -1)[0]
    if len(s) == 0:
        return 0, 0.0
    durations = (e - s) / SAMPLE_RATE_HZ
    return len(s), float(np.median(durations))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="data/project_AidFOG")
    args = ap.parse_args()

    rows = []
    for path in list_trials(args.trials):
        try:
            tr = load_trial(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if tr.binary.sum() == 0:
            continue  # nothing to cue
        starts_4, frames_4 = _simulate(tr.binary, FSMMode.FOUR_STATE)
        starts_d, frames_d = _simulate(tr.binary, FSMMode.DEFOG)
        n_eps, med_dur = _episode_stats(tr.fog_label)
        fog_s = float(tr.fog_label.sum()) / SAMPLE_RATE_HZ
        cue_4_s = frames_4 / SAMPLE_RATE_HZ
        cue_d_s = frames_d / SAMPLE_RATE_HZ
        rows.append({
            "trial":      os.path.basename(path),
            "n_eps":      n_eps,
            "med_freeze": med_dur,
            "fog_s":      fog_s,
            "ours_s":     cue_4_s,
            "defog_s":    cue_d_s,
            "delta_s":    cue_d_s - cue_4_s,
            "delta_evt":  starts_4 - starts_d,
            "over_ratio": (cue_d_s / fog_s) if fog_s > 0 else 0.0,
        })

    if not rows:
        print("no usable trials found in", args.trials)
        return

    rows.sort(key=lambda r: abs(r["delta_s"]), reverse=True)

    hdr = (f"{'trial':<28} {'n_eps':>6} {'med_fz':>7} {'FoG_s':>7} "
           f"{'Ours_s':>8} {'DeFOG_s':>9} {'Δ_audio_s':>11} {'over_x':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['trial'][:28]:<28} "
              f"{r['n_eps']:>6} "
              f"{r['med_freeze']:>7.2f} "
              f"{r['fog_s']:>7.1f} "
              f"{r['ours_s']:>8.1f} "
              f"{r['defog_s']:>9.1f} "
              f"{r['delta_s']:>+11.1f} "
              f"{r['over_ratio']:>7.2f}")
    print()
    best = rows[0]
    print(f"Recommended demo trial: {best['trial']}")
    print(f"  → Ours plays {best['ours_s']:.1f} s of cue audio")
    print(f"  → DeFOG plays {best['defog_s']:.1f} s of cue audio "
          f"({best['over_ratio']:.1f}× the GT freeze time)")
    print(f"  → audible difference: {abs(best['delta_s']):.1f} s of audio "
          f"({'DeFOG more' if best['delta_s'] > 0 else 'Ours more'})")


if __name__ == "__main__":
    main()
