"""HDF5 trial loader for the demo dashboard.

A recorded trial folder has:
  - aidfog_ai.hdf5      pytorch-worker/{logits, prediction, process_time_s}
                        + an aligned copy of dots-imu/{acceleration, fog_label, ...}
  - aidfog_replay.hdf5  the full-length replay IMU (we don't need this; the AI
                        file already contains the IMU rows that align with the
                        AI output).

We use the AI file as the single source of truth so probability, binary, IMU,
and ground-truth label are all the same length and same timeline. Padding
rows (process_time_s == 0) are trimmed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np

SAMPLE_RATE_HZ = 60


@dataclass
class TrialData:
    """One trial, all streams aligned, padding trimmed."""

    name: str
    t: np.ndarray              # seconds since first sample (N,)
    probability: np.ndarray    # FoG probability ∈ [0, 1]            (N,)
    binary: np.ndarray         # post-hysteresis 0/1                  (N,) int8
    fog_label: np.ndarray      # ground-truth 0/1                     (N,) int8
    acc_magnitude: np.ndarray  # ‖acc‖ from sensor 0                  (N,)

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def fog_fraction(self) -> float:
        return float(self.fog_label.mean()) if len(self.fog_label) else 0.0


def _softmax_prob(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax → class-1 probability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e[:, 1] / e.sum(axis=1)


def load_trial(trial_dir: str, synthesize_if_empty: bool = True) -> TrialData:
    """Load one trial folder. Raises FileNotFoundError if aidfog_ai.hdf5 missing.

    If the AI stream is empty (no valid rows) or all-NaN — which happens on
    recordings where the model didn't actually run — we fall back to the
    replay-side IMU + fog_label and synthesise a plausible probability +
    binary from the ground truth. This is a *dev-only* path so the dashboard
    can be exercised without real model output. A warning is printed.
    """
    ai_path = os.path.join(trial_dir, "aidfog_ai.hdf5")
    if not os.path.exists(ai_path):
        raise FileNotFoundError(f"missing {ai_path}")

    with h5py.File(ai_path, "r") as f:
        pt = f["aidfog_ai/pytorch-worker/process_time_s"][:].flatten()
        logits = f["aidfog_ai/pytorch-worker/logits"][:]
        prediction = f["aidfog_ai/pytorch-worker/prediction"][:].flatten()
        acc = f["aidfog_replay/dots-imu/acceleration"][:]
        label = f["aidfog_replay/dots-imu/fog_label"][:].flatten()

    ai_mask = pt > 0
    # "Real" AI data requires: rows exist, logits aren't NaN, AND the model
    # actually fired at some point. An all-zero prediction stream from an
    # earlier broken model is treated like no AI at all so the synthesis
    # fallback kicks in (useful when GT > 0 but predictions are flat).
    have_real_ai = (ai_mask.any()
                    and np.isfinite(logits[ai_mask]).all()
                    and prediction[ai_mask].any())

    if have_real_ai:
        pt = pt[ai_mask]
        logits = logits[ai_mask]
        prediction = prediction[ai_mask].astype(np.int8)
        acc = acc[ai_mask]
        label = label[ai_mask].astype(np.int8)
        t = pt - pt[0]
        prob = _softmax_prob(logits)
        binary = prediction
    else:
        if not synthesize_if_empty:
            raise ValueError(
                f"{ai_path}: AI stream is empty or all-NaN; pass "
                "synthesize_if_empty=True to fall back to synthetic data."
            )
        import sys
        print(f"warning: {ai_path} has no valid AI rows — synthesising "
              "probability + binary from ground-truth label (dev mode).",
              file=sys.stderr)
        # Use replay rows whose process_time_s is meaningful by checking the
        # acceleration sensor (all-zero rows = padding).
        keep = np.linalg.norm(acc[:, 0, :], axis=1) > 0
        acc = acc[keep]
        label = label[keep].astype(np.int8)
        n = len(label)
        t = np.arange(n) / SAMPLE_RATE_HZ
        prob, binary = _synthesise_from_label(label)

    acc_mag = np.linalg.norm(acc[:, 0, :], axis=1)

    return TrialData(
        name=os.path.basename(os.path.normpath(trial_dir)),
        t=t,
        probability=prob,
        binary=binary,
        fog_label=label,
        acc_magnitude=acc_mag,
    )


def _synthesise_from_label(label: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dev-only: fabricate probability + post-hysteresis binary from GT.

    Probability follows the label with a 12-frame (200 ms @ 60 Hz) onset/offset
    smoothing and gaussian noise. Binary is the probability passed through a
    tiny enter/exit hysteresis (entry=12, exit=6 consecutive frames > 0.5 /
    < 0.5). The result roughly resembles what a working detector would emit.
    """
    rng = np.random.default_rng(seed)
    n = len(label)

    # Smooth the GT onsets/offsets to look like a real probability rise/fall
    target = label.astype(np.float64)
    k = 12
    kernel = np.ones(k) / k
    prob = np.convolve(target, kernel, mode="same")
    prob = np.clip(prob + rng.normal(0, 0.05, n), 0.0, 1.0)
    # Push positive episodes a bit higher
    prob = np.where(label == 1, np.clip(prob + 0.3, 0.0, 1.0), prob)

    thresholded = (prob >= 0.5).astype(np.int8)
    binary = np.zeros(n, dtype=np.int8)
    enter, exit_ = 12, 6
    in_fog = False
    high_run = low_run = 0
    for i in range(n):
        if in_fog:
            binary[i] = 1
            if thresholded[i] == 0:
                low_run += 1
                if low_run >= exit_:
                    in_fog = False
                    low_run = 0
            else:
                low_run = 0
        else:
            if thresholded[i] == 1:
                high_run += 1
                if high_run >= enter:
                    in_fog = True
                    high_run = 0
                    binary[i] = 1
            else:
                high_run = 0
    return prob, binary


def list_trials(trials_root: str) -> list[str]:
    """Return sorted list of trial folders with usable data under `trials_root`.

    A trial is usable if either (a) the AI stream has any valid rows, or
    (b) the replay-side IMU + label have non-zero data we can synthesise
    from. Empty placeholder folders are silently dropped.
    """
    out = []
    if not os.path.isdir(trials_root):
        return out
    for name in sorted(os.listdir(trials_root)):
        path = os.path.join(trials_root, name)
        ai_path = os.path.join(path, "aidfog_ai.hdf5")
        if not (os.path.isdir(path) and os.path.exists(ai_path)):
            continue
        try:
            with h5py.File(ai_path, "r") as f:
                ai_pt = f["aidfog_ai/pytorch-worker/process_time_s"][:].flatten()
                acc = f["aidfog_replay/dots-imu/acceleration"][:]
        except Exception:
            continue
        if (ai_pt > 0).any():
            out.append(path)
            continue
        # No AI rows — usable only if the replay-side has non-zero acc
        keep = np.linalg.norm(acc[:, 0, :], axis=1) > 0
        if keep.any():
            out.append(path)
    return out
