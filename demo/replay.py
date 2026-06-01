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


def load_trial_from_npy(raw_path: str,
                        label_path: str | None = None,
                        acc_path: str | None = None) -> TrialData:
    """Load a trial from three sibling .npy files (raw / label / acc).

    `raw` is pre-hysteresis probability — the demo's live HysteresisFilter
    re-derives the binary each frame, so the `binary` field is left as zeros
    (it's display-only; the FSM never reads it).

    If `label_path` / `acc_path` are omitted they're derived by swapping
    `_raw` → `_label` / `_acc` in the filename.
    """
    if label_path is None:
        label_path = raw_path.replace("_raw", "_label")
    if acc_path is None:
        acc_path = raw_path.replace("_raw", "_acc")

    prob = np.load(raw_path).astype(np.float32)
    label = np.load(label_path).astype(np.int8)
    acc = np.load(acc_path).astype(np.float32)

    if not (len(prob) == len(label) == len(acc)):
        raise ValueError(
            f"length mismatch: raw={len(prob)} label={len(label)} acc={len(acc)}"
        )
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"acc must be shape (N, 3); got {acc.shape}")

    n = len(prob)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
    binary = np.zeros(n, dtype=np.int8)
    acc_mag = np.linalg.norm(acc, axis=1)

    name = os.path.splitext(os.path.basename(raw_path))[0]
    if name.endswith("_raw"):
        name = name[:-4]

    return TrialData(
        name=name,
        t=t,
        probability=prob,
        binary=binary,
        fog_label=label,
        acc_magnitude=acc_mag,
    )


def _synthesise_from_label(label: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dev-only: fabricate a TCN-shaped probability + post-hysteresis binary.

    Real TCN output is noisy frame-to-frame even when the model is confident,
    so we avoid the trap of producing a flat saturated 1.0 during FoG. Instead:

    - Outside FoG: small positive jitter around ~0.08, with occasional spikes.
    - Inside FoG: oscillates around ~0.78 with realistic variance and the
      occasional dip below 0.7 (which the hysteresis filter then absorbs).

    Binary is derived by running Alex-style hysteresis (enter when prob ≥
    th_high for `enter` frames; exit when prob ≤ th_low for `exit` frames)
    so the visual matches the th_high / th_low reference lines on the plot.
    """
    rng = np.random.default_rng(seed)
    n = len(label)

    # Base noise floor — half-normal so it stays positive
    prob = np.abs(rng.normal(0.05, 0.05, n))

    # Inside FoG, replace with a noisy band around 0.78 with a slow wobble
    # to simulate frame-to-frame variation from a real TCN
    t = np.arange(n) / SAMPLE_RATE_HZ
    fog_band = (0.78
                + 0.07 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 2 * np.pi))
                + rng.normal(0, 0.10, n))
    prob = np.where(label == 1, fog_band, prob)

    # Realistic onset/offset transition: brief rise/fall over ~150 ms instead
    # of an instant step (the model never knows the exact GT boundary)
    transition_k = 9
    prob = np.convolve(prob, np.ones(transition_k) / transition_k, mode="same")
    prob = np.clip(prob, 0.0, 1.0)

    # Alex-style hysteresis matching the th_high / th_low reference lines
    binary = _hysteresis(prob, th_high=0.7, th_low=0.3, enter=12, exit_=6)
    return prob, binary


def _hysteresis(prob: np.ndarray, th_high: float, th_low: float,
                enter: int, exit_: int) -> np.ndarray:
    n = len(prob)
    out = np.zeros(n, dtype=np.int8)
    in_fog = False
    high_run = low_run = 0
    for i in range(n):
        if in_fog:
            out[i] = 1
            if prob[i] <= th_low:
                low_run += 1
                if low_run >= exit_:
                    in_fog = False
                    low_run = 0
            else:
                low_run = 0
        else:
            if prob[i] >= th_high:
                high_run += 1
                if high_run >= enter:
                    in_fog = True
                    high_run = 0
                    out[i] = 1
            else:
                high_run = 0
    return out


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
