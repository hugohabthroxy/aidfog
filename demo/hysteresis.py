"""Stateful hysteresis filter mirroring Alex's `HysteresisFilter`.

Threshold on probability is a fixed 0.5 (per Alex's pipeline). The two
clinician-tunable knobs are `enter_thresh` and `exit_thresh` — how many
consecutive 1s / 0s are needed to enter / leave the FoG state. Attributes
can be mutated between calls; the demo dashboard does this when sliders
move so the live binary reshapes in real time.
"""

from __future__ import annotations


PROB_THRESHOLD = 0.5


class HysteresisFilter:
    def __init__(self, enter_thresh: int = 20, exit_thresh: int = 5):
        self.enter_thresh = enter_thresh
        self.exit_thresh = exit_thresh
        self.in_fog = False
        self.consec1 = 0
        self.consec0 = 0

    def reset(self) -> None:
        self.in_fog = False
        self.consec1 = 0
        self.consec0 = 0

    def step(self, prob: float) -> int:
        """One sample in, one binary out. Threshold on prob is fixed at 0.5."""
        raw = 1 if prob >= PROB_THRESHOLD else 0
        if not self.in_fog:
            if raw == 1:
                self.consec1 += 1
            else:
                self.consec1 = 0
            if self.consec1 >= self.enter_thresh:
                self.in_fog = True
                self.consec0 = 0
            return 1 if self.in_fog else 0
        else:
            if raw == 0:
                self.consec0 += 1
            else:
                self.consec0 = 0
            if self.consec0 >= self.exit_thresh:
                self.in_fog = False
                self.consec1 = 0
            return 1
