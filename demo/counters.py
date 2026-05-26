"""Running counters consumed by the dashboard header.

Tracks, frame-by-frame, the metrics a clinician cares about while watching
the demo:

  - Ground-truth FoG episodes seen and how many produced at least one cue.
  - Total cue events the FSM has emitted.
  - Cumulative cue duration (seconds the speaker has been on).
  - Mean latency from a GT episode's onset to the first cue inside it.

Latency is computed per episode: when GT rises 0→1 we mark `t_onset`; the
first cue-active frame inside that episode gives `t_cue`. If a GT episode
ends without any cue overlap, it is recorded as "missed" and excluded
from the latency average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from demo.replay import SAMPLE_RATE_HZ


@dataclass
class Counters:
    gt_episodes: int = 0
    gt_episodes_cued: int = 0
    cue_starts: int = 0
    cue_frames: int = 0
    _latencies_s: list[float] = field(default_factory=list)

    # Internal — current GT episode tracking
    _in_gt: bool = False
    _gt_onset_t: float = 0.0
    _gt_was_cued: bool = False
    _gt_cue_onset_t: float | None = None

    def reset(self) -> None:
        self.gt_episodes = 0
        self.gt_episodes_cued = 0
        self.cue_starts = 0
        self.cue_frames = 0
        self._latencies_s.clear()
        self._in_gt = False
        self._gt_was_cued = False
        self._gt_cue_onset_t = None

    def update(self, t: float, gt: int, cue_active: bool, cue_started: bool) -> None:
        if cue_started:
            self.cue_starts += 1
        if cue_active:
            self.cue_frames += 1

        if gt == 1 and not self._in_gt:
            self._in_gt = True
            self._gt_onset_t = t
            self._gt_was_cued = False
            self._gt_cue_onset_t = None
            self.gt_episodes += 1

        if self._in_gt and cue_active and not self._gt_was_cued:
            self._gt_was_cued = True
            self.gt_episodes_cued += 1
            self._gt_cue_onset_t = t
            self._latencies_s.append(t - self._gt_onset_t)

        if gt == 0 and self._in_gt:
            self._in_gt = False

    @property
    def cue_seconds(self) -> float:
        return self.cue_frames / SAMPLE_RATE_HZ

    @property
    def mean_latency_s(self) -> float | None:
        return (sum(self._latencies_s) / len(self._latencies_s)
                if self._latencies_s else None)

    def format_line(self) -> str:
        lat = self.mean_latency_s
        lat_str = f"{lat:.2f} s" if lat is not None else "—"
        return (f"GT episodes: {self.gt_episodes_cued}/{self.gt_episodes} cued  "
                f"·  Cue events: {self.cue_starts}  "
                f"·  Total cue: {self.cue_seconds:.1f} s  "
                f"·  Mean latency: {lat_str}")
