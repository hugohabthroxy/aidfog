"""Cueing finite-state machines for the demo.

Two controllers, both consuming the *same* fixed binary stream:

  FSM_B_4State  IDLE → CUEING → CUEING_TAIL → REFRACTORY → IDLE
                (the controller deployed in `hermes/aidfog/pipeline.py`)

  FSM_A_DeFOG   IDLE → CUEING(fixed duration) → REFRACTORY → IDLE
                (Zoetewei 2021 / DeFOG-style: cue plays for a fixed duration
                 regardless of whether the FoG episode is still active)

Both read parameters from a shared mutable `DemoConfig` instance so the
dashboard can hot-reload values mid-playback. Changes take effect on the
next frame; no retroactive re-simulation.

State transitions emit a `Command` ("start" / "stop") that the dashboard
forwards to BudsHandler over BLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FSMMode(str, Enum):
    FOUR_STATE = "4-state"
    DEFOG = "FSM B"


class State(str, Enum):
    IDLE = "IDLE"
    CUEING = "CUEING"
    CUEING_TAIL = "CUEING_TAIL"
    REFRACTORY = "REFRACTORY"


@dataclass
class DemoConfig:
    """All live-tunable parameters. Shared between UI + FSM; mutated in place."""

    mode: FSMMode = FSMMode.FOUR_STATE

    # Upstream hysteresis filter (Alex's HysteresisFilter). Applied to the
    # raw probability stream live, so changing these thresholds reshapes the
    # binary signal the FSM sees. Defaults match Alex's published values.
    hyst_enter_thresh: int = 20       # ~333 ms @ 60 Hz consecutive ones to enter
    hyst_exit_thresh: int = 5         # ~83 ms  @ 60 Hz consecutive zeros to exit

    # 4-state FSM
    entry_consec: int = 1
    cueing_tail_frames: int = 30      # ~500 ms @ 60 Hz
    refractory_frames: int = 60       # ~1 s   @ 60 Hz

    # DeFOG-style FSM
    defog_cue_frames: int = 600       # 10 s @ 60 Hz  (Zoetewei 2021)
    defog_refractory_frames: int = 300  # 5 s @ 60 Hz

    # Metronome (sent with every START command)
    volume: int = 80
    tone_id: int = 0


@dataclass
class Command:
    action: str          # "start" | "stop"
    tone_id: int = 0
    volume: int = 80

    def to_dict(self) -> dict:
        if self.action == "stop":
            return {"action": "stop"}
        return {"action": "start", "tone_id": self.tone_id, "volume": self.volume}


@dataclass
class StepResult:
    state: State
    cue_active: bool
    command: Command | None = None     # set on state transitions that change BLE


class CueingFSM:
    """Dispatches per-frame to whichever controller the config selects.

    Switching modes mid-trial resets the FSM to IDLE on the next step (cleaner
    than carrying state across two different topologies).
    """

    def __init__(self, config: DemoConfig):
        self._config = config
        self._mode = config.mode
        self._state = State.IDLE
        self._consec_high = 0
        self._cueing_tail_remaining = 0
        self._refractory_remaining = 0
        self._defog_cue_remaining = 0

    @property
    def state(self) -> State:
        return self._state

    def reset(self) -> None:
        self._state = State.IDLE
        self._consec_high = 0
        self._cueing_tail_remaining = 0
        self._refractory_remaining = 0
        self._defog_cue_remaining = 0

    def step(self, binary: int) -> StepResult:
        # Mode switch detected: reset and start fresh
        if self._config.mode != self._mode:
            self._mode = self._config.mode
            self.reset()

        if self._mode == FSMMode.FOUR_STATE:
            return self._step_4state(int(binary))
        return self._step_defog(int(binary))

    # ------------------------------------------------------------------ 4-state
    def _step_4state(self, binary: int) -> StepResult:
        cfg = self._config
        cmd: Command | None = None

        if self._state == State.IDLE:
            self._consec_high = self._consec_high + 1 if binary == 1 else 0
            if self._consec_high >= cfg.entry_consec:
                self._state = State.CUEING
                self._consec_high = 0
                cmd = Command("start", cfg.tone_id, cfg.volume)
        elif self._state == State.CUEING:
            if binary == 0:
                self._state = State.CUEING_TAIL
                self._cueing_tail_remaining = cfg.cueing_tail_frames
        elif self._state == State.CUEING_TAIL:
            if binary == 1:
                # FoG re-asserted before tail expired — restart cue
                self._state = State.CUEING
                cmd = Command("start", cfg.tone_id, cfg.volume)
            else:
                self._cueing_tail_remaining -= 1
                if self._cueing_tail_remaining <= 0:
                    self._state = State.REFRACTORY
                    self._refractory_remaining = cfg.refractory_frames
                    cmd = Command("stop")
        elif self._state == State.REFRACTORY:
            self._refractory_remaining -= 1
            if self._refractory_remaining <= 0:
                self._state = State.IDLE
                self._consec_high = 0

        cue_active = self._state in (State.CUEING, State.CUEING_TAIL)
        return StepResult(self._state, cue_active, cmd)

    # ------------------------------------------------------------------ DeFOG
    def _step_defog(self, binary: int) -> StepResult:
        cfg = self._config
        cmd: Command | None = None

        if self._state == State.IDLE:
            if binary == 1:
                self._state = State.CUEING
                self._defog_cue_remaining = cfg.defog_cue_frames
                cmd = Command("start", cfg.tone_id, cfg.volume)
        elif self._state == State.CUEING:
            self._defog_cue_remaining -= 1
            if self._defog_cue_remaining <= 0:
                self._state = State.REFRACTORY
                self._refractory_remaining = cfg.defog_refractory_frames
                cmd = Command("stop")
        elif self._state == State.REFRACTORY:
            self._refractory_remaining -= 1
            if self._refractory_remaining <= 0:
                self._state = State.IDLE

        cue_active = self._state == State.CUEING
        return StepResult(self._state, cue_active, cmd)
