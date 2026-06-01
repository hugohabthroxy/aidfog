"""Demo dashboard entry point.

    .venv/bin/python -m demo.app --trial data/project_AidFOG/trial_4
    .venv/bin/python -m demo.app --trial ... --no-ble   # skip BLE entirely

Single-window PyQt6 app. Replays one trial's binary stream at 60 Hz, runs
the cueing FSM on top, draws the four-row timeline, sends start/stop
commands to the PineBuds Pro via the existing BudsHandler.
"""

from __future__ import annotations

import argparse
import os
import sys

from PyQt6 import QtCore, QtWidgets

from demo.ble import BLEBridge
from demo.controls import ControlPanel
from demo.counters import Counters
from demo.fsm import CueingFSM, DemoConfig
from demo.hysteresis import HysteresisFilter
from demo.replay import SAMPLE_RATE_HZ, list_trials, load_trial, load_trial_from_npy
from demo.widgets import TimelinePanel

_SPEEDS = [0.5, 1.0, 2.0, 4.0]


class DemoMainWindow(QtWidgets.QMainWindow):
    def __init__(self, trials_root: str | None = None,
                 initial_trial: str | None = None,
                 npy_path: str | None = None,
                 speed: float = 1.0, ble: BLEBridge | None = None):
        super().__init__()
        self.setWindowTitle("AidFOG Cueing Demo")
        self.resize(1280, 820)

        self._trials_root = trials_root
        if npy_path:
            self._trial_paths = [npy_path]
            initial = npy_path
            self._trial = load_trial_from_npy(npy_path)
        else:
            self._trial_paths = list_trials(trials_root or "")
            if not self._trial_paths:
                raise SystemExit(f"no trial folders with aidfog_ai.hdf5 in {trials_root}")
            initial = initial_trial if initial_trial in self._trial_paths \
                else self._trial_paths[0]
            self._trial = load_trial(initial)
        self._config = DemoConfig()
        self._fsm = CueingFSM(self._config)
        self._hyst = HysteresisFilter(
            enter_thresh=self._config.hyst_enter_thresh,
            exit_thresh=self._config.hyst_exit_thresh,
        )
        self._counters = Counters()
        self._frame_idx = 0
        self._speed = speed
        self._ble = ble
        self._is_playing = True

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)

        # Playback toolbar
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self._trial_picker = QtWidgets.QComboBox()
        for p in self._trial_paths:
            self._trial_picker.addItem(os.path.basename(p), p)
        self._trial_picker.setCurrentIndex(self._trial_paths.index(initial))
        self._trial_picker.currentIndexChanged.connect(self._on_trial_changed)
        bar.addWidget(QtWidgets.QLabel("Trial:"))
        bar.addWidget(self._trial_picker)

        self._play_btn = QtWidgets.QPushButton("⏸ Pause")
        self._play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self._play_btn)

        self._restart_btn = QtWidgets.QPushButton("⟲ Restart")
        self._restart_btn.clicked.connect(self._restart)
        bar.addWidget(self._restart_btn)

        self._metronome_btn = QtWidgets.QPushButton("♩ Push metronome")
        self._metronome_btn.setToolTip(
            "Re-send the 60 BPM metronome config to the earbuds (100 ms beep + "
            "900 ms gap × 255). Use if the buds are stuck playing single beeps.")
        self._metronome_btn.clicked.connect(self._push_metronome)
        self._metronome_btn.setEnabled(bool(ble and ble.connected))
        bar.addWidget(self._metronome_btn)

        bar.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for s in _SPEEDS:
            self._speed_combo.addItem(f"{s:g}×", s)
        self._speed_combo.setCurrentIndex(_SPEEDS.index(speed) if speed in _SPEEDS else 1)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        bar.addWidget(self._speed_combo)
        bar.addStretch(1)
        outer.addLayout(bar)

        header = QtWidgets.QLabel(self._trial_header())
        header.setStyleSheet("font-family: monospace; padding: 4px;")
        outer.addWidget(header)
        self._header = header

        ble_ok = bool(ble and ble.connected)
        ble_msg = ("BLE connected — cues will play on PineBuds"
                   if ble_ok else
                   (ble.error if ble and ble.error else
                    "BLE disabled — visual-only demo"))
        banner = QtWidgets.QLabel(("● " if ble_ok else "○ ") + ble_msg)
        banner.setStyleSheet(
            "padding: 4px; border-radius: 4px; "
            + ("background-color: #e6f4ea; color: #1b5e20;" if ble_ok
               else "background-color: #fff3e0; color: #6d4c00;"))
        outer.addWidget(banner)

        self._counter_label = QtWidgets.QLabel("")
        self._counter_label.setStyleSheet(
            "font-family: monospace; padding: 4px; "
            "background-color: #f0f0f0; border-radius: 4px;")
        outer.addWidget(self._counter_label)
        self._refresh_counter_label()

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(split, stretch=1)

        self._panel = TimelinePanel()
        split.addWidget(self._panel)

        self._controls = ControlPanel(self._config)
        controls_wrap = QtWidgets.QWidget()
        cw_layout = QtWidgets.QVBoxLayout(controls_wrap)
        cw_layout.setContentsMargins(0, 0, 0, 0)
        cw_layout.addWidget(self._controls)
        split.addWidget(controls_wrap)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 320])

        ms = max(1, int(1000 / (SAMPLE_RATE_HZ * speed)))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(ms)

    def _trial_header(self) -> str:
        tr = self._trial
        return (f"trial={tr.name}  duration={tr.duration_s:.1f}s  "
                f"GT FoG={tr.fog_fraction:.3f}  "
                f"binary positive={tr.binary.mean():.3f}  "
                f"speed={self._speed:.1f}x")

    def _tick(self):
        i = self._frame_idx
        if i >= self._trial.n_samples:
            self._timer.stop()
            return
        # Hot-reload hysteresis thresholds and derive binary live from prob
        self._hyst.enter_thresh = self._config.hyst_enter_thresh
        self._hyst.exit_thresh = self._config.hyst_exit_thresh
        b = int(self._hyst.step(float(self._trial.probability[i])))
        t = float(self._trial.t[i])
        gt = int(self._trial.fog_label[i])
        res = self._fsm.step(b)
        cue_started = bool(res.command and res.command.action == "start")
        if res.command and self._ble:
            self._ble.send(res.command)
        self._counters.update(t, gt, res.cue_active, cue_started)
        self._panel.append_frame(
            t=t,
            prob=float(self._trial.probability[i]),
            label=gt,
            binary=b,
            state=res.state,
            cue_active=res.cue_active,
            acc=float(self._trial.acc_magnitude[i]),
        )
        # Repaint at ~10 Hz to keep CPU low
        if i % 6 == 0:
            self._panel.redraw()
            self._refresh_counter_label()
        self._frame_idx += 1

    def _refresh_counter_label(self):
        self._counter_label.setText(self._counters.format_line())

    def _toggle_play(self):
        self._is_playing = not self._is_playing
        if self._is_playing:
            self._timer.start()
            self._play_btn.setText("⏸ Pause")
        else:
            self._timer.stop()
            self._play_btn.setText("▶ Play")

    def _push_metronome(self):
        if self._ble:
            self._ble.send_metronome_config()

    def _restart(self):
        # Stop any in-flight cue before rewinding
        if self._ble:
            from demo.fsm import Command
            self._ble.send(Command("stop"))
        self._fsm.reset()
        self._hyst.reset()
        self._counters.reset()
        self._panel.reset()
        self._frame_idx = 0
        if not self._is_playing:
            self._toggle_play()
        self._refresh_counter_label()
        self._header.setText(self._trial_header())

    def _on_trial_changed(self, idx: int):
        path = self._trial_picker.itemData(idx)
        self._trial = (load_trial_from_npy(path) if path.endswith(".npy")
                       else load_trial(path))
        self._restart()

    def _on_speed_changed(self, idx: int):
        self._speed = float(self._speed_combo.itemData(idx))
        ms = max(1, int(1000 / (SAMPLE_RATE_HZ * self._speed)))
        self._timer.setInterval(ms)
        self._header.setText(self._trial_header())

    def closeEvent(self, event):  # noqa: N802 — Qt API
        if self._ble:
            self._ble.shutdown()
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="data/project_AidFOG",
                    help="Folder containing trial_* subfolders (default: data/project_AidFOG)")
    ap.add_argument("--trial", default=None,
                    help="Optional initial trial folder (default: first in --trials)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Initial playback speed (0.5 / 1.0 / 2.0 / 4.0)")
    ap.add_argument("--no-ble", action="store_true",
                    help="Skip the BudsHandler subprocess (visual-only)")
    ap.add_argument("--ble-address", default=None,
                    help="Override PineBuds MAC (default: from demo.ble)")
    ap.add_argument("--npy", default=None,
                    help="Load a trial from a _raw.npy file (Alex's stream format). "
                         "Sibling _label.npy and _acc.npy are auto-derived from the name.")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    ble: BLEBridge | None = None
    if not args.no_ble:
        ble = BLEBridge(address=args.ble_address) if args.ble_address \
            else BLEBridge()
        ble.start(timeout_s=8.0)
        # Force the firmware into 60 BPM metronome mode regardless of any stale
        # config it may have retained from a previous session. Cheap, idempotent.
        if ble.connected:
            ble.send_metronome_config()

    w = DemoMainWindow(trials_root=args.trials, initial_trial=args.trial,
                       npy_path=args.npy, speed=args.speed, ble=ble)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
