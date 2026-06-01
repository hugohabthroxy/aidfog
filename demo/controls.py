"""Right-side control panel: FSM mode toggle, sliders, presets.

Every control mutates the shared `DemoConfig` instance in place. The FSM
reads the config fresh on every step, so changes take effect on the next
frame — no restart needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6 import QtCore, QtWidgets

from demo.fsm import DemoConfig, FSMMode


@dataclass
class _SliderSpec:
    attr: str
    label: str
    minimum: int
    maximum: int
    suffix: str = ""

    def value_label(self, v: int) -> str:
        if self.suffix == "s":
            return f"{v / 60:.2f} s ({v} fr)"
        return f"{v}{self.suffix}"


_HYSTERESIS_SLIDERS = [
    _SliderSpec("hyst_enter_thresh", "Enter threshold", 1, 120, ""),
    _SliderSpec("hyst_exit_thresh",  "Exit threshold",  1, 60,  ""),
]
_FOUR_STATE_SLIDERS = [
    _SliderSpec("entry_consec",       "Entry consecutive",    1, 30,  ""),
    _SliderSpec("cueing_tail_frames", "Cueing tail",          0, 600, "s"),
    _SliderSpec("refractory_frames",  "Refractory",           0, 600, "s"),
]
_DEFOG_SLIDERS = [
    _SliderSpec("defog_cue_frames",        "FSM A cue duration", 60,  1200, "s"),
    _SliderSpec("defog_refractory_frames", "FSM A refractory",   0,   1200, "s"),
]
_METRONOME_SLIDERS = [
    _SliderSpec("volume",        "Volume",   0,   100, "%"),
    _SliderSpec("tone_id",       "Tone ID",  0,   5,   ""),
    _SliderSpec("metronome_bpm", "Tempo",    30,  180, " BPM"),
]


class ControlPanel(QtWidgets.QWidget):
    """Owns sliders, mode toggle, and preset buttons; mutates DemoConfig."""

    config_changed = QtCore.pyqtSignal()

    def __init__(self, config: DemoConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._sliders: dict[str, tuple[QtWidgets.QSlider, QtWidgets.QLabel, _SliderSpec]] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Mode toggle ---
        mode_group = QtWidgets.QGroupBox("Controller")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self._mode_4state = QtWidgets.QRadioButton("FSM B (ours)")
        self._mode_defog = QtWidgets.QRadioButton("FSM A (DeFOG)")
        self._mode_4state.setChecked(config.mode == FSMMode.FOUR_STATE)
        self._mode_defog.setChecked(config.mode == FSMMode.DEFOG)
        self._mode_4state.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_4state)
        mode_layout.addWidget(self._mode_defog)
        layout.addWidget(mode_group)

        # --- Presets ---
        preset_group = QtWidgets.QGroupBox("Presets")
        preset_layout = QtWidgets.QGridLayout(preset_group)
        for i, name in enumerate(_PRESETS.keys()):
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _checked=False, n=name: self._apply_preset(n))
            preset_layout.addWidget(btn, i // 2, i % 2)
        layout.addWidget(preset_group)

        # --- Sliders grouped by purpose ---
        self._hyst_box = self._make_slider_group(
            "Hysteresis (upstream)", _HYSTERESIS_SLIDERS)
        self._four_state_box = self._make_slider_group(
            "FSM B parameters", _FOUR_STATE_SLIDERS)
        self._defog_box = self._make_slider_group(
            "FSM A parameters", _DEFOG_SLIDERS)
        self._metronome_box = self._make_slider_group(
            "Metronome", _METRONOME_SLIDERS)
        layout.addWidget(self._hyst_box)
        layout.addWidget(self._four_state_box)
        layout.addWidget(self._defog_box)
        layout.addWidget(self._metronome_box)

        self._refresh_mode_visibility()
        layout.addStretch(1)

    def _make_slider_group(self, title: str, specs: list[_SliderSpec]) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(box)
        grid.setVerticalSpacing(4)
        for row, spec in enumerate(specs):
            lbl = QtWidgets.QLabel(spec.label)
            val_lbl = QtWidgets.QLabel("")
            val_lbl.setMinimumWidth(110)
            val_lbl.setStyleSheet("font-family: monospace;")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(spec.minimum)
            slider.setMaximum(spec.maximum)
            slider.setValue(int(getattr(self._config, spec.attr)))
            slider.valueChanged.connect(
                lambda v, s=spec: self._on_slider(s, v))
            grid.addWidget(lbl,    row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val_lbl, row, 2)
            self._sliders[spec.attr] = (slider, val_lbl, spec)
            val_lbl.setText(spec.value_label(slider.value()))
        return box

    def _on_slider(self, spec: _SliderSpec, value: int):
        setattr(self._config, spec.attr, int(value))
        _, val_lbl, _ = self._sliders[spec.attr]
        val_lbl.setText(spec.value_label(value))
        self.config_changed.emit()

    def _on_mode_changed(self):
        self._config.mode = (FSMMode.FOUR_STATE if self._mode_4state.isChecked()
                             else FSMMode.DEFOG)
        self._refresh_mode_visibility()
        self.config_changed.emit()

    def _refresh_mode_visibility(self):
        is_4 = self._config.mode == FSMMode.FOUR_STATE
        self._four_state_box.setEnabled(is_4)
        self._defog_box.setEnabled(not is_4)

    def _apply_preset(self, name: str):
        preset = _PRESETS[name]
        for attr, value in preset.items():
            if attr == "mode":
                self._config.mode = value
                self._mode_4state.setChecked(value == FSMMode.FOUR_STATE)
                self._mode_defog.setChecked(value == FSMMode.DEFOG)
                continue
            setattr(self._config, attr, value)
            if attr in self._sliders:
                slider, val_lbl, spec = self._sliders[attr]
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                val_lbl.setText(spec.value_label(int(value)))
        self._refresh_mode_visibility()
        self.config_changed.emit()


# Canonical configurations the demo can switch to with one click.
# Frame numbers assume 60 Hz sampling.
_PRESETS: dict[str, dict] = {
    "Ours (FSM B)": {
        "mode": FSMMode.FOUR_STATE,
        "hyst_enter_thresh": 20,      # Alex's published defaults
        "hyst_exit_thresh": 5,
        "entry_consec": 1,
        "cueing_tail_frames": 30,
        "refractory_frames": 60,
        "volume": 80,
        "tone_id": 0,
    },
    "FSM A (10/5 s)": {
        "mode": FSMMode.DEFOG,
        "hyst_enter_thresh": 20,
        "hyst_exit_thresh": 5,
        "defog_cue_frames": 600,
        "defog_refractory_frames": 300,
        "volume": 80,
        "tone_id": 0,
    },
    "Aggressive": {
        "mode": FSMMode.FOUR_STATE,
        "hyst_enter_thresh": 5,       # fires fast — accepts more false positives
        "hyst_exit_thresh": 12,
        "entry_consec": 1,
        "cueing_tail_frames": 60,
        "refractory_frames": 15,
        "volume": 90,
        "tone_id": 0,
    },
    "Conservative": {
        "mode": FSMMode.FOUR_STATE,
        "hyst_enter_thresh": 40,      # waits for strong signal — may miss short freezes
        "hyst_exit_thresh": 3,
        "entry_consec": 6,
        "cueing_tail_frames": 0,
        "refractory_frames": 300,
        "volume": 70,
        "tone_id": 0,
    },
}
