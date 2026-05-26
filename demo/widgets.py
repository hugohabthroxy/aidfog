"""pyqtgraph panels for the demo dashboard.

`TimelinePanel` owns a scrolling 10-second window of four stacked plots:

    1. Probability + GT shading + threshold reference lines
    2. Binary input (post-hysteresis, fixed)
    3. FSM state strip (coloured by state)
    4. Acceleration magnitude (proof of life)

Data is fed one frame at a time via `append_frame()`. The widget keeps an
internal ring buffer; only the visible window is plotted on each update.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from demo.fsm import State

WINDOW_S = 10.0
SAMPLE_RATE_HZ = 60
WINDOW_FRAMES = int(WINDOW_S * SAMPLE_RATE_HZ)

# State → fill color (RGBA, semi-transparent)
STATE_COLORS = {
    State.IDLE:        (200, 200, 200, 60),
    State.CUEING:      (220, 70, 70, 200),
    State.CUEING_TAIL: (240, 160, 70, 180),
    State.REFRACTORY:  (130, 130, 220, 120),
}
STATE_INDEX = {s: i for i, s in enumerate(
    [State.IDLE, State.REFRACTORY, State.CUEING_TAIL, State.CUEING]
)}


class TimelinePanel(QtWidgets.QWidget):
    """Four stacked scrolling plots, fed frame-by-frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        self._layout = pg.GraphicsLayoutWidget()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._layout)

        self._n = WINDOW_FRAMES
        self._buf_t = np.zeros(self._n, dtype=np.float64)
        self._buf_prob = np.full(self._n, np.nan, dtype=np.float64)
        self._buf_label = np.zeros(self._n, dtype=np.int8)
        self._buf_binary = np.zeros(self._n, dtype=np.int8)
        self._buf_state_idx = np.zeros(self._n, dtype=np.int8)
        self._buf_state_cue = np.zeros(self._n, dtype=np.int8)
        self._buf_acc = np.zeros(self._n, dtype=np.float64)
        self._head = 0
        self._filled = 0

        self._make_plots()

    def _make_plots(self):
        # Row 0 — Probability (tall)
        self._p_prob = self._layout.addPlot(row=0, col=0)
        self._p_prob.setLabel("left", "FoG prob")
        self._p_prob.setYRange(-0.05, 1.05)
        self._p_prob.showGrid(x=True, y=True, alpha=0.3)
        self._p_prob.hideAxis("bottom")
        self._curve_prob = self._p_prob.plot(pen=pg.mkPen("#1f77b4", width=1.5))
        # Threshold reference lines (informational, not interactive in this demo)
        self._p_prob.addLine(y=0.7, pen=pg.mkPen("#d62728", width=1.0,
                                                  style=QtCore.Qt.PenStyle.DashLine))
        self._p_prob.addLine(y=0.3, pen=pg.mkPen("#d62728", width=1.0,
                                                  style=QtCore.Qt.PenStyle.DotLine))
        # GT shading uses a FillBetweenItem against a flat zero baseline,
        # rendered as a step curve scaled to fill the plot
        self._gt_curve = pg.PlotCurveItem(pen=pg.mkPen(None),
                                          fillLevel=0,
                                          brush=pg.mkBrush(255, 80, 80, 60))
        self._p_prob.addItem(self._gt_curve)

        # Row 1 — Binary input
        self._p_bin = self._layout.addPlot(row=1, col=0)
        self._p_bin.setLabel("left", "binary\ninput")
        self._p_bin.setYRange(-0.1, 1.1)
        self._p_bin.setMaximumHeight(70)
        self._p_bin.hideAxis("bottom")
        self._p_bin.showGrid(x=True, y=True, alpha=0.3)
        self._curve_bin = pg.PlotCurveItem(pen=pg.mkPen("#888", width=1.0),
                                           fillLevel=0,
                                           brush=pg.mkBrush(60, 60, 60, 120))
        self._p_bin.addItem(self._curve_bin)

        # Row 2 — FSM state (taller; the "main" panel)
        self._p_state = self._layout.addPlot(row=2, col=0)
        self._p_state.setLabel("left", "FSM\nstate")
        self._p_state.setYRange(-0.5, 3.5)
        self._p_state.getAxis("left").setTicks([[
            (0, "IDLE"), (1, "REFR"), (2, "TAIL"), (3, "CUE"),
        ]])
        self._p_state.hideAxis("bottom")
        self._p_state.showGrid(x=True, y=True, alpha=0.2)
        self._curve_state = pg.PlotCurveItem(pen=pg.mkPen("#444", width=1.2))
        self._p_state.addItem(self._curve_state)
        # A second curve, only filled where cue is active, in red, to make
        # the "speaker on" times jump out visually
        self._curve_cue_fill = pg.PlotCurveItem(
            pen=pg.mkPen(None),
            fillLevel=0,
            brush=pg.mkBrush(220, 70, 70, 130),
        )
        self._p_state.addItem(self._curve_cue_fill)

        # Row 3 — Acceleration magnitude
        self._p_acc = self._layout.addPlot(row=3, col=0)
        self._p_acc.setLabel("left", "‖acc‖")
        self._p_acc.setLabel("bottom", "time (s)")
        self._p_acc.setMaximumHeight(90)
        self._p_acc.showGrid(x=True, y=True, alpha=0.3)
        self._curve_acc = self._p_acc.plot(pen=pg.mkPen("#2ca02c", width=1.0))

        # Link x-axes
        for p in (self._p_bin, self._p_state, self._p_acc):
            p.setXLink(self._p_prob)

    def reset(self):
        self._buf_t[:] = 0
        self._buf_prob[:] = np.nan
        self._buf_label[:] = 0
        self._buf_binary[:] = 0
        self._buf_state_idx[:] = 0
        self._buf_state_cue[:] = 0
        self._buf_acc[:] = 0
        self._head = 0
        self._filled = 0

    def append_frame(self, t: float, prob: float, label: int, binary: int,
                     state: State, cue_active: bool, acc: float):
        i = self._head
        self._buf_t[i] = t
        self._buf_prob[i] = prob
        self._buf_label[i] = label
        self._buf_binary[i] = binary
        self._buf_state_idx[i] = STATE_INDEX[state]
        self._buf_state_cue[i] = 3 if cue_active else 0
        self._buf_acc[i] = acc
        self._head = (i + 1) % self._n
        if self._filled < self._n:
            self._filled += 1

    def redraw(self):
        """Re-render the current window from the ring buffer."""
        if self._filled == 0:
            return
        # Reconstruct in chronological order
        if self._filled < self._n:
            sl = slice(0, self._filled)
            t = self._buf_t[sl]
            prob = self._buf_prob[sl]
            label = self._buf_label[sl]
            binary = self._buf_binary[sl]
            state = self._buf_state_idx[sl]
            cue = self._buf_state_cue[sl]
            acc = self._buf_acc[sl]
        else:
            idx = (np.arange(self._n) + self._head) % self._n
            t = self._buf_t[idx]
            prob = self._buf_prob[idx]
            label = self._buf_label[idx]
            binary = self._buf_binary[idx]
            state = self._buf_state_idx[idx]
            cue = self._buf_state_cue[idx]
            acc = self._buf_acc[idx]

        self._curve_prob.setData(t, prob)
        # GT shading: a 0/1 curve filled down to 0; scale to plot top
        self._gt_curve.setData(t, label.astype(float) * 1.05)
        self._curve_bin.setData(t, binary)
        self._curve_state.setData(t, state.astype(float), stepMode="right")
        # Cue-active fill: drop to NaN where not cueing so the fill clips
        cue_f = cue.astype(float)
        cue_f[cue == 0] = np.nan
        self._curve_cue_fill.setData(t, cue_f, stepMode=False, connect="finite")
        self._curve_acc.setData(t, acc)

        # Pin x-range to a trailing 10 s window
        x_max = float(t[-1])
        x_min = max(0.0, x_max - WINDOW_S)
        self._p_prob.setXRange(x_min, x_max, padding=0)
