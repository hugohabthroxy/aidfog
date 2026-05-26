"""BLE bridge: spawns `BudsHandler` in a subprocess and forwards commands.

The handler expects the same set of multiprocessing events and queues that
the live `BudsPipeline` provides in `hermes/aidfog/pipeline.py`. We
replicate that bootstrap minus the HERMES ZMQ plumbing.

Designed to fail gracefully: if the import or process spawn fails, or if
the handler never signals `is_ready_event` within the timeout, `BLEBridge`
reports `connected=False` and `send()` becomes a no-op. The dashboard
remains usable as a visual-only demo.
"""

from __future__ import annotations

import logging
import os
import time
from multiprocessing import Event, Process, Queue
from typing import Any

from demo.fsm import Command

logger = logging.getLogger(__name__)

# Default device address / name comes from resources/buds.yml. Hard-coded
# here so the demo runs without parsing the HERMES yaml.
DEFAULT_DEVICE_NAME = "D&D TECH"
DEFAULT_DEVICE_ADDRESS = "12:34:56:C2:A2:30"  # right earbud from buds.yml


class BLEBridge:
    """Thin façade around the BudsHandler subprocess.

    Usage:
        bridge = BLEBridge()
        bridge.start(timeout_s=8.0)
        if bridge.connected:
            bridge.send(Command("start", volume=80))
        ...
        bridge.shutdown()
    """

    def __init__(self, device_name: str = DEFAULT_DEVICE_NAME,
                 address: str | None = DEFAULT_DEVICE_ADDRESS,
                 op_log_path: str | None = None):
        self._device_name = device_name
        self._address = address
        self._op_log_path = op_log_path

        self._proc: Process | None = None
        self._cmd_queue: "Queue[dict]" = Queue()
        self._status_queue: "Queue[tuple[float, int]]" = Queue()
        self._is_ready = Event()
        self._is_keep_data = Event()
        self._is_stop_new_data = Event()
        self._is_cleanup = Event()
        self._is_finished = Event()

        self._connected = False
        self._error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> str | None:
        return self._error

    def start(self, timeout_s: float = 8.0) -> bool:
        """Spawn the handler process and wait for is_ready (timeout → fail soft)."""
        try:
            from hermes.aidfog.controller import BudsHandler
            from hermes.utils.mp_utils import launch_handler
        except Exception as e:  # pragma: no cover — depends on hermes install
            self._error = f"hermes import failed: {e}"
            logger.warning(self._error)
            return False

        buds_kwargs: dict[str, Any] = {
            "device_name": self._device_name,
            "address": self._address,
        }
        try:
            self._proc = Process(
                target=launch_handler,
                args=(BudsHandler,),
                kwargs={
                    "buds": buds_kwargs,
                    "cueing_command_queue": self._cmd_queue,
                    "cueing_status_queue": self._status_queue,
                    "ref_time_s": time.time(),
                    "is_ready_event": self._is_ready,
                    "is_keep_data_event": self._is_keep_data,
                    "is_stop_new_data_event": self._is_stop_new_data,
                    "is_cleanup_event": self._is_cleanup,
                    "is_finished_event": self._is_finished,
                    "dt": 0.5,
                    "op_log_path": self._op_log_path,
                },
                daemon=True,
            )
            self._proc.start()
        except Exception as e:
            self._error = f"BudsHandler process spawn failed: {e}"
            logger.warning(self._error)
            return False

        if not self._is_ready.wait(timeout=timeout_s):
            self._error = f"BLE not ready within {timeout_s:.0f}s — running visual-only"
            logger.warning(self._error)
            self._connected = False
            return False

        self._connected = True
        return True

    def send(self, command: Command) -> None:
        """Push a command to the handler queue. No-op when not connected."""
        if not self._connected:
            return
        try:
            self._cmd_queue.put_nowait(command.to_dict())
        except Exception as e:
            logger.warning("BLE send failed: %s", e)

    def shutdown(self, timeout_s: float = 3.0) -> None:
        if self._proc is None:
            return
        try:
            self._is_cleanup.set()
            self._is_stop_new_data.set()
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                logger.warning("BudsHandler did not exit cleanly; terminating")
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        except Exception as e:
            logger.warning("shutdown error: %s", e)
