"""
sim_bridge.py — TCP JSON bridge embedded inside the simulation.

A background thread listens on TCP port 9871.  Each sim tick the main
loop calls push_state() to update the shared snapshot.  External clients
(e.g. mcp_sim_server.py) open short-lived connections, send a single JSON
request, and receive a single JSON response — both terminated by newline.

Protocol (newline-delimited JSON)
──────────────────────────────────
Request:
    {"method": "<name>", "params": {...}}   # params optional

Responses always include:
    {"ok": true, "result": ...}
    {"ok": false, "error": "<msg>"}

Supported methods
──────────────────
  ping                → {"pong": true}
  get_state           → full latest telemetry snapshot
  get_config          → full CONFIG dict (serialised)
  start_recording     → params: {vars: [str, ...] or omit for all}
                        Starts buffering every sim tick into a ring buffer.
  stop_recording      → returns {n_samples: int, samples: [...]}
                        Stops buffering and flushes the ring buffer.
  drive_command       → params: {fwd: float, yaw: float, ticks: int}
  set_lean            → params: {lean_rad: float}
  set_target_position → params: {position: float}
  set_param           → params: {section: str, key: str, value: any}
  set_drive_mode      → params: {mode: "2wd" | "4wd"}
  reset_sim           → queues a physics reset on the next tick
  run_experiment      → compound: reset+params+record+drive → downsampled results
"""

from __future__ import annotations

import json
import logging
import socket
import time
import threading
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

_PORT = 9871
_HOST = "127.0.0.1"
_RECORD_MAXLEN = 20_000   # ~40 s at 500 Hz


class SimBridge:
    """
    Instantiate once, call start(), then call push_state() each sim tick
    and call pop_commands() each sim tick to drain injected commands.
    """

    def __init__(self, config, port: int = _PORT):
        self._config = config
        self._port = port
        self._lock = threading.Lock()

        # Latest telemetry snapshot (set by push_state every tick)
        self._state: dict[str, Any] = {}

        # Pending command queue  (consumed by sim loop via pop_commands)
        self._cmds: list[dict] = []

        # Ring-buffer recording (filled by push_state when active)
        self._recording: bool = False
        self._record_vars: list[str] | None = None   # None = all
        self._record_buf: deque = deque(maxlen=_RECORD_MAXLEN)

        self._server_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------ #
    # API called by the simulation loop
    # ------------------------------------------------------------------ #

    def start(self):
        """Launch the background TCP server thread."""
        self._running = True
        self._server_thread = threading.Thread(
            target=self._serve, daemon=True, name="SimBridge"
        )
        self._server_thread.start()
        log.info("SimBridge listening on %s:%d", _HOST, self._port)
        print(f"[SimBridge] TCP bridge active on {_HOST}:{self._port}")

    def stop(self):
        self._running = False

    def push_state(self, state_dict: dict):
        """Called every sim tick — updates the shared state snapshot and
        appends to the recording ring buffer when recording is active."""
        with self._lock:
            self._state = state_dict
            if self._recording:
                if self._record_vars:
                    sample = {k: state_dict[k]
                               for k in self._record_vars
                               if k in state_dict}
                else:
                    sample = dict(state_dict)
                self._record_buf.append(sample)

    def pop_commands(self) -> list[dict]:
        """Called every sim tick — returns and clears any injected commands."""
        with self._lock:
            cmds, self._cmds = self._cmds, []
        return cmds

    # ------------------------------------------------------------------ #
    # Internal TCP server
    # ------------------------------------------------------------------ #

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((_HOST, self._port))
        srv.listen(8)
        srv.settimeout(1.0)
        while self._running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True
            ).start()
        srv.close()

    def _handle(self, conn: socket.socket):
        try:
            data = b""
            conn.settimeout(5.0)   # longer for stop_recording (big payload)
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            line = data.split(b"\n")[0]
            req = json.loads(line.decode())
            resp = self._dispatch(req)
            payload = (json.dumps(resp) + "\n").encode()
            # Send in chunks so large recording payloads don't block
            view = memoryview(payload)
            sent = 0
            while sent < len(payload):
                n = conn.send(view[sent:])
                sent += n
        except Exception as exc:
            try:
                conn.sendall(
                    (json.dumps({"ok": False, "error": str(exc)}) + "\n").encode()
                )
            except Exception:
                pass
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Method dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(self, req: dict) -> dict:
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "ping":
            return {"ok": True, "result": {"pong": True}}

        elif method == "get_state":
            with self._lock:
                return {"ok": True, "result": dict(self._state)}

        elif method == "get_config":
            return {"ok": True, "result": self._config_to_dict()}

        elif method == "drive_command":
            vel = float(params.get("vel", 0.0))
            yaw = float(params.get("yaw", 0.0))
            ticks = int(params.get("ticks", 500))   # ~1 s at 500 Hz
            with self._lock:
                self._cmds.append({"type": "drive", "vel": vel, "yaw": yaw,
                                   "ticks": ticks})
            return {"ok": True, "result": {"queued": "drive_command"}}

        elif method == "set_lean":
            lean = float(params.get("lean_rad", 0.0))
            with self._lock:
                self._cmds.append({"type": "lean", "lean_rad": lean})
            return {"ok": True, "result": {"queued": "set_lean"}}

        elif method == "set_target_position":
            pos = float(params.get("position", 0.0))
            with self._lock:
                self._cmds.append({"type": "target_position", "position": pos})
            return {"ok": True, "result": {"queued": "set_target_position"}}

        elif method == "set_param":
            section = params.get("section", "")
            key = params.get("key", "")
            value = params.get("value")
            try:
                sec = getattr(self._config, section)
                old = getattr(sec, key)
                # Coerce to the same type as the existing value
                if isinstance(old, list):
                    value = list(value)
                elif isinstance(old, float):
                    value = float(value)
                elif isinstance(old, int):
                    value = int(value)
                setattr(sec, key, value)
                return {"ok": True, "result": {
                    "section": section, "key": key,
                    "old": old, "new": value
                }}
            except AttributeError as e:
                return {"ok": False, "error": str(e)}

        elif method == "start_recording":
            vars_ = params.get("vars")   # list or None
            with self._lock:
                self._record_buf.clear()
                self._record_vars = list(vars_) if vars_ else None
                self._recording = True
            return {"ok": True, "result": {
                "recording": True,
                "vars": self._record_vars or "all",
                "buffer_maxlen": _RECORD_MAXLEN,
            }}

        elif method == "stop_recording":
            with self._lock:
                self._recording = False
                samples = list(self._record_buf)
                self._record_buf.clear()
            return {"ok": True, "result": {
                "n_samples": len(samples),
                "samples": samples,
            }}

        elif method == "set_drive_mode":
            mode = params.get("mode", "").lower()
            if mode not in ("2wd", "4wd"):
                return {"ok": False, "error": "mode must be '2wd' or '4wd'"}
            with self._lock:
                self._cmds.append({"type": "set_drive_mode", "mode": mode})
            return {"ok": True, "result": {"queued": "set_drive_mode", "mode": mode}}

        elif method == "reset_sim":
            with self._lock:
                self._cmds.append({"type": "reset"})
            return {"ok": True, "result": {"queued": "reset_sim"}}

        elif method == "run_experiment":
            return self._run_experiment(params)

        else:
            return {"ok": False, "error": f"Unknown method: {method!r}"}

    # ------------------------------------------------------------------ #
    # Compound experiment
    # ------------------------------------------------------------------ #

    def _run_experiment(self, params: dict) -> dict:
        """Reset → set params → record while driving → return downsampled results.

        Runs entirely in the TCP handler thread; sleeps while the sim loop
        processes commands and fills the recording buffer.
        """
        do_reset = params.get("reset", True)
        settle_s = float(params.get("settle_s", 0.5))
        param_changes = params.get("params") or []
        command = params.get("command") or {}
        record_vars = params.get("record_vars")
        extra_s = float(params.get("extra_s", 0.5))
        decimation = max(1, int(params.get("decimation", 10)))

        # 1. Reset
        if do_reset:
            with self._lock:
                self._cmds.append({"type": "reset"})
            time.sleep(settle_s)

        # 2. Apply parameter changes
        applied = []
        for pc in param_changes:
            section = pc.get("section", "")
            key = pc.get("key", "")
            value = pc.get("value")
            try:
                sec = getattr(self._config, section)
                old = getattr(sec, key)
                if isinstance(old, list):
                    value = list(value)
                elif isinstance(old, float):
                    value = float(value)
                elif isinstance(old, int):
                    value = int(value)
                setattr(sec, key, value)
                applied.append({"section": section, "key": key,
                                "old": old, "new": value})
            except AttributeError as exc:
                return {"ok": False,
                        "error": f"set_param failed for {section}.{key}: {exc}"}

        # 3. Start recording
        with self._lock:
            self._record_buf.clear()
            self._record_vars = list(record_vars) if record_vars else None
            self._recording = True

        # 4. Queue drive command
        vel = float(command.get("vel", 0.0))
        yaw = float(command.get("yaw", 0.0))
        ticks = int(command.get("ticks", 500))
        with self._lock:
            self._cmds.append({"type": "drive", "vel": vel, "yaw": yaw,
                                "ticks": ticks})

        # 5. Wait for command duration + extra settling time
        wait_s = ticks * self._config.sim.timestep + extra_s
        time.sleep(wait_s)

        # 6. Stop recording
        with self._lock:
            self._recording = False
            raw_samples = list(self._record_buf)
            self._record_buf.clear()

        # 7. Downsample and compute stats
        downsampled, stats = self._downsample_and_stats(raw_samples, decimation)

        return {"ok": True, "result": {
            "n_samples_raw": len(raw_samples),
            "n_samples_returned": len(downsampled),
            "decimation": decimation,
            "duration_s": round(wait_s, 3),
            "params_applied": applied,
            "stats": stats,
            "samples": downsampled,
        }}

    @staticmethod
    def _downsample_and_stats(samples: list[dict], decimation: int
                              ) -> tuple[list[dict], dict]:
        """Return (downsampled_samples, per_variable_stats)."""
        if not samples:
            return [], {}

        # Per-variable min / max / mean / final
        stats: dict[str, dict] = {}
        for key in samples[0]:
            if not isinstance(samples[0][key], (int, float)):
                continue
            vals = [s[key] for s in samples
                    if key in s and isinstance(s[key], (int, float))]
            if vals:
                stats[key] = {
                    "min":   round(min(vals), 6),
                    "max":   round(max(vals), 6),
                    "mean":  round(sum(vals) / len(vals), 6),
                    "final": round(vals[-1], 6),
                }

        # Take every Nth sample, round floats to shrink payload
        picked = samples[::decimation] if decimation > 1 else samples
        rounded = []
        for s in picked:
            rounded.append({k: round(v, 6) if isinstance(v, float) else v
                            for k, v in s.items()})
        return rounded, stats

    # ------------------------------------------------------------------ #
    # Config serialisation helper
    # ------------------------------------------------------------------ #

    def _config_to_dict(self) -> dict:
        """Recursively serialise dataclass config to plain dict."""
        import dataclasses
        def _cvt(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: _cvt(getattr(obj, f.name))
                        for f in dataclasses.fields(obj)}
            if isinstance(obj, (list, tuple)):
                return [_cvt(x) for x in obj]
            return obj
        return _cvt(self._config)
