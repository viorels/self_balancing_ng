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
  get_state           → full latest RobotState + ControlOutput snapshot
  get_config          → full CONFIG dict (serialised)
  drive_command       → params: {fwd: float, yaw: float}
                        Injects a synthetic ControlGoals for N ticks.
  set_lean            → params: {lean_rad: float}
  set_target_position → params: {position: float}
  set_param           → params: {section: str, key: str, value: any}
                        Live-patches CONFIG (e.g. section="lqr", key="q_diag")
  reset_sim           → queues a physics reset on the next tick
  ping                → returns {"pong": true}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any

log = logging.getLogger(__name__)

_PORT = 9871
_HOST = "127.0.0.1"


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
        """Called every sim tick — updates the shared state snapshot."""
        with self._lock:
            self._state = state_dict

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
            conn.settimeout(3.0)
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            line = data.split(b"\n")[0]
            req = json.loads(line.decode())
            resp = self._dispatch(req)
            conn.sendall((json.dumps(resp) + "\n").encode())
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
            fwd = float(params.get("fwd", 0.0))
            yaw = float(params.get("yaw", 0.0))
            ticks = int(params.get("ticks", 500))   # ~1 s at 500 Hz
            with self._lock:
                self._cmds.append({"type": "drive", "fwd": fwd, "yaw": yaw,
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

        else:
            return {"ok": False, "error": f"Unknown method: {method!r}"}

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
