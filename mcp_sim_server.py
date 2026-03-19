#!/usr/bin/env python3
"""
mcp_sim_server.py — MCP server that exposes the running tribot simulation
as callable tools to GitHub Copilot (or any MCP client).

Run this BEFORE or AFTER starting the simulation; it will reconnect
automatically each time a tool is called.

Tools exposed
─────────────
  sim_ping                → check bridge is alive
  sim_get_state           → full robot state snapshot (all variables)
  sim_get_config          → full CONFIG dump
  sim_start_recording     → start buffering every sim tick into a ring buffer
  sim_stop_recording      → stop + return all buffered samples at full sim rate
  sim_drive_command       → inject fwd/yaw commands for N ticks
  sim_set_lean            → set operator lean bias (rad)
  sim_set_target_position → set position setpoint (m)
  sim_set_param           → live-patch any CONFIG field
  sim_set_drive_mode      → switch between 2WD and 4WD
  sim_reset               → reset physics to initial conditions

Usage (VS Code MCP integration)
────────────────────────────────
  Add to .vscode/mcp.json (see docs or README).
  VS Code will launch this script via: python3 mcp_sim_server.py
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

_BRIDGE_HOST = "127.0.0.1"
_BRIDGE_PORT = 9871
_TIMEOUT = 3.0
_TIMEOUT_RECORDING = 30.0   # stop_recording can return a large payload

# ---------------------------------------------------------------------------
# Low-level bridge client
# ---------------------------------------------------------------------------

def _call_bridge(method: str, params: dict | None = None,
                 timeout: float = _TIMEOUT) -> dict:
    """Open a short-lived connection, send one request, return parsed response."""
    req = {"method": method}
    if params:
        req["params"] = params
    raw = (json.dumps(req) + "\n").encode()

    try:
        sock = socket.create_connection((_BRIDGE_HOST, _BRIDGE_PORT),
                                        timeout=timeout)
        sock.sendall(raw)
        buf = b""
        sock.settimeout(timeout)
        while b"\n" not in buf:
            chunk = sock.recv(1 << 20)   # 1 MiB chunks for large payloads
            if not chunk:
                break
            buf += chunk
        sock.close()
        return json.loads(buf.split(b"\n")[0].decode())
    except ConnectionRefusedError:
        return {"ok": False,
                "error": "Simulation not running (bridge not reachable on port 9871). "
                         "Start tribot_sim.py first."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _result_text(resp: dict) -> str:
    if resp.get("ok"):
        return json.dumps(resp["result"], indent=2)
    return f"ERROR: {resp.get('error', 'unknown')}"


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------

app = Server("tribot-sim")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="sim_ping",
            description="Check whether the sim bridge is live.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="sim_get_state",
            description=(
                "Return a full snapshot of the robot state: pitch, pitch_rate, "
                "yaw_rate, forward_velocity, position, wheel velocities, triplet "
                "angles, torques, drive_mode, sim_time, and controller internals."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="sim_get_config",
            description="Return the full CONFIG tree (all sections + values).",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="sim_drive_command",
            description=(
                "Inject a drive command into the simulation. "
                "`fwd` is target position offset (metres, + = forward). "
                "`yaw` is yaw-rate (rad/s, + = left). "
                "`ticks` is how many simulation steps to hold the command "
                "(default 500 ≈ 1 s at 500 Hz)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fwd":   {"type": "number", "description": "Forward position offset (m)"},
                    "yaw":   {"type": "number", "description": "Yaw rate (rad/s)"},
                    "ticks": {"type": "integer", "description": "Hold duration in sim steps (default 500)"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="sim_set_lean",
            description="Set operator lean bias in radians (+ = lean forward).",
            inputSchema={
                "type": "object",
                "properties": {
                    "lean_rad": {"type": "number", "description": "Lean angle (rad)"},
                },
                "required": ["lean_rad"],
            },
        ),
        types.Tool(
            name="sim_set_target_position",
            description="Set the position setpoint (metres from start).",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "number", "description": "Target position (m)"},
                },
                "required": ["position"],
            },
        ),
        types.Tool(
            name="sim_set_param",
            description=(
                "Live-patch a CONFIG parameter. "
                "Examples: section='lqr', key='q_diag', value=[10,1,5,0.5]; "
                "section='motor', key='tau', value=0.03; "
                "section='plant', key='cog_height', value=0.12. "
                "Change takes effect on the NEXT control tick."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "Config section (e.g. 'lqr', 'motor', 'plant', 'pid')"},
                    "key":     {"type": "string", "description": "Field name within the section"},
                    "value":   {"description": "New value (number, list, bool, string)"},
                },
                "required": ["section", "key", "value"],
            },
        ),
        types.Tool(
            name="sim_set_drive_mode",
            description=(
                "Switch between 2WD (two-wheel balance) and 4WD (four-wheel stable) "
                "drive modes. In 2WD the triplets rotate so only one wheel per side "
                "touches the ground and the robot actively balances; in 4WD all four "
                "wheels are on the ground."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["2wd", "4wd"],
                        "description": "Target drive mode",
                    },
                },
                "required": ["mode"],
            },
        ),
        types.Tool(
            name="sim_start_recording",
            description=(
                "Start buffering every simulation tick into a ring buffer "
                "(up to 20 000 samples ≈ 40 s at 500 Hz). "
                "Optionally restrict to a subset of variable names to keep "
                "the payload small. "
                "Typical flow: sim_start_recording → sim_drive_command → "
                "sim_stop_recording."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Variable names to record, e.g. "
                            '["timestamp","pitch","pitch_rate","forward_velocity"]. '
                            "Omit to record all telemetry fields."
                        ),
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="sim_stop_recording",
            description=(
                "Stop buffering and return all collected samples as a JSON list. "
                "Each entry is one simulation tick. "
                "Returns {n_samples, samples}."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="sim_reset",
            description=(
                "Reset the robot to its initial upright pose and zero velocities. "
                "Queued on the sim loop and executed at the next tick."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    # --------------------------------------------------------------------------
    # Build (bridge_method, params, timeout) without doing any I/O, then run
    # the blocking socket call off the event loop via asyncio.to_thread so that
    # VS Code's stdio pipe to this process stays responsive at all times.
    # --------------------------------------------------------------------------
    timeout = _TIMEOUT

    if name == "sim_ping":
        method, params = "ping", None

    elif name == "sim_get_state":
        method, params = "get_state", None

    elif name == "sim_get_config":
        method, params = "get_config", None

    elif name == "sim_drive_command":
        method = "drive_command"
        params = {
            "fwd":   arguments.get("fwd", 0.0),
            "yaw":   arguments.get("yaw", 0.0),
            "ticks": arguments.get("ticks", 500),
        }

    elif name == "sim_set_lean":
        method = "set_lean"
        params = {"lean_rad": arguments["lean_rad"]}

    elif name == "sim_set_target_position":
        method = "set_target_position"
        params = {"position": arguments["position"]}

    elif name == "sim_set_param":
        method = "set_param"
        params = {
            "section": arguments["section"],
            "key":     arguments["key"],
            "value":   arguments["value"],
        }

    elif name == "sim_set_drive_mode":
        method = "set_drive_mode"
        params = {"mode": arguments["mode"]}

    elif name == "sim_start_recording":
        method = "start_recording"
        params = {"vars": arguments["vars"]} if "vars" in arguments else None

    elif name == "sim_stop_recording":
        method, params, timeout = "stop_recording", None, _TIMEOUT_RECORDING

    elif name == "sim_reset":
        method, params = "reset_sim", None

    else:
        resp = {"ok": False, "error": f"Unknown tool: {name}"}
        return [types.TextContent(type="text", text=_result_text(resp))]

    resp = await asyncio.to_thread(_call_bridge, method, params, timeout)
    return [types.TextContent(type="text", text=_result_text(resp))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
