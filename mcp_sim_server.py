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
  sim_drive_command       → inject fwd/yaw velocity commands for N ticks
  sim_set_lean            → set operator lean bias (rad)
  sim_set_target_position → set position setpoint (m)
  sim_set_param           → live-patch any CONFIG field
  sim_reset               → reset physics to initial conditions
  sim_run_experiment      → apply command, wait N ticks, report final state

Usage (VS Code MCP integration)
────────────────────────────────
  Add to .vscode/mcp.json (see docs or README).
  VS Code will launch this script via: python3 mcp_sim_server.py
"""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

_BRIDGE_HOST = "127.0.0.1"
_BRIDGE_PORT = 9871
_TIMEOUT = 3.0

# ---------------------------------------------------------------------------
# Low-level bridge client
# ---------------------------------------------------------------------------

def _call_bridge(method: str, params: dict | None = None) -> dict:
    """Open a short-lived connection, send one request, return parsed response."""
    req = {"method": method}
    if params:
        req["params"] = params
    raw = (json.dumps(req) + "\n").encode()

    try:
        sock = socket.create_connection((_BRIDGE_HOST, _BRIDGE_PORT),
                                        timeout=_TIMEOUT)
        sock.sendall(raw)
        buf = b""
        sock.settimeout(_TIMEOUT)
        while b"\n" not in buf:
            chunk = sock.recv(65536)
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
            name="sim_reset",
            description=(
                "Reset the robot to its initial upright pose and zero velocities. "
                "Queued on the sim loop and executed at the next tick."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="sim_run_experiment",
            description=(
                "Apply a drive command, wait `wait_ms` milliseconds (real time), "
                "then return the final state. Useful for step-response tests. "
                "`fwd`, `yaw`, `ticks` same as sim_drive_command."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fwd":     {"type": "number", "default": 0.0},
                    "yaw":     {"type": "number", "default": 0.0},
                    "ticks":   {"type": "integer", "default": 500},
                    "wait_ms": {"type": "integer",
                                "description": "Real-time milliseconds to wait before sampling state",
                                "default": 1200},
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:

    if name == "sim_ping":
        resp = _call_bridge("ping")

    elif name == "sim_get_state":
        resp = _call_bridge("get_state")

    elif name == "sim_get_config":
        resp = _call_bridge("get_config")

    elif name == "sim_drive_command":
        resp = _call_bridge("drive_command", {
            "fwd":   arguments.get("fwd", 0.0),
            "yaw":   arguments.get("yaw", 0.0),
            "ticks": arguments.get("ticks", 500),
        })

    elif name == "sim_set_lean":
        resp = _call_bridge("set_lean", {"lean_rad": arguments["lean_rad"]})

    elif name == "sim_set_target_position":
        resp = _call_bridge("set_target_position",
                            {"position": arguments["position"]})

    elif name == "sim_set_param":
        resp = _call_bridge("set_param", {
            "section": arguments["section"],
            "key":     arguments["key"],
            "value":   arguments["value"],
        })

    elif name == "sim_set_drive_mode":
        resp = _call_bridge("set_drive_mode", {"mode": arguments["mode"]})

    elif name == "sim_reset":
        resp = _call_bridge("reset_sim")

    elif name == "sim_run_experiment":
        fwd     = arguments.get("fwd", 0.0)
        yaw     = arguments.get("yaw", 0.0)
        ticks   = arguments.get("ticks", 500)
        wait_ms = arguments.get("wait_ms", 1200)

        # Inject the command
        _call_bridge("drive_command",
                     {"fwd": fwd, "yaw": yaw, "ticks": ticks})

        # Wait for the sim to run it
        time.sleep(wait_ms / 1000.0)

        # Sample final state
        resp = _call_bridge("get_state")
        if resp.get("ok"):
            resp["result"]["experiment"] = {
                "fwd": fwd, "yaw": yaw,
                "ticks": ticks, "wait_ms": wait_ms,
            }

    else:
        resp = {"ok": False, "error": f"Unknown tool: {name}"}

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
