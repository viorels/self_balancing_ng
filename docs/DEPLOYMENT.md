# Deployment

How to set up and run the tribot simulation on a fresh Linux machine.

## Requirements

- Linux with a display (the MuJoCo viewer opens an OpenGL window)
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for environment and package management

Install uv if it is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you launch from a snap-packaged VS Code, the installer may put `uv` under
`~/snap/code/...` instead of `~/.local/bin`. Force the location with:

```bash
UV_INSTALL_DIR="$HOME/.local/bin" sh -c "$(curl -LsSf https://astral.sh/uv/install.sh)"
```

## Setup

From the repository root:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The environment lives in `.venv/` (gitignored). It must be at that exact path:
`.mcp.json` points to `.venv/bin/python` for the MCP simulation server.

Dependencies (`requirements.txt`): numpy, numpy-stl, scipy, mujoco, osqp, mcp.
scipy and osqp are required by the MPC controller that `robot.py` loads by
default (see `docs/MPC_CONTROLLER.md`).

Verify the install:

```bash
.venv/bin/python -c "import mujoco, scipy, osqp, robot, controllers; print('ok')"
```

Headless checks (no viewer needed):

```bash
.venv/bin/python tools/validate_mpc_plant.py   # model vs MuJoCo accelerations
.venv/bin/python tools/run_headless.py all     # balance, drive, transition, push scenarios
```

## Running

Run the simulation with the interactive MuJoCo viewer:

```bash
.venv/bin/python tribot_sim.py
```

Or activate the environment first with `source .venv/bin/activate` and run
`python tribot_sim.py`. Simulation parameters live in `config.py`.

### Optional: MCP simulation server

`mcp/mcp_sim_server.py` exposes the running simulation as MCP tools (state
snapshots, drive commands, parameter patching, experiments). Claude Code and
VS Code pick it up from `.mcp.json` automatically once `.venv` exists. To run
it by hand:

```bash
.venv/bin/python mcp/mcp_sim_server.py
```

### Optional: PlotJuggler live plots

The sim streams JSON over UDP to `127.0.0.1:9870`. In PlotJuggler choose
Streaming, UDP Server, port 9870, message parser JSON, and load
`plot_juggler_layout.xml`.

## Updating

After pulling changes that touch `requirements.txt`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

To rebuild from scratch, delete `.venv/` and repeat the setup steps.

## Notes

- `old/` contains legacy PyBullet and pygame scripts. They are not part of the
  current setup and their dependencies are intentionally not installed.
- Headless machines can run MuJoCo physics, but `tribot_sim.py` always opens
  the viewer, so it needs a display or a virtual framebuffer such as Xvfb.
