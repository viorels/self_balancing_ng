"""
PlotJuggler UDP Streamer

Sends JSON-encoded signal dictionaries over UDP for real-time visualisation
in PlotJuggler.

PlotJuggler setup:
  Streaming → UDP Server → port 9870 → MessageParser: JSON

Usage:
    streamer = PlotJugglerStreamer()       # default 127.0.0.1:9870
    streamer.send({"timestamp": t, "pitch": 0.01, "torque": 0.5})
    streamer.close()
"""

import json
import socket


class PlotJugglerStreamer:
    """Lightweight UDP sender for PlotJuggler JSON streaming plugin."""

    def __init__(self, host="127.0.0.1", port=9870):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: dict):
        """
        Send a dictionary of signals as a single JSON UDP packet.

        The dict MUST contain a "timestamp" key (float seconds).
        All other keys become PlotJuggler signal names.
        """
        try:
            msg = json.dumps(data, separators=(",", ":"))
            self.sock.sendto(msg.encode("utf-8"), self.addr)
        except OSError:
            pass  # silently drop if PlotJuggler isn't listening

    def close(self):
        self.sock.close()
