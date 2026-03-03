"""
Linux Joystick Gamepad Reader

Reads from /dev/input/jsN using the Linux joystick API.
No external dependencies — uses only the standard library.

Typical Logitech F710 axis mapping (XInput mode, X/D switch on back):
    Axis 0: Left stick X
    Axis 1: Left stick Y
    Axis 2: Left trigger
    Axis 3: Right stick X
    Axis 4: Right stick Y
    Axis 5: Right trigger
    Axis 6: D-pad X
    Axis 7: D-pad Y

Stick axes: -1.0 = left/up, +1.0 = right/down
"""

import os
import struct
import time


# Linux joystick event types
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80


class Gamepad:
    """
    Non-blocking Linux joystick reader.

    Usage:
        gp = Gamepad('/dev/input/js0')
        while True:
            gp.poll()
            speed = -gp.axis(4)       # right stick Y (inverted)
            yaw   =  gp.axis(3)       # right stick X
    """

    def __init__(self, device='/dev/input/js0', deadzone=0.08):
        self.device = device
        self.deadzone = deadzone
        self._axes = {}
        self._buttons = {}
        self._prev_buttons = {}
        self._fd = None
        self._connected = False
        self._open()

    def _open(self):
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
            self._connected = True
            print(f"  Gamepad: opened {self.device}")
        except OSError as e:
            self._connected = False
            print(f"  Gamepad: {self.device} not available ({e})")

    @property
    def connected(self):
        return self._connected

    def poll(self):
        """Read all pending joystick events (non-blocking)."""
        if not self._connected:
            return
        # Snapshot previous button state for edge detection
        self._prev_buttons = dict(self._buttons)
        while True:
            try:
                data = os.read(self._fd, 8)
                _timestamp, value, type_, number = struct.unpack('IhBB', data)
                type_ &= ~_JS_EVENT_INIT
                if type_ == _JS_EVENT_AXIS:
                    self._axes[number] = value / 32767.0
                elif type_ == _JS_EVENT_BUTTON:
                    self._buttons[number] = bool(value)
            except OSError:
                break

    def axis(self, index, default=0.0):
        """Get axis value [-1, 1] with deadzone applied."""
        val = self._axes.get(index, default)
        if abs(val) < self.deadzone:
            return 0.0
        # Rescale so the output starts from 0 right after the deadzone
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)

    def button(self, index):
        """Get button state (True/False)."""
        return self._buttons.get(index, False)

    def button_pressed(self, index):
        """True on the poll-cycle when button transitions from released to pressed."""
        now = self._buttons.get(index, False)
        prev = self._prev_buttons.get(index, False)
        return now and not prev

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            self._connected = False

    def __del__(self):
        self.close()
