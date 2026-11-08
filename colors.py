"""Colour parsing and HSV range calculation.

Everything here is pure (no camera, no windows), so it is easy to unit-test.

OpenCV conventions used throughout:
  * Colours are BGR, not RGB.
  * Hue runs 0-179 (degrees / 2), saturation and value run 0-255.
"""

from dataclasses import dataclass

import cv2
import numpy as np

HUE_MAX = 179  # OpenCV stores hue as 0..179

# Below this saturation a colour is treated as grey/white/black: its hue is
# meaningless, so we match on saturation and brightness instead.
ACHROMATIC_SAT = 50

# Chromatic ranges never go below these, so grey noise and near-black pixels
# (whose hue is essentially random) are not picked up.
MIN_SAT = 30
MIN_VAL = 30

# Named colours in BGR order.
NAMED_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 128, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "cyan": (255, 255, 0),
    "blue": (255, 0, 0),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "pink": (203, 192, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


@dataclass
class HSVRange:
    """An HSV colour range.

    If ``h_low > h_high`` the hue range wraps around the end of the hue
    circle (e.g. red: 170..179 and 0..10).
    """

    h_low: int
    h_high: int
    s_low: int
    s_high: int
    v_low: int
    v_high: int

    @property
    def wraps(self):
        return self.h_low > self.h_high

    def bounds(self):
        """Return a list of (lower, upper) arrays ready for ``cv2.inRange``.

        A wrapping hue range needs two separate ``inRange`` calls, so this
        returns one or two pairs.
        """
        def pair(h_lo, h_hi):
            return (
                np.array([h_lo, self.s_low, self.v_low], dtype=np.uint8),
                np.array([h_hi, self.s_high, self.v_high], dtype=np.uint8),
            )

        if self.wraps:
            return [pair(self.h_low, HUE_MAX), pair(0, self.h_high)]
        return [pair(self.h_low, self.h_high)]


def bgr_to_hsv(color):
    """Convert one BGR colour to an (h, s, v) tuple of ints."""
    pixel = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def get_limits(color, tolerance=10):
    """Build the HSV range that matches ``color`` (BGR).

    ``tolerance`` is how far either side of the colour's hue to accept.

    * Chromatic colours match hue +/- tolerance (wrapping around red), with
      saturation and brightness floors of 100 for vivid colours. The floors
      drop for paler or darker target colours so those still match.
    * Achromatic colours (white, grey, black) ignore hue and match low
      saturation plus a brightness band around the target.
    """
    if not 0 <= tolerance <= 90:
        raise ValueError("tolerance must be between 0 and 90")

    h, s, v = bgr_to_hsv(color)

    if s < ACHROMATIC_SAT:
        return HSVRange(
            h_low=0,
            h_high=HUE_MAX,
            s_low=0,
            s_high=min(255, s + ACHROMATIC_SAT),
            v_low=max(0, v - 60),
            v_high=min(255, v + 60),
        )

    h_low = (h - tolerance) % (HUE_MAX + 1)
    h_high = (h + tolerance) % (HUE_MAX + 1)
    if tolerance == 90:  # whole hue circle
        h_low, h_high = 0, HUE_MAX

    return HSVRange(
        h_low=h_low,
        h_high=h_high,
        s_low=max(MIN_SAT, min(100, s - 50)),
        s_high=255,
        v_low=max(MIN_VAL, min(100, v - 50)),
        v_high=255,
    )


def parse_color(text):
    """Parse a colour name, ``#RRGGBB`` hex, or ``B,G,R`` triple into BGR.

    Returns a ``(name, (b, g, r))`` tuple.
    """
    raw = text.strip()
    key = raw.lower()

    if key in NAMED_COLORS:
        return key, NAMED_COLORS[key]

    if key.startswith("#") and len(key) == 7:
        try:
            r, g, b = (int(key[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
        else:
            return key, (b, g, r)

    parts = key.split(",")
    if len(parts) == 3:
        try:
            bgr = tuple(int(p) for p in parts)
        except ValueError:
            pass
        else:
            if all(0 <= c <= 255 for c in bgr):
                return key, bgr

    names = ", ".join(NAMED_COLORS)
    raise ValueError(
        f"Unknown colour {raw!r}. Use a name ({names}), "
        "a hex code like #00ff00, or B,G,R like 0,255,0."
    )
