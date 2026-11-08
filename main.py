"""Real-time colour tracker.

Run ``python main.py --help`` for options. While running:

    left-click   pick the colour under the cursor and start tracking it
    n / Tab      make the next tracked colour active
    t            show/hide the HSV tuning sliders (edit the active colour)
    m            show/hide the mask window (shows the active colour's mask)
    r            reset the active colour's range to its default
    d            stop tracking the active colour
    q / Esc      quit
"""

import argparse
import sys
from dataclasses import dataclass, replace

import cv2

from colors import HUE_MAX, NAMED_COLORS, HSVRange, get_limits, parse_color
from detection import (
    build_mask,
    draw_label,
    draw_objects,
    find_objects,
    preprocess,
    sample_color,
)

MAIN_WINDOW = "Color Detection"
MASK_WINDOW = "Mask"
TUNE_WINDOW = "Tune"

# (trackbar name, HSVRange field, max value)
SLIDERS = [
    ("H low", "h_low", HUE_MAX),
    ("H high", "h_high", HUE_MAX),
    ("S low", "s_low", 255),
    ("S high", "s_high", 255),
    ("V low", "v_low", 255),
    ("V high", "v_high", 255),
]
AREA_SLIDER = ("Min area", 20000)

HELP_LINE = "click: pick  n: next  t: tune  m: mask  r: reset  d: drop  q: quit"


@dataclass
class TrackedColor:
    name: str
    bgr: tuple
    hsv_range: HSVRange


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect and box coloured objects in a live webcam feed.",
    )
    parser.add_argument(
        "--color", "-c", action="append", dest="colors", metavar="COLOR",
        help="colour to track: a name, #RRGGBB or B,G,R. Repeat to track "
             "several colours. Default: green.",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="camera index to open (default: 0)",
    )
    parser.add_argument(
        "--tolerance", type=int, default=10,
        help="hue tolerance either side of the colour, 0-90 (default: 10)",
    )
    parser.add_argument(
        "--min-area", type=int, default=500,
        help="ignore blobs smaller than this many pixels (default: 500)",
    )
    parser.add_argument(
        "--max-objects", type=int, default=0,
        help="max boxes per colour, 0 for no limit (default: 0)",
    )
    parser.add_argument(
        "--blur", type=int, default=5,
        help="Gaussian blur kernel size, 0 to disable (default: 5)",
    )
    parser.add_argument(
        "--kernel", type=int, default=5,
        help="morphology kernel size for mask clean-up, 0 to disable "
             "(default: 5)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="open the HSV tuning sliders at start-up",
    )
    parser.add_argument(
        "--show-mask", action="store_true",
        help="open the mask window at start-up",
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="flip the image horizontally, like a mirror",
    )
    parser.add_argument(
        "--list-colors", action="store_true",
        help="print the named colours and exit",
    )

    args = parser.parse_args(argv)

    if not 0 <= args.tolerance <= 90:
        parser.error("--tolerance must be between 0 and 90")
    for name in ("min_area", "max_objects", "blur", "kernel"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must not be negative")

    tracked = []
    for text in args.colors or ["green"]:
        try:
            name, bgr = parse_color(text)
        except ValueError as exc:
            parser.error(str(exc))
        tracked.append(TrackedColor(name, bgr, get_limits(bgr, args.tolerance)))
    args.tracked = tracked
    return args


class App:
    def __init__(self, args):
        self.args = args
        self.colors = list(args.tracked)
        self.active = 0
        self.min_area = args.min_area
        self.max_objects = args.max_objects or None
        self.show_mask = args.show_mask
        self.tuning = False
        self.raw_frame = None
        self.pick_count = 0

    # ---- colour management -------------------------------------------

    @property
    def active_color(self):
        return self.colors[self.active] if self.colors else None

    def add_color(self, name, bgr):
        self.colors.append(
            TrackedColor(name, bgr, get_limits(bgr, self.args.tolerance))
        )
        self.set_active(len(self.colors) - 1)

    def set_active(self, index):
        self.active = index
        self.sync_sliders()

    def next_color(self):
        if self.colors:
            self.set_active((self.active + 1) % len(self.colors))

    def drop_active(self):
        if not self.colors:
            return
        removed = self.colors.pop(self.active)
        print(f"Stopped tracking {removed.name}")
        self.set_active(min(self.active, max(0, len(self.colors) - 1)))

    def reset_active(self):
        color = self.active_color
        if color:
            color.hsv_range = get_limits(color.bgr, self.args.tolerance)
            self.sync_sliders()

    # ---- mouse --------------------------------------------------------

    def on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.raw_frame is None:
            return
        bgr = sample_color(self.raw_frame, x, y)
        self.pick_count += 1
        name = f"pick {self.pick_count}"
        self.add_color(name, bgr)
        print(f"Picked {name}: BGR {bgr} -> {self.active_color.hsv_range}")

    # ---- windows ------------------------------------------------------

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        if not self.show_mask:
            cv2.destroyWindow(MASK_WINDOW)

    def toggle_tuning(self):
        self.tuning = not self.tuning
        if self.tuning:
            cv2.namedWindow(TUNE_WINDOW)
            for label, _field, maximum in SLIDERS:
                cv2.createTrackbar(label, TUNE_WINDOW, 0, maximum, lambda _v: None)
            cv2.createTrackbar(
                AREA_SLIDER[0], TUNE_WINDOW, 0, AREA_SLIDER[1], lambda _v: None
            )
            self.sync_sliders()
        else:
            cv2.destroyWindow(TUNE_WINDOW)

    def sync_sliders(self):
        """Move the sliders to match the active colour."""
        if not self.tuning:
            return
        cv2.setTrackbarPos(AREA_SLIDER[0], TUNE_WINDOW, min(self.min_area, AREA_SLIDER[1]))
        color = self.active_color
        if color is None:
            return
        for label, field, _maximum in SLIDERS:
            cv2.setTrackbarPos(label, TUNE_WINDOW, getattr(color.hsv_range, field))

    def read_sliders(self):
        """Apply slider positions to the active colour."""
        if not self.tuning:
            return
        self.min_area = cv2.getTrackbarPos(AREA_SLIDER[0], TUNE_WINDOW)
        color = self.active_color
        if color is None:
            return
        values = {
            field: cv2.getTrackbarPos(label, TUNE_WINDOW)
            for label, field, _maximum in SLIDERS
        }
        color.hsv_range = replace(color.hsv_range, **values)

    # ---- main loop ----------------------------------------------------

    def process(self, frame):
        """Detect every tracked colour, draw boxes, return the active mask."""
        hsv = preprocess(frame, self.args.blur)
        active_mask = None
        for i, color in enumerate(self.colors):
            mask = build_mask(hsv, color.hsv_range, self.args.kernel)
            objects = find_objects(mask, self.min_area, self.max_objects)
            draw_objects(frame, objects, color.name, color.bgr)
            if i == self.active:
                active_mask = mask
        return active_mask

    def draw_hud(self, frame):
        color = self.active_color
        if color:
            status = f"active: {color.name} ({self.active + 1}/{len(self.colors)})"
        else:
            status = "no colours: click to pick one"
        draw_label(frame, status, 8, 30, (40, 40, 40))
        draw_label(frame, HELP_LINE, 8, frame.shape[0] - 4, (40, 40, 40))

    def handle_key(self, key):
        """Return False when the app should quit."""
        if key in (ord("q"), 27):  # 27 = Esc
            return False
        if key in (ord("n"), 9):  # 9 = Tab
            self.next_color()
        elif key == ord("t"):
            self.toggle_tuning()
        elif key == ord("m"):
            self.toggle_mask()
        elif key == ord("r"):
            self.reset_active()
        elif key == ord("d"):
            self.drop_active()
        return True

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(
                f"Error: could not open camera {self.args.camera}.\n"
                "Check that a webcam is connected, try another --camera index, "
                "and on macOS allow camera access for your terminal in "
                "System Settings > Privacy & Security > Camera.",
                file=sys.stderr,
            )
            return 1

        cv2.namedWindow(MAIN_WINDOW)
        cv2.setMouseCallback(MAIN_WINDOW, self.on_mouse)
        if self.args.tune:
            self.toggle_tuning()

        names = ", ".join(c.name for c in self.colors)
        print(f"Tracking: {names}")
        print(__doc__.split("While running:")[1].rstrip())

        status = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Error: lost the camera feed.", file=sys.stderr)
                    status = 1
                    break
                if self.args.mirror:
                    frame = cv2.flip(frame, 1)

                self.raw_frame = frame.copy()
                self.read_sliders()
                mask = self.process(frame)
                self.draw_hud(frame)

                cv2.imshow(MAIN_WINDOW, frame)
                if self.show_mask and mask is not None:
                    cv2.imshow(MASK_WINDOW, mask)

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return status


def main(argv=None):
    args = parse_args(argv)
    if args.list_colors:
        for name, bgr in NAMED_COLORS.items():
            print(f"{name:8} BGR {bgr}")
        return 0
    return App(args).run()


if __name__ == "__main__":
    sys.exit(main())
