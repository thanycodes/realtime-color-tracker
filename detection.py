"""Mask building, clean-up, object finding and drawing.

Pure image functions: they take numpy arrays and return numpy arrays or
lists, so they can be tested with synthetic images.
"""

import cv2
import numpy as np


def preprocess(frame, blur=5):
    """Blur a BGR frame (to reduce sensor noise) and convert it to HSV.

    ``blur`` is the Gaussian kernel size; 0 or 1 disables blurring. Even
    sizes are bumped to the next odd number, as OpenCV requires.
    """
    if blur and blur > 1:
        k = blur if blur % 2 else blur + 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def build_mask(hsv, hsv_range, kernel_size=5):
    """Return a cleaned binary mask of the pixels inside ``hsv_range``.

    Handles wrapping hue ranges (red) by OR-ing the two ``inRange`` masks,
    then runs a morphological open (removes specks) and close (fills small
    holes). ``kernel_size`` 0 disables the clean-up.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_range.bounds():
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if kernel_size and kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_objects(mask, min_area=500, max_objects=None):
    """Find separate blobs in ``mask``.

    Returns a list of ``(x, y, w, h, area)`` tuples, largest first. Blobs
    smaller than ``min_area`` pixels are ignored.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            objects.append((x, y, w, h, area))
    objects.sort(key=lambda o: o[4], reverse=True)
    if max_objects is not None:
        objects = objects[:max_objects]
    return objects


def contrast_color(bgr):
    """Black or white, whichever reads better on top of ``bgr``."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def box_color(bgr):
    """Colour to draw a box in. Very dark colours get a grey so they show."""
    return (128, 128, 128) if max(bgr) < 60 else tuple(int(c) for c in bgr)


def draw_label(frame, text, x, y, bgr):
    """Draw ``text`` on a filled background just above ``(x, y)``."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 6), bgr, -1)
    cv2.putText(
        frame, text, (x + 4, top + th + 3), font, scale,
        contrast_color(bgr), thickness, cv2.LINE_AA,
    )


def draw_objects(frame, objects, name, bgr, thickness=3):
    """Draw a labelled box for each object. Modifies ``frame`` in place."""
    color = box_color(bgr)
    for x, y, w, h, _area in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        draw_label(frame, name, x, y, color)
    return frame


def sample_color(frame, x, y, radius=2):
    """Average BGR colour of the small square around ``(x, y)``.

    Averaging a patch instead of one pixel makes click-to-pick much less
    sensitive to camera noise.
    """
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = frame[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(int(round(c)) for c in patch.mean(axis=0))
