import numpy as np
import pytest

from colors import NAMED_COLORS, HSVRange, bgr_to_hsv, get_limits, parse_color


def hues_matched(hsv_range):
    """Set of hue values (0..179) a range accepts."""
    hues = set()
    for lower, upper in hsv_range.bounds():
        hues.update(range(int(lower[0]), int(upper[0]) + 1))
    return hues


def test_green_matches_hue_window_around_60():
    r = get_limits(NAMED_COLORS["green"])
    assert (r.h_low, r.h_high) == (50, 70)
    assert (r.s_low, r.s_high, r.v_low, r.v_high) == (100, 255, 100, 255)
    assert len(r.bounds()) == 1


def test_red_covers_both_ends_of_hue_circle():
    r = get_limits(NAMED_COLORS["red"])
    assert r.wraps
    hues = hues_matched(r)
    assert {0, 5, 10, 170, 175, 179} <= hues
    assert not hues & {11, 90, 169}


def test_red_bounds_never_exceed_opencv_hue_max():
    for lower, upper in get_limits(NAMED_COLORS["red"]).bounds():
        assert lower[0] <= upper[0] <= 179


def test_hue_near_top_wraps_into_low_hues():
    # A crimson with hue ~175.
    h, s, v = bgr_to_hsv((60, 0, 255))
    assert h >= 170
    hues = hues_matched(get_limits((60, 0, 255)))
    assert h in hues and 0 in hues and 90 not in hues


def test_tolerance_widens_range():
    narrow = hues_matched(get_limits(NAMED_COLORS["blue"], tolerance=5))
    wide = hues_matched(get_limits(NAMED_COLORS["blue"], tolerance=20))
    assert len(narrow) == 11 and len(wide) == 41
    assert narrow < wide


def test_tolerance_90_covers_whole_circle():
    assert hues_matched(get_limits(NAMED_COLORS["blue"], tolerance=90)) == set(range(180))


@pytest.mark.parametrize("bad", [-1, 91])
def test_invalid_tolerance_rejected(bad):
    with pytest.raises(ValueError):
        get_limits(NAMED_COLORS["green"], tolerance=bad)


def test_white_is_not_treated_as_red():
    r = get_limits(NAMED_COLORS["white"])
    assert (r.h_low, r.h_high) == (0, 179)  # hue ignored
    assert r.s_high <= 60  # only low-saturation pixels
    assert r.v_low >= 190  # only bright pixels
    red = get_limits(NAMED_COLORS["red"])
    assert r.s_high < red.s_low  # vivid red can't match white


def test_black_matches_dark_pixels_only():
    r = get_limits(NAMED_COLORS["black"])
    assert r.v_low == 0 and r.v_high <= 60


def test_dark_or_pale_targets_lower_the_floors():
    dark_green = get_limits((0, 100, 0))  # v = 100
    assert dark_green.v_low < 100
    vivid = get_limits(NAMED_COLORS["green"])
    assert dark_green.v_low < vivid.v_low


def test_chromatic_floors_stay_above_grey_noise():
    r = get_limits((40, 90, 40))  # dull, dark green
    assert r.s_low >= 30 and r.v_low >= 30


def test_bounds_are_uint8_arrays():
    for lower, upper in get_limits(NAMED_COLORS["red"]).bounds():
        assert lower.dtype == np.uint8 and upper.dtype == np.uint8
        assert lower.shape == upper.shape == (3,)


def test_manual_wrapping_range_from_sliders():
    r = HSVRange(175, 5, 50, 255, 50, 255)
    assert hues_matched(r) == set(range(175, 180)) | set(range(0, 6))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("green", ("green", (0, 255, 0))),
        ("  RED ", ("red", (0, 0, 255))),
        ("#ff8000", ("#ff8000", (0, 128, 255))),
        ("#FF8000", ("#ff8000", (0, 128, 255))),
        ("0,255,0", ("0,255,0", (0, 255, 0))),
    ],
)
def test_parse_color(text, expected):
    assert parse_color(text) == expected


@pytest.mark.parametrize("bad", ["chartreuse", "#12345", "#gggggg", "1,2", "0,0,256", ""])
def test_parse_color_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_color(bad)
