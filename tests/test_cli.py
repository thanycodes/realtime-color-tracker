import pytest

from main import parse_args


def test_defaults_track_green():
    args = parse_args([])
    assert [c.name for c in args.tracked] == ["green"]
    assert args.camera == 0 and args.tolerance == 10 and args.min_area == 500


def test_multiple_colours_and_formats():
    args = parse_args(["-c", "red", "--color", "#0000ff", "-c", "255,0,0"])
    assert [c.bgr for c in args.tracked] == [(0, 0, 255), (255, 0, 0), (255, 0, 0)]


def test_tolerance_applied_to_ranges():
    args = parse_args(["--color", "blue", "--tolerance", "20"])
    r = args.tracked[0].hsv_range
    assert (r.h_low, r.h_high) == (100, 140)


@pytest.mark.parametrize(
    "argv",
    [["--color", "nope"], ["--tolerance", "100"], ["--min-area", "-1"], ["--camera", "x"]],
)
def test_bad_arguments_exit(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
