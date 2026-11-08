import numpy as np
import pytest

from colors import NAMED_COLORS, get_limits
from detection import build_mask, find_objects, preprocess, sample_color


def canvas(h=200, w=300, bgr=(40, 40, 40)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def detect(img, color, min_area=100, blur=0, kernel=5):
    hsv = preprocess(img, blur)
    mask = build_mask(hsv, get_limits(color), kernel)
    return find_objects(mask, min_area)


def test_finds_green_square_with_correct_box():
    img = canvas()
    img[50:100, 60:140] = NAMED_COLORS["green"]
    objects = detect(img, NAMED_COLORS["green"])
    assert len(objects) == 1
    x, y, w, h, _area = objects[0]
    assert (x, y, w, h) == (60, 50, 80, 50)


def test_two_blobs_give_two_boxes_largest_first():
    img = canvas()
    img[10:40, 10:40] = NAMED_COLORS["blue"]  # 30x30
    img[100:180, 150:260] = NAMED_COLORS["blue"]  # 80x110
    objects = detect(img, NAMED_COLORS["blue"])
    assert len(objects) == 2
    assert objects[0][4] > objects[1][4]
    assert objects[0][:2] == (150, 100)


def test_min_area_ignores_small_noise():
    img = canvas()
    img[50:100, 50:100] = NAMED_COLORS["green"]
    img[5:9, 280:284] = NAMED_COLORS["green"]  # 4x4 speck far away
    objects = detect(img, NAMED_COLORS["green"], min_area=100, kernel=0)
    assert len(objects) == 1
    assert objects[0][:2] == (50, 50)  # box not stretched by the speck


def test_morphology_removes_single_pixel_specks():
    img = canvas()
    img[::10, ::10] = NAMED_COLORS["green"]  # scattered single pixels
    hsv = preprocess(img, 0)
    assert build_mask(hsv, get_limits(NAMED_COLORS["green"]), 0).any()
    assert not build_mask(hsv, get_limits(NAMED_COLORS["green"]), 5).any()


def test_red_detected_on_both_sides_of_hue_wrap():
    img = canvas()
    img[20:70, 20:70] = (0, 0, 255)  # hue 0
    img[120:170, 200:250] = (50, 0, 255)  # hue ~176
    assert len(detect(img, NAMED_COLORS["red"])) == 2


def test_white_paper_not_detected_as_red():
    img = canvas()
    img[20:120, 20:120] = (250, 250, 250)
    assert detect(img, NAMED_COLORS["red"]) == []
    assert len(detect(img, NAMED_COLORS["white"])) == 1


def test_tracks_several_colours_independently():
    img = canvas()
    img[20:80, 20:80] = NAMED_COLORS["green"]
    img[100:160, 200:260] = NAMED_COLORS["yellow"]
    green = detect(img, NAMED_COLORS["green"])
    yellow = detect(img, NAMED_COLORS["yellow"])
    assert [o[:2] for o in green] == [(20, 20)]
    assert [o[:2] for o in yellow] == [(200, 100)]


def test_max_objects_limits_results():
    img = canvas()
    for i in range(3):
        img[20:60, 20 + i * 90:70 + i * 90] = NAMED_COLORS["green"]
    hsv = preprocess(img, 0)
    mask = build_mask(hsv, get_limits(NAMED_COLORS["green"]))
    assert len(find_objects(mask, 100)) == 3
    assert len(find_objects(mask, 100, max_objects=2)) == 2


@pytest.mark.parametrize("blur", [0, 1, 4, 5])
def test_preprocess_accepts_any_blur_size(blur):
    hsv = preprocess(canvas(), blur)
    assert hsv.shape == (200, 300, 3)


def test_sample_color_averages_patch_and_clips_at_edges():
    img = canvas(bgr=(0, 0, 0))
    img[0:5, 0:5] = (100, 150, 200)
    assert sample_color(img, 2, 2) == (100, 150, 200)
    assert sample_color(img, 0, 0) == (100, 150, 200)  # corner, clipped patch
    assert sample_color(img, 299, 199) == (0, 0, 0)
