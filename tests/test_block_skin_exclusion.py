import cv2
import numpy as np
import pytest

from interaction_recon.vision.blocks import BlockDetector, skin_mask
from interaction_recon.vision.table import extend_table_mask


def _warm_scene():
    wood = (85, 65, 48)
    image = np.full((320, 320, 3), 32, np.uint8)
    image[15:65, 25:295] = wood
    image[60:110, 150:165] = wood
    image[105:305, 20:300] = wood
    image[113:155, 130:145] = 235
    image[140:155, 130:172] = 235
    image[25:50, 70:105] = 235
    clipped = np.zeros((320, 320), np.uint8)
    clipped[140:305, 20:300] = 255
    center = cv2.cvtColor(np.array([[wood]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return image, clipped, center


def _assert_l_only(blocks):
    assert len(blocks) == 2
    points = np.concatenate([b.corners for b in blocks])
    np.testing.assert_allclose(points.min(axis=0), [130, 113], atol=3)
    np.testing.assert_allclose(points.max(axis=0), [171, 154], atol=3)


@pytest.mark.parametrize("supply_center", [False, True])
def test_skin_colored_wood_remains_table_evidence(supply_center):
    image, clipped, center = _warm_scene()
    assert skin_mask(image)[200, 200] == 255
    assert skin_mask(image)[118, 136] == 0
    repaired = extend_table_mask(image, clipped, center)
    assert repaired[118, 136] == 255
    assert not repaired[:95].any()
    _assert_l_only(BlockDetector().detect(
        image, clipped, table_color_lab=center if supply_center else None,
    ))


@pytest.mark.parametrize("supply_center", [False, True])
def test_table_color_does_not_override_landmark_exclusion(supply_center):
    image, clipped, center = _warm_scene()
    hand = np.zeros_like(clipped)
    hand[110:158, 127:175] = 255
    assert BlockDetector().detect(
        image, clipped, hand, table_color_lab=center if supply_center else None,
    ) == []


def test_distinct_skin_color_is_still_excluded_on_warm_wood():
    image, clipped, center = _warm_scene()
    image[210:242, 210:248] = (190, 125, 90)
    assert skin_mask(image)[220, 220] == 255
    _assert_l_only(BlockDetector().detect(image, clipped, table_color_lab=center))


def test_uniform_skin_colored_wood_is_not_a_block():
    image = np.full((320, 320, 3), (85, 65, 48), np.uint8)
    assert np.all(skin_mask(image) == 255)
    assert BlockDetector().detect(image, np.full((320, 320), 255, np.uint8)) == []
