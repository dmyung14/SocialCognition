import cv2
import numpy as np
import pytest

from interaction_recon.vision.blocks import BlockDetector, BlockTracker
from interaction_recon.vision.hands import HandObservation
from interaction_recon.vision.table import TableDetector, valid_image_mask
from interaction_recon.vision.tracking import _hand_mask


def _scene(shift=0):
    image = np.full((240, 320, 3), 25, np.uint8)
    cv2.rectangle(image, (15, 75), (305, 230), (92, 102, 88), -1)
    cv2.rectangle(image, (90 + shift, 140), (128 + shift, 164), (235, 235, 235), -1)
    cv2.rectangle(image, (205, 175), (235, 195), (245, 245, 245), -1)
    return image


def test_automatic_table_and_blocks_on_synthetic_image():
    image = _scene()
    table = TableDetector().detect(image)
    assert table.confidence > 0
    assert table.mask[150, 110] == 255 and table.mask[20, 20] == 0
    blocks = BlockDetector().detect(image, table.mask)
    assert len(blocks) == 2
    np.testing.assert_allclose(blocks[0].bbox[:2] + blocks[0].bbox[2:] / 2, [109.5, 152.5], atol=2)


def test_hungarian_tracking_and_hand_exclusion():
    detector, tracker = BlockDetector(), BlockTracker()
    first = _scene()
    table = TableDetector().detect(first)
    ids = [b.track_id for b in tracker.update(detector.detect(first, table.mask), first.shape)]
    moved = _scene(5)
    next_blocks = detector.detect(moved, table.mask)
    next_blocks.reverse()
    tracked = tracker.update(next_blocks, moved.shape)
    tracked.sort(key=lambda b: b.bbox[0])
    assert [b.track_id for b in tracked] == ids
    hand = np.zeros(first.shape[:2], np.uint8)
    hand[130:175, 80:145] = 255
    assert len(detector.detect(first, table.mask, hand)) == 1
    assert tracker.update([], first.shape) == []


def _light_scene(size):
    image = np.full((size, size, 3), 45, np.uint8)
    table = np.zeros((size, size), np.uint8)
    x0, x1, y0, y1 = round(size * .08), round(size * .94), round(size * .35), round(size * .96)
    image[y0:y1, x0:x1], table[y0:y1, x0:x1] = (205, 195, 180), 255
    bx, by, bw, bh = round(size * .54), round(size * .57), round(size * .14), round(size * .065)
    image[by:by + bh, bx:bx + bw] = (194, 194, 192)
    return image, table, (bx, by, bw, bh)


@pytest.mark.parametrize("size", [320, 768])
def test_low_contrast_gray_block_inside_table_hole(size):
    image, mask, (x, y, w, h) = _light_scene(size)
    mask[y - 2:y + h + 2, x - 2:x + w + 2] = 0
    blocks = BlockDetector().detect(image, mask)
    assert len(blocks) == 1
    np.testing.assert_allclose(blocks[0].bbox, [x, y, w, h], atol=3)
    automatic = TableDetector().detect(image)
    assert automatic.mask[y + h // 2, x + w // 2] == 255
    blocks = BlockDetector().detect(image, automatic.mask, table_color_lab=automatic.color_lab)
    assert len(blocks) == 1
    np.testing.assert_allclose(blocks[0].bbox, [x, y, w, h], atol=3)


def test_current_frame_dilated_hand_polygon_excludes_gray_object():
    image, table, (x, y, w, h) = _light_scene(768)
    points = np.zeros((21, 4), np.float32)
    points[:, 3] = .9
    corners = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
    points[:, :2] = corners[np.arange(21) % 4]
    exclusion = _hand_mask([HandObservation(points, 1, .9)], image.shape)
    assert exclusion[y - 3, x] == 255
    assert BlockDetector().detect(image, table, exclusion) == []
    assert len(BlockDetector().detect(image, table, _hand_mask([], image.shape))) == 1


def test_guider_l_structure_wall_leak_and_small_component_rejection():
    size = 704
    image = np.full((size, size, 3), 32, np.uint8)
    wood = (85, 65, 48)
    image[30:240, 80:624] = wood
    image[230:380, 330:355] = wood
    image[370:680, 55:649] = wood
    image[193:204, 400:418] = 240
    image[100:150, 170:225] = 235
    image[470:560, 320:347] = 235
    image[533:560, 320:390] = 235
    image[600:611, 450:468] = 240
    yy, xx = np.mgrid[:size, :size]
    circle = (xx - 352) ** 2 + (yy - 352) ** 2 <= 345 ** 2
    image[~circle] = 0
    table = TableDetector().detect(image)
    assert table.confidence > 0
    assert not table.mask[:300].any() and not table.mask[~circle].any()
    assert table.mask[490, 330] == 255
    blocks = BlockDetector().detect(image, table.mask, table_color_lab=table.color_lab)
    assert len(blocks) == 2
    points = np.concatenate([b.corners for b in blocks])
    np.testing.assert_allclose(points.min(axis=0), [320, 470], atol=3)
    np.testing.assert_allclose(points.max(axis=0), [389, 559], atol=3)
    small = np.full((704, 704, 3), wood, np.uint8)
    small[193:204, 400:418] = 240
    assert BlockDetector().detect(small, np.full((704, 704), 255, np.uint8)) == []


def test_vignette_pixels_remain_invalid():
    image = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(image, (150, 150), 140, (90, 70, 50), -1)
    image[140:150, 140:150] = 20
    valid = valid_image_mask(image)
    assert valid[150, 200] == 255
    assert valid[0, 0] == valid[145, 145] == 0
    table = TableDetector().detect(image)
    assert table.mask[0, 0] == table.mask[145, 145] == 0


def test_no_blocks_on_uniform_light_table():
    image, table, (x, y, w, h) = _light_scene(768)
    image[y:y + h, x:x + w] = (205, 195, 180)
    assert BlockDetector().detect(image, table) == []
