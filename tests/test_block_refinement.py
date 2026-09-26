import cv2
import numpy as np

from interaction_recon.io.cache import cache_key
from interaction_recon.io.media import MediaSequence
from interaction_recon.stages import HUMAN_KEYS, _read_legacy_humans
from interaction_recon.vision.blocks import (
    BlockDetection, BlockDetector, BlockTracker, gradient_texture, table_texture_matches,
)
from interaction_recon.vision.scene_tracking import refine_scene
from interaction_recon.vision.table import TableDetector, extend_far_edge, extend_table_mask


def _detection(x=70.0):
    return BlockDetection(
        np.array([x, 80, 32, 24], np.float32),
        np.array([[x, 80], [x + 32, 80], [x + 32, 104], [x, 104]], np.float32), 0.9,
    )


def test_persistence_is_causal_and_does_not_report_missing_frames():
    tracker = BlockTracker(max_missing=8)
    first = tracker.update([_detection()], (240, 320, 3))[0]
    second = tracker.update([_detection(71)], (240, 320, 3))[0]
    third = tracker.update([_detection(72)], (240, 320, 3))[0]
    assert first.track_id == second.track_id == third.track_id
    assert not first.observed and not second.observed and third.observed
    assert tracker.update([], (240, 320, 3)) == []
    fifth = tracker.update([_detection(74)], (240, 320, 3))[0]
    assert fifth.observed and fifth.track_id == first.track_id


def test_confirmation_expires_when_hits_leave_the_five_frame_window():
    tracker = BlockTracker(max_missing=8)
    for _ in range(3):
        current = tracker.update([_detection()], (240, 320, 3))[0]
    identity = current.track_id
    assert current.observed
    for _ in range(3):
        assert tracker.update([], (240, 320, 3)) == []
    for expected in (False, False, True):
        current = tracker.update([_detection()], (240, 320, 3))[0]
        assert current.track_id == identity
        assert current.observed == expected


def test_three_nonconsecutive_hits_in_five_are_sufficient():
    tracker, confirmed = BlockTracker(), []
    for present in (True, False, True, False, True):
        result = tracker.update([_detection()] if present else [], (240, 320, 3))
        confirmed.append(bool(result and result[0].observed))
    assert confirmed == [False, False, False, False, True]


def test_far_edge_rows_include_objects_bounded_by_table_color():
    colored = np.zeros((240, 320), bool)
    colored[70:230, 20:300] = True
    colored[76:104, 110:154] = False
    clipped = np.zeros((240, 320), np.uint8)
    clipped[98:230, 20:300] = 255
    repaired = extend_far_edge(clipped, colored, np.ones_like(colored))
    assert repaired[80, 125] == repaired[72, 200] == 255
    assert not repaired[:68].any()


def test_extension_does_not_follow_a_narrow_bridge_to_a_wall():
    colored = np.zeros((240, 320), bool)
    colored[10:65, 20:300] = True
    colored[60:130, 150:168] = True
    colored[125:230, 20:300] = True
    clipped = np.zeros((240, 320), np.uint8)
    clipped[140:230, 20:300] = 255
    repaired = extend_far_edge(clipped, colored, np.ones_like(colored))
    assert repaired[128, 70] == 255
    assert not repaired[:120].any()


def test_far_edge_gray_block_survives_a_deliberately_clipped_hull():
    image = np.full((320, 320, 3), 40, np.uint8)
    image[90:305, 20:300] = (205, 195, 180)
    image[99:122, 130:176] = (194, 194, 192)
    clipped = np.zeros((320, 320), np.uint8)
    clipped[116:305, 20:300] = 255
    center = cv2.cvtColor(np.array([[[205, 195, 180]]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    repaired = extend_table_mask(image, clipped, center)
    assert repaired[105, 150] == 255
    blocks = BlockDetector().detect(image, clipped, table_color_lab=center)
    assert len(blocks) == 1
    np.testing.assert_allclose(blocks[0].bbox, [130, 99, 46, 23], atol=3)


def test_far_edge_white_l_is_included_without_admitting_the_wall():
    image = np.full((320, 320, 3), 32, np.uint8)
    wood = (85, 65, 48)
    image[15:65, 25:295] = wood
    image[60:110, 150:165] = wood
    image[105:305, 20:300] = wood
    image[113:155, 130:145] = 235
    image[140:155, 130:172] = 235
    clipped = np.zeros((320, 320), np.uint8)
    clipped[140:305, 20:300] = 255
    center = cv2.cvtColor(np.array([[wood]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    repaired = extend_table_mask(image, clipped, center)
    assert repaired[118, 136] == 255
    assert not repaired[:95].any()
    blocks = BlockDetector().detect(image, clipped, table_color_lab=center)
    assert len(blocks) == 2
    assert all(b.bbox[1] >= 110 for b in blocks)
    points = np.concatenate([b.corners for b in blocks])
    np.testing.assert_allclose(points.min(axis=0), [130, 113], atol=3)
    np.testing.assert_allclose(points.max(axis=0), [171, 154], atol=3)


def test_wood_interior_texture_is_rejected_but_contrasting_gray_is_not():
    yy, _ = np.mgrid[:160, :200]
    lab = np.empty((160, 200, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = 160 + 7 * np.sin(yy * .7), 133, 143
    interior = np.zeros((160, 200), bool)
    interior[50:110, 65:135] = True
    surrounding = np.zeros_like(interior)
    surrounding[35:125, 50:150] = True
    surrounding &= ~interior
    center = np.array([160, 133, 143], np.float32)
    magnitude, bins = gradient_texture(lab)
    assert table_texture_matches(lab, interior, surrounding, magnitude, bins, center, 8)
    lab[interior] = [165, 128, 128]
    magnitude, bins = gradient_texture(lab)
    assert not table_texture_matches(lab, interior, surrounding, magnitude, bins, center, 8)


def test_far_edge_extension_respects_invalid_pixels():
    image = np.zeros((320, 320, 3), np.uint8)
    cv2.circle(image, (160, 160), 150, (85, 65, 48), -1)
    assert not TableDetector().detect(image).mask[image.max(axis=2) == 0].any()


def _human_observations(count, times):
    hands = np.full((count, 4, 21, 4), np.nan, np.float32)
    hands[..., 3] = 0
    return {
        "timestamps_s": times, "hands": hands,
        "handedness": np.full((count, 4), -1, np.int8),
        "handedness_confidence": np.zeros((count, 4), np.float32),
    }


def test_scene_stage_publishes_only_confirmed_current_detections():
    count = 10
    image = np.full((240, 320, 3), 25, np.uint8)
    image[75:230, 15:305] = (92, 102, 88)
    image[140:164, 90:128] = 235
    frames = np.repeat(image[None], count, axis=0)
    times = np.arange(count, dtype=float) / 15
    humans = _human_observations(count, times)
    polygon = np.array([[80, 130], [140, 130], [140, 175], [80, 175]])
    for i in (4, 5, 6):
        frames[i, 130:175, 80:140] = (190, 125, 90)
        humans["hands"][i, 0, :, :2] = polygon[np.arange(21) % 4]
        humans["hands"][i, 0, :, 3] = .9
    frames[0, 180:200, 220:250] = 235
    sequence = MediaSequence(
        rgb_frames=frames, timestamps_s=times, fps=15, width=320, height=240,
        duration_s=count / 15, sample_times_s=times, source_timestamps_s=times,
        source_frame_indices=np.arange(count),
    )
    arrays = refine_scene(sequence, humans)
    assert arrays["block_ids"].shape == (count, 1)
    assert arrays["block_observed"][:4].all()
    assert not arrays["block_observed"][4:7].any()
    assert arrays["block_interpolated"][4:7].all()
    assert np.all(arrays["block_method"][4:7] == "occluded_hold")
    assert arrays["block_observed"][7:].all()
    assert len(np.unique(arrays["block_ids"])) == 1
    assert arrays["block_confidence"][4:7].max() < arrays["block_confidence"][0, 0]


def test_legacy_cache_migration_copies_only_human_arrays(tmp_path):
    key, path = cache_key({"stage": "legacy"}), tmp_path / "legacy.npz"
    values = {name: np.array([1.0]) for name in HUMAN_KEYS}
    np.savez_compressed(
        path, _cache_key=np.array(key), block_ids=np.array([[999]]),
        table_masks=np.ones((1, 2, 2), np.uint8), **values,
    )
    imported = _read_legacy_humans(path, key)
    assert imported is not None and set(imported) == set(HUMAN_KEYS)
    assert _read_legacy_humans(path, "wrong-key") is None
