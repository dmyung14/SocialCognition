import cv2
import numpy as np

from interaction_recon.fusion.block_geometry import export_blocks, lift_blocks
from interaction_recon.io.cache import cache_key
from interaction_recon.io.media import MediaSequence
from interaction_recon.vision.blocks import (
    BlockDetector, PersistentBlockTracker, BlockDetection,
)
from interaction_recon.vision.scene_tracking import refine_scene
from interaction_recon.vision.camera import CameraModel


def scene():
    lab = np.full((240, 320, 3), [110, 133, 141], np.uint8)
    lab[75:230, 15:305] = [150, 137, 153]
    lab[140:164, 90:128] = [150, 132, 140]
    lab[175:199, 210:250] = [195, 137, 148]
    lab[20:50, 80:130] = [180, 128, 128]
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def humans(count, times):
    points = np.full((count, 4, 21, 4), np.nan, np.float32)
    points[..., 3] = 0
    return {
        "timestamps_s": times, "hands": points,
        "handedness": np.full((count, 4), -1, np.int8),
        "handedness_confidence": np.zeros((count, 4)),
    }


def test_same_lightness_low_chroma_and_bright_blocks_not_wall():
    image = scene()
    table = np.zeros(image.shape[:2], np.uint8)
    table[75:230, 15:305] = 255
    detections = BlockDetector().detect(image, table)
    assert len(detections) == 2
    assert all(d.bbox[1] >= 75 for d in detections)


def test_two_touching_offset_blocks_decompose():
    image = scene()
    image[140:164, 90:128] = image[130, 80]
    image[175:199, 210:250] = image[130, 80]
    image[130:180, 100:120] = 230
    image[160:180, 120:160] = 230
    table = np.zeros(image.shape[:2], np.uint8)
    table[75:230, 15:305] = 255
    detections = BlockDetector().detect(image, table)
    assert len(detections) == 2
    assert all(d.bbox[1] >= 75 for d in detections)


def test_occlusion_persists_id_and_never_becomes_observed():
    count = 12
    times = np.arange(count, dtype=float) / 15
    frames = np.repeat(scene()[None], count, axis=0)
    h = humans(count, times)
    polygon = np.array([[80, 130], [140, 130], [140, 175], [80, 175]])
    for i in range(4, 8):
        frames[i, 130:175, 80:140] = (190, 125, 90)
        h["hands"][i, 0, :, :2] = polygon[np.arange(21) % 4]
        h["hands"][i, 0, :, 3] = 0.9
    media = MediaSequence(
        rgb_frames=frames, timestamps_s=times, fps=15, width=320, height=240,
        duration_s=count / 15, sample_times_s=times, source_timestamps_s=times,
        source_frame_indices=np.arange(count),
    )
    result = refine_scene(media, h)
    assert result["block_ids"].shape[1] == 2
    j = int(np.nanargmin(result["block_boxes"][0, :, 0]))
    identity = result["block_ids"][0, j]
    assert np.all(result["block_ids"][:, j] == identity)
    assert not result["block_observed"][4:8, j].any()
    assert result["block_interpolated"][4:8, j].all()
    assert np.all(result["block_method"][4:8, j] == "occluded_hold")
    assert np.max(result["block_confidence"][4:8, j]) < result["block_confidence"][0, j]
    assert result["block_observed"][8:, j].all()


def test_stabilized_table_coordinates_keep_id_during_camera_translation():
    tracker = PersistentBlockTracker()
    corners = np.array([[80, 90], [120, 90], [120, 115], [80, 115]], np.float32)
    exclusion = np.zeros((240, 320), np.uint8)
    ids = []
    for offset in (0, 35, 70):
        transform = np.array([[1, 0, offset], [0, 1, 0], [0, 0, 1]], float)
        moved = corners + [offset, 0]
        detection = BlockDetection(np.array([80 + offset, 90, 40, 25]), moved, 0.9)
        result = tracker.update([detection], transform, exclusion, (240, 320, 3))
        ids.append(result[0].track_id)
    assert len(set(ids)) == 1


def test_repeated_analysis_frames_cannot_confirm_one_source_detection():
    count = 6
    times = np.arange(count, dtype=float) / 15
    media = MediaSequence(
        rgb_frames=np.repeat(scene()[None], count, axis=0),
        timestamps_s=np.zeros(count), fps=15, width=320, height=240,
        duration_s=count / 15, sample_times_s=times,
        source_timestamps_s=np.array([0.0]), source_frame_indices=np.zeros(count, int),
    )
    result = refine_scene(media, humans(count, times))
    assert result["block_ids"].shape[1] == 0


def test_metric_hold_and_separate_export_provenance():
    model = CameraModel(
        np.array([[250., 0, 160], [0, 250., 120], [0, 0, 1]]),
        "pinhole", np.array([160, 120, 120]), 105, 10, 0,
    )
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1, -1, -1])
    transform[2, 3] = 1
    corners = np.array([[145, 110], [175, 110], [175, 130], [145, 130]], float)
    observations = {
        "timestamps_s": np.arange(4) / 15,
        "block_ids": np.zeros((4, 1), int),
        "block_observed": np.array([[1], [0], [0], [1]], bool),
        "block_interpolated": np.array([[0], [1], [1], [0]], bool),
        "block_confidence": np.array([[1], [.25], [.25], [1]]),
        "block_corners": np.tile(corners, (4, 1, 1, 1)),
    }
    camera = {
        **model.arrays(), "T_initial_from_camera": np.tile(transform, (4, 1, 1)),
        "block_lab": np.zeros((4, 1, 3)),
    }
    blocks = lift_blocks(observations, camera, np.eye(4))
    output = {}
    export_blocks(output, blocks, np.arange(4), np.ones(4, bool))
    export_blocks(output, blocks, np.arange(4), np.ones(4, bool), prefix="guider_", source_bit=1)
    assert output["block_poses_interpolated_mask"][1:3].all()
    assert not output["block_poses_observed_mask"][1:3].any()
    np.testing.assert_allclose(output["block_poses"][1], output["block_poses"][0])
    assert np.all(output["block_poses_source"] == 2)
    assert np.all(output["guider_block_poses_source"] == 1)
    assert output["block_height_prior_mask"].all()


def test_block_cache_salt_does_not_change_human_configuration(monkeypatch):
    import interaction_recon.io.cache as cache
    human = {"stage": "m3-human-egocentric-wrist-association-9", "input": "abc"}
    scene_config = {"stage": "m2-scene-local-table-blocks-7", "input": "abc"}
    before_human, before_scene = cache_key(human), cache_key(scene_config)
    monkeypatch.setattr(cache, "BLOCK_CACHE_GENERATION", "next")
    assert cache_key(human) == before_human
    assert cache_key(scene_config) != before_scene
