import json

import numpy as np

from interaction_recon.fusion.physics_prerequisites import (
    consolidate_blocks, constrain_hand_depth, nearby_block_slots,
    prepare_physics_observations, smooth_depth_windows,
)
from interaction_recon.vision.camera import CameraModel
from test_scene import empty_observations


def block_tracks():
    arrays = empty_observations(times=np.arange(12) / 15, blocks=4)
    arrays["builder_source_frame_indices"] = np.arange(12)
    for j, interval, x in (
        (0, slice(0, 6), 0.0),
        (1, slice(6, 12), 0.003),
        (2, slice(0, 12), 0.13),
        (3, slice(0, 2), -0.3),
    ):
        arrays["block_poses"][interval, j] = [x, 0, 0.0125, 1, 0, 0, 0]
        arrays["block_dimensions"][interval, j] = [0.10, 0.05, 0.025]
        for name in ("block_poses", "block_dimensions"):
            arrays[f"{name}_confidence"][interval, j] = 0.8
            arrays[f"{name}_observed_mask"][interval, j] = True
            arrays[f"{name}_source"][interval, j] = 2
    return arrays


def test_disjoint_reacquisition_merges_and_short_track_drops():
    arrays = block_tracks()
    report = consolidate_blocks(arrays)
    assert report["block_count"] == 2
    assert len(report["block_merge_log"]) == 1
    assert report["block_dropped_tracks"][0]["ids"] == [3]
    assert arrays["block_ids"].tolist() == [0, 2]
    assert arrays["block_poses_observed_mask"][:, 0].all()
    assert not arrays["block_poses_interpolated_mask"][:, 0].any()


def test_concurrent_overlapping_tracks_merge():
    arrays = block_tracks()
    arrays["block_poses"][:, 1] = [0.04, 0, 0.0125, 1, 0, 0, 0]
    arrays["block_dimensions"][:, 1] = [0.10, 0.05, 0.025]
    for name in ("block_poses", "block_dimensions"):
        arrays[f"{name}_observed_mask"][:, 1] = True
        arrays[f"{name}_confidence"][:, 1] = 0.8
    report = consolidate_blocks(arrays)
    assert report["block_count"] == 2
    assert len(report["block_merge_log"]) == 1
    assert not report["block_merge_log"][0]["identity_verified"]


def test_small_overlap_below_threshold_does_not_merge():
    arrays = empty_observations(times=np.arange(8) / 15, blocks=2)
    arrays["block_dimensions"][:] = [0.10, 0.05, 0.025]
    arrays["block_poses"][:, 0] = [0, 0, 0.0125, 1, 0, 0, 0]
    arrays["block_poses"][:, 1] = [0.08, 0, 0.0125, 1, 0, 0, 0]
    for name in ("block_poses", "block_dimensions"):
        arrays[f"{name}_observed_mask"][:] = True
        arrays[f"{name}_confidence"][:] = 0.8
    report = consolidate_blocks(arrays)
    assert report["block_count"] == 2
    assert not report["block_merge_log"]


def test_initial_off_table_center_is_dropped_with_provenance():
    arrays = block_tracks()
    arrays["block_poses"][:, 2, 1] = arrays["table_extent_xy"][1] / 2 + 0.05
    before = arrays["block_poses"][:, 2].copy()
    report = consolidate_blocks(arrays)
    assert arrays["block_ids"].tolist() == [0]
    record = next(
        item for item in report["block_dropped_tracks"]
        if item["reason"] == "initial_center_outside_table"
    )
    assert record["ids"] == [2]
    np.testing.assert_allclose(record["initial_center_xy_m"], before[0, :2])
    assert 2 not in arrays["block_original_to_consolidated"][:, 0]


def test_repeated_source_frames_do_not_supply_support():
    arrays = block_tracks()
    arrays["builder_source_frame_indices"][:] = 0
    report = consolidate_blocks(arrays)
    assert report["block_count"] == 0


def camera_fixture():
    model = CameraModel(
        np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]]),
        "pinhole", np.array([320, 240, 240]), 90, 10, 0,
    )
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1, -1, -1])
    camera[2, 3] = 0.65
    return model, camera


def test_large_depth_error_is_corrected_along_camera_ray_and_bones_preserved():
    model, camera = camera_fixture()
    points = np.tile([0.1, -0.02, 0.25], (21, 1))
    points[0] += [0, -0.05, 0.02]
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    pixels = np.c_[model.project(local), np.zeros(21), np.ones(21)]
    tip = pixels[8, :2]
    corners = tip + [[-10, -10], [10, -10], [10, 10], [-10, 10]]
    corrected, shift, selected = constrain_hand_depth(
        points, pixels, corners, 0.025, model, camera
    )
    assert selected in (4, 8)
    assert abs(corrected[selected, 2] - 0.025) < 1e-10
    np.testing.assert_allclose(corrected - corrected[0], points - points[0])
    ray = points[selected] - camera[:3, 3]
    assert np.linalg.norm(np.cross(shift, ray)) < 1e-10


def test_held_box_proximity_without_literal_overlap():
    hand = np.zeros((21, 4))
    hand[:, :2] = np.array([[112, 45], [120, 45], [120, 55]])[np.arange(21) % 3]
    hand[:, 3] = 1
    raw = {
        "block_boxes": np.array([[[60., 40, 50, 20]]]),
        "block_ids": np.array([[7]]),
        "block_observed": np.array([[False]]),
        "block_interpolated": np.array([[True]]),
    }
    assert hand[:, 0].min() > 110
    assert nearby_block_slots(raw, 0, hand).tolist() == [0]
    raw["block_interpolated"][:] = False
    assert nearby_block_slots(raw, 0, hand).size == 0
    raw["block_method"] = np.array([["occluded_hold"]])
    assert nearby_block_slots(raw, 0, hand).tolist() == [0]
    hand[:, 0] += 30
    assert nearby_block_slots(raw, 0, hand).size == 0


def test_held_proximity_applies_depth_and_smooth_window_before_ik():
    count = 11
    times = np.arange(count) / 15
    arrays = empty_observations(times=times, blocks=1)
    arrays["metrics_json"] = np.array(json.dumps({"streams": {"builder": {}}}))
    arrays["block_poses"][:] = [0, 0, 0.0125, 1, 0, 0, 0]
    arrays["block_dimensions"][:] = [0.10, 0.05, 0.025]
    for name in ("block_poses", "block_dimensions"):
        arrays[f"{name}_confidence"][:] = 0.5
        arrays[f"{name}_observed_mask"][:] = True
        arrays[f"{name}_source"][:] = 2
        arrays[f"{name}_observed_mask"][5] = False
        arrays[f"{name}_interpolated_mask"][5] = True
    model, camera = camera_fixture()
    arrays["builder_T_world_from_camera_source"] = np.tile(camera, (count, 1, 1))
    name = "builder_right_hand_21"
    points = np.tile([0.0, 0.0, 0.25], (21, 1))
    points[0] += [0, -0.02, 0.02]
    arrays[name][:] = points
    arrays[f"{name}_confidence"][:] = 0.4
    arrays[f"{name}_observed_mask"][:] = True
    arrays[f"{name}_source"][:] = 2
    arrays[f"{name}_method"] = np.full(count, "metric_hand_root", dtype="<U64")
    arrays["builder_forearm_keypoints"][:, 1] = [points[0] - [0, 0.2, 0], points[0]]
    arrays["builder_forearm_keypoints_confidence"][:, 1] = 0.2

    # Hand points are just beyond x=318; the held box ends there.
    pixels = np.c_[
        model.project((points - camera[:3, 3]) @ camera[:3, :3]),
        np.zeros(21), np.ones(21),
    ]
    hands = np.full((count, 1, 21, 4), np.nan)
    hands[..., 3] = 0
    hands[5, 0] = pixels
    raw = {
        "timestamps_s": times,
        "source_timestamps_s": times,
        "source_frame_indices": np.arange(count),
        "duration_s": np.array(count / 15),
        "hands": hands,
        "hand_owner": np.full((count, 1), -1),
        "handedness": np.full((count, 1), -1),
        "hand_owner_confidence": np.ones((count, 1)),
        "block_ids": np.zeros((count, 1), int),
        "block_boxes": np.tile([278., 220, 40, 60], (count, 1, 1)),
        "block_corners": np.tile(
            [[278., 220], [318, 220], [318, 280], [278, 280]], (count, 1, 1, 1)
        ),
        "block_observed": np.ones((count, 1), bool),
        "block_interpolated": np.zeros((count, 1), bool),
    }
    raw["hand_owner"][5] = 1
    raw["handedness"][5] = 1
    raw["block_observed"][5] = False
    raw["block_interpolated"][5] = True
    before = arrays[name].copy()
    prepare_physics_observations(arrays, raw, model.arrays(), {"offset_s": 0.0})
    assert arrays["builder_hand_contact_prior_method"][5, 1] == "held_box_proximity"
    assert arrays["builder_hand_contact_prior_block_ids"][5, 1] == 0
    assert arrays["builder_hand_contact_window_weight"][4, 1] > 0
    assert arrays["builder_hand_contact_window_weight"][6, 1] > 0
    assert arrays["builder_hand_contact_window_weight"][0, 1] == 0
    assert abs(arrays[name][5, 8, 2] - 0.025) < 1e-9
    assert arrays[name][4, 8, 2] < before[4, 8, 2]
    np.testing.assert_allclose(
        arrays[name][5] - arrays[name][5, 0], before[5] - before[5, 0]
    )
    np.testing.assert_allclose(arrays[f"{name}_before_m5_contact_prior"], before)
    assert not arrays["block_poses_observed_mask"][5, 0]
    assert arrays["block_poses_interpolated_mask"][5, 0]
    metrics = json.loads(arrays["metrics_json"].item())
    assert metrics["hand_block_contact_evaluated_source_frames"] == 1


def test_smooth_windows_do_not_bridge_long_occlusions():
    times = np.arange(31) / 10
    shifts = np.zeros((31, 3))
    shifts[5, 2] = -0.2
    shifts[25, 2] = -0.3
    anchors = np.zeros(31, bool)
    anchors[[5, 25]] = True
    correction, weight, nearest = smooth_depth_windows(times, shifts, anchors)
    np.testing.assert_allclose(correction[[5, 25]], shifts[[5, 25]])
    assert weight[15] == 0 and nearest[15] == -1
    assert 0 < weight[4] < 1
    assert 0 < weight[26] < 1
