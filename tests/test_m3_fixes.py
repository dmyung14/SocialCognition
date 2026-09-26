import json

import cv2
import numpy as np
import pytest

from interaction_recon.fusion import world
from interaction_recon.fusion.triangulation import transform_points
from interaction_recon.io.media import MediaSequence
from interaction_recon.vision.camera import CameraModel
from interaction_recon.vision.detectors import MODEL_FILES
from interaction_recon.vision.hands import (
    HandObservation, OwnershipClassifier, arm_blob_forearms,
    assign_pose_wrists, egocentric_passes,
)
from interaction_recon.vision.metric_tracking import track_metric_stream


def camera_model():
    return CameraModel(
        np.array([[250., 0, 160], [0, 250., 160], [0, 0, 1]]),
        "pinhole", np.array([160, 160, 160]), 105, 10, 0,
    )


def hand_at(x, y, label=0):
    points = np.zeros((21, 4), np.float32)
    points[:, :2] = np.array([x, y]) + np.column_stack((
        np.arange(21) % 5 * 3, np.arange(21) // 5 * 4,
    ))
    points[:, 3] = .9
    return HandObservation(points, label, .9, np.zeros((21, 3), np.float32))


def pose_2d():
    pose = np.full((33, 4), np.nan)
    pose[:, 3] = 0
    pose[15], pose[16] = [230, 130, 0, .95], [90, 140, 0, .95]
    pose[13], pose[14] = [250, 80, 0, .95], [70, 90, 0, .95]
    return pose


def test_d1_orientation_invariant_reports_and_raises():
    head = np.array([[0, .6, .48], [0, .6, -.48]])
    shoulders = np.array([
        [[-.2, .6, .28], [.2, .6, .28]],
        [[-.2, .6, -.25], [.2, .6, -.25]],
    ])
    torso = np.array([[0, .6, .15], [0, .6, -.7]])
    report = world.seated_orientation_invariant(head, shoulders, torso)
    assert report["status"] == "failed" and report["severity"] == "error"
    assert report["head_below_shoulders_frames"] == report["torso_below_table_frames"] == 1
    with pytest.raises(ValueError, match="invariant FAILED"):
        world.seated_orientation_invariant(head, shoulders, torso, raise_on_failure=True)
    assert world.seated_orientation_invariant(head[:1], shoulders[:1], torso[:1])["status"] == "passed"
    assert world.seated_orientation_invariant(
        head[:1] * np.nan, shoulders[:1] * np.nan, torso[:1] * np.nan
    )["status"] == "not_evaluable"


def test_d1_normal_sign_and_actor_height_are_separate_corrections():
    normal = world.orient_table_normal(np.array([0., .8, .6]), np.array([0., -1, 0]))
    np.testing.assert_allclose(normal, [0, -.8, -.6])
    frame = world.table_frame(normal, -.65 * normal)
    assert transform_points(np.zeros((1, 3)), frame)[0, 2] == pytest.approx(.65)
    pose = np.full((4, 33, 3), np.nan)
    pose[:, [11, 12]] = [[-.2, .1, -.55], [.2, .1, -.55]]
    pose[:, [23, 24]] = [[-.15, .1, -.91], [.15, .1, -.91]]
    pose[:, [0, 7, 8]] = [0, .1, -.35]
    before = pose.copy()
    transform, report = world.seated_actor_prior(pose, .4)
    corrected = transform_points(pose, transform)
    np.testing.assert_allclose(pose, before)
    assert report["raw_invariant"]["status"] == "failed"
    assert report["camera_and_table_changed"] is False
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    assert world._pose_invariant(corrected)["status"] == "passed"
    assert np.median(corrected[:, 0, 2]) == pytest.approx(.48)
    assert np.min(corrected[:, [0, 11, 12, 23, 24], 1]) >= .4


def test_d3_pose_wrist_assignment_ignores_duplicate_mirrored_labels():
    left, right = hand_at(231, 129, 1), hand_at(89, 141, 1)
    hands = np.array([right.keypoints, left.keypoints])
    np.testing.assert_array_equal(assign_pose_wrists(hands, pose_2d(), 320, 320), [1, 0])
    missing = pose_2d()
    missing[15, 3] = 0
    np.testing.assert_array_equal(assign_pose_wrists(hands, missing, 320, 320), [-1, 0])
    np.testing.assert_array_equal(
        assign_pose_wrists(np.array([hand_at(300, 300).keypoints]), pose_2d(), 320, 320), [-1, -1]
    )


def test_d3_pose_association_takes_precedence_over_border_ownership():
    pose = pose_2d()
    pose[15, :2], pose[13, :2] = [304, 230], [280, 190]
    owners, _, _ = OwnershipClassifier().update([hand_at(305, 230)], 320, 320, pose=pose)
    assert owners.tolist() == [0]


def test_d2_padding_pass_inverts_coordinates_depth_and_merges():
    source = hand_at(245, 160)
    source.keypoints[:, 2] = .04
    calls = []

    def infer(image):
        calls.append(image.shape[:2])
        if image.shape[:2] == (200, 300):
            return []
        sx, sy = image.shape[1] / 450, image.shape[0] / 300
        points = source.keypoints.copy()
        points[:, 0] = (points[:, 0] + 75) * sx
        points[:, 1] = (points[:, 1] + 50) * sy
        points[:, 2] *= 300 / 450
        return [HandObservation(points, 1, .9, source.world_keypoints.copy())]

    result = egocentric_passes(np.zeros((200, 300, 3), np.uint8), infer)
    assert len(calls) == 3 and calls[1] == (300, 450) and calls[2][0] < calls[1][0]
    assert len(result) == 1
    np.testing.assert_allclose(result[0].keypoints, source.keypoints, atol=1e-4)
    np.testing.assert_array_equal(result[0].world_keypoints, source.world_keypoints)
    assert result[0].method == "mediapipe_padded"


def arm_image():
    image = np.full((320, 320, 3), 45, np.uint8)
    cv2.fillConvexPoly(
        image, np.array([[319, 319], [319, 270], [225, 195], [200, 221]], np.int32),
        (190, 125, 90),
    )
    return image


def test_d2_border_blob_pca_and_metric_height_prior():
    image = arm_image()
    arms = arm_blob_forearms(image)
    assert len(arms) == 1
    arm = arms[0]
    assert arm.side == 1 and arm.method == "border_skin_blob_pca" and arm.confidence < .2
    elbow, wrist = arm.keypoints[:, :2]
    assert elbow[0] > 310 and wrist[0] < elbow[0] - 50 and wrist[1] < elbow[1]
    assert np.isnan(arm.keypoints[:, 2]).all()
    camera = np.eye(4)
    camera[:3, :3], camera[:3, 3] = np.diag([1, -1, -1]), [0, -.4, .65]
    points = world.lift_forearm_blob(arm.keypoints, camera_model(), camera)
    assert np.isfinite(points).all() and np.all(points[:, 2] > 0)
    assert points[1, 2] == pytest.approx(.08)
    assert np.linalg.norm(points[0] - points[1]) <= .38 + 1e-8
    assert arm_blob_forearms(np.full_like(image, (190, 125, 90))) == []
    interior = np.full_like(image, 45)
    cv2.rectangle(interior, (100, 150), (220, 190), (190, 125, 90), -1)
    assert arm_blob_forearms(interior) == []


def test_d2_blob_connection_can_own_a_nonborder_hand():
    arm = arm_blob_forearms(arm_image())[0]
    owners, confidence, _ = OwnershipClassifier().update(
        [hand_at(*arm.keypoints[1, :2])], 320, 320, forearms=[arm],
    )
    assert owners.tolist() == [1] and 0 < confidence[0] < .5


class EmptyBody:
    def detect(self, rgb, timestamp_ms):
        points = np.full((33, 4), np.nan, np.float32)
        points[:, 3] = 0
        return points

    def close(self):
        pass


class EmptyHands:
    def detect(self, rgb, timestamp_ms):
        return []

    def close(self):
        pass


class EmptyHead:
    def detect(self, rgb, timestamp_ms):
        from interaction_recon.vision.head import HeadObservation
        return HeadObservation(np.array([np.nan, np.nan, np.nan, 0]), np.full((4, 4), np.nan), 0)

    def close(self):
        pass


def test_d2_metric_tracker_persists_fallback_without_fabricating_hands(tmp_path):
    for filename in MODEL_FILES.values():
        (tmp_path / filename).write_bytes(b"injected")
    times = np.array([0.])
    sequence = MediaSequence(
        rgb_frames=arm_image()[None], timestamps_s=times, fps=15, width=320, height=320,
        duration_s=1 / 15, sample_times_s=times, source_timestamps_s=times,
        source_frame_indices=np.array([0]),
    )
    arrays = track_metric_stream(
        sequence, "builder", tmp_path, include_scene=False,
        detector_factory=lambda _: (EmptyBody(), EmptyHands(), EmptyHead()),
    )
    assert arrays["wearer_forearm_method"][0, 1] == "border_skin_blob_pca"
    assert np.isfinite(arrays["wearer_forearms_2d"][0, 1, :, :2]).all()
    assert not arrays["hands_observed"].any() and np.isnan(arrays["hands_world"]).all()
    assert "joint hypotheses" in json.loads(arrays["metadata_json"].item())["forearm_fallback"]


def block_inputs():
    model = camera_model()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.diag([1, -1, -1]), [0, 0, 1]
    corners = []
    for sx in (.10, .44, .10):
        points = np.array([
            [-sx / 2, -.025, 0], [sx / 2, -.025, 0],
            [sx / 2, .025, 0], [-sx / 2, .025, 0],
        ])
        corners.append(model.project(transform_points(points, np.linalg.inv(T))))
    observations = {
        "timestamps_s": np.arange(3) / 15, "block_ids": np.zeros((3, 1), int),
        "block_observed": np.ones((3, 1), bool), "block_confidence": np.ones((3, 1)),
        "block_corners": np.asarray(corners)[:, None],
    }
    cameras = {
        **model.arrays(), "T_initial_from_camera": np.repeat(T[None], 3, axis=0),
        "block_lab": np.zeros((3, 1, 3)),
    }
    return observations, cameras


def test_d4_metric_footprint_rejection_keeps_raw_evidence():
    observations, cameras = block_inputs()
    blocks = world._blocks(observations, cameras, np.eye(4))
    np.testing.assert_array_equal(blocks["dimension_clamped"][:, 0], [False, True, False])
    assert blocks["raw_dimensions"][1, 0, 0] == pytest.approx(.44, abs=1e-5)
    assert np.isfinite(blocks["poses"]).all()
    assert not blocks["dimension_outlier"].any()
    np.testing.assert_allclose(blocks["dimensions"][:, 0, 0], .10, atol=1e-5)
    assert blocks["dimension_variance"][0, 0] > 0
    assert blocks["confidence"][1, 0] < blocks["confidence"][0, 0]


def test_d4_filter_cannot_resurrect_a_rejected_block():
    observations, cameras = block_inputs()
    observations["block_corners"][1, 0] = np.nan
    blocks = world._blocks(observations, cameras, np.eye(4))
    output = {}
    world._add_track(
        output, "block_dimensions", observations["timestamps_s"],
        blocks["dimensions"], blocks["confidence"],
        np.where(blocks["confidence"] > 0, 2, 0).astype(np.uint8),
    )
    assert output["block_dimensions_interpolated_mask"][1, 0]
    world._reject_output(output, "block_dimensions", blocks["dimension_outlier"])
    assert np.isnan(output["block_dimensions"][1, 0]).all()
    assert not output["block_dimensions_interpolated_mask"][1, 0]
    assert not output["block_dimensions_observed_mask"][1, 0]
    assert output["block_dimensions_rejected_mask"][1, 0]
    assert output["block_dimensions_source"][1, 0] == 0
