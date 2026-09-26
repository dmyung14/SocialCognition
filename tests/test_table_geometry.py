import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from interaction_recon.fusion.table_geometry import (
    NormalCandidate, contact_distance, contact_height_anchor, edge_normal,
    fit_table, fuse_height, normal_candidates, orient_and_gate, robust_normal,
    table_frame,
)
from interaction_recon.fusion.triangulation import ray_plane, transform_points
from interaction_recon.fusion.world import _lift_stream, reconstruct_world
from interaction_recon.vision.camera import CameraModel
from test_milestone3 import empty_camera, empty_stream


def model():
    return CameraModel(
        np.array([[520.0, 0, 400], [0, 520.0, 300], [0, 0, 1]]),
        "pinhole", np.array([400, 300, 300]), 90, 10, 0,
    )


def synthetic_scene(count=12):
    """Known camera, projected textured rectangle, and metric shoulder evidence."""
    camera_model = model()
    pitch = np.deg2rad(43)
    normal = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    height = 0.48
    # Put the camera's optical-axis/table intersection at local table origin.
    origin = np.array([0.0, 0.0, height / np.sin(pitch)])
    frame = table_frame(normal, origin)
    table_to_camera = np.linalg.inv(frame)
    corners = np.array([
        [-0.32, -0.18, 0], [0.32, -0.18, 0],
        [0.32, 0.30, 0], [-0.32, 0.30, 0],
    ])
    pixels = camera_model.project(transform_points(corners, table_to_camera))
    mask = np.zeros((600, 800), np.uint8)
    cv2.fillConvexPoly(mask, np.round(pixels).astype(np.int32), 255)

    rng = np.random.default_rng(181)
    texture = rng.integers(70, 170, (160, 240), np.uint8)
    homography = cv2.getPerspectiveTransform(
        np.array([[0, 0], [239, 0], [239, 159], [0, 159]], np.float32),
        pixels.astype(np.float32),
    )
    image = cv2.warpPerspective(texture, homography, (800, 600))
    image[mask == 0] = 20

    observations = empty_stream(count)
    observations["image_size"] = np.array([800, 600])
    observations["source_frame_indices"] = np.arange(count)
    observations["table_masks"] = np.repeat(mask[None], count, axis=0)
    observations["table_confidence"][:] = 1
    camera = empty_camera(count)
    camera.update(camera_model.arrays())
    camera["pose_confidence"][:] = 0.9

    # WORLD geometry is relative to hips; lift_metric must recover the absolute
    # skeleton from its image projection and metric bone lengths.
    body = np.zeros((33, 3))
    body[:] = [0, 0.28, 0.10]
    body[[11, 12]] = [[-0.19, 0.28, 0.325], [0.19, 0.28, 0.325]]
    body[[23, 24]] = [[-0.15, 0.28, -0.075], [0.15, 0.28, -0.075]]
    body[[7, 8]] = [[-0.07, 0.28, 0.53], [0.07, 0.28, 0.53]]
    body[0] = [0, 0.26, 0.54]
    actual = transform_points(body, table_to_camera)
    relative = actual - actual[[23, 24]].mean(axis=0)
    observations["pose_world"][:] = relative
    observations["pose"][..., :2] = camera_model.project(actual)
    observations["pose"][..., 2] = 0
    observations["pose"][..., 3] = 0.95
    return observations, camera, normal, height, table_to_camera, image


def test_vertical_plane_is_rejected_not_replaced_with_prior():
    assert orient_and_gate(np.array([-0.982, 0.002, -0.190])) is None
    assert orient_and_gate(np.array([1.0, 0, 0])) is None
    assert orient_and_gate(np.array([0, -1.0, 0])) is None
    plausible = np.array([-0.002, -0.648, -0.761])
    accepted = orient_and_gate(plausible)
    assert accepted is not None
    np.testing.assert_allclose(accepted, plausible / np.linalg.norm(plausible))
    np.testing.assert_allclose(orient_and_gate(-plausible), accepted)


def test_textured_rectangle_normal_height_and_block_raycast():
    observations, camera, expected, height, table_to_camera, image = synthetic_scene()
    assert image.std() > 20
    recovered, confidence = edge_normal(
        observations["table_masks"][0], observations["image_size"], model()
    )
    assert recovered is not None and confidence > 0
    angle = np.rad2deg(np.arccos(np.clip(recovered @ expected, -1, 1)))
    assert angle < 3

    lifted = _lift_stream(observations, camera)
    table = fit_table(observations, camera, lifted)
    recovered = table["plane_initial"][:3]
    angle = np.rad2deg(np.arccos(np.clip(recovered @ expected, -1, 1)))
    assert angle < 3
    assert abs(table["height_m"] - height) / height < 0.10
    assert table["height_method"] == "metric_evidence_fused_with_prior"
    assert table["height_sigma_m"] > 0
    assert table["minimum_camera_height_m"] > 0

    blocks = np.array([
        [-0.12, -0.10, 0], [-0.06, -0.10, 0],
        [-0.06, -0.07, 0], [-0.12, -0.07, 0],
    ])
    actual = transform_points(blocks, table_to_camera)
    pixels = model().project(actual)
    recovered_points = ray_plane(
        pixels, model(), np.eye(4), table["plane_initial"]
    )
    # Scale error follows the accepted 10% height tolerance.
    relative = np.linalg.norm(recovered_points - actual, axis=1) / np.linalg.norm(actual, axis=1)
    assert np.max(relative) < 0.10
    local = transform_points(recovered_points, table["transform"])
    assert np.all(np.abs(local[:, :2]) <= table["extent"] / 2)
    assert np.max(np.abs(local[:, 2])) < 1e-8


def test_normals_are_transported_once_with_camera_rotation_and_weighted():
    rng = np.random.default_rng(44)
    normal = np.array([0.0, -0.8, -0.6])
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, :3, :3] = Rotation.from_euler(
        "xyz", rng.uniform(-8, 8, (40, 3)), degrees=True
    ).as_matrix()
    candidates = []
    for i in range(40):
        camera_normal = poses[i, :3, :3].T @ normal
        noise = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.5), 3)).as_matrix()
        candidates.append(NormalCandidate(noise @ camera_normal, i, 0.9, "edges"))
        wrong = poses[i, :3, :3].T @ np.array([0.0, -0.4, -np.sqrt(0.84)])
        for _ in range(3):
            candidates.append(NormalCandidate(wrong, i, 0.03, "weak_outlier"))
        candidates.append(NormalCandidate(np.array([-0.982, 0.002, -0.190]), i, 1, "invalid"))
    result = robust_normal(candidates, poses)
    assert np.rad2deg(np.arccos(np.clip(result["normal"] @ normal, -1, 1))) < 1
    assert not result["inliers"][4::5].any()


def test_existing_initial_gauge_homography_is_not_double_rotated():
    observations, camera, normal, _, _, _ = synthetic_scene(4)
    observations["table_confidence"][:] = 0
    observations["pose"][..., 3] = 0
    camera["T_initial_from_camera"][1:, :3, :3] = Rotation.from_euler(
        "y", [[4], [8], [12]], degrees=True
    ).as_matrix()
    camera["table_normal_initial"][1:] = normal
    camera["table_normal_confidence"][1:] = 0.8
    candidates = normal_candidates(observations, camera)
    homographies = [c for c in candidates if c.method == "table_region_homography"]
    assert len(homographies) == 3
    result = robust_normal(homographies, camera["T_initial_from_camera"])
    np.testing.assert_allclose(result["normal"], normal, atol=1e-8)


def test_repeated_source_frames_do_not_inflate_normal_evidence():
    observations, camera, _, _, _, _ = synthetic_scene(10)
    observations["source_frame_indices"][:] = 0
    candidates = normal_candidates(observations, camera)
    assert candidates
    assert all(candidate.frame == 0 for candidate in candidates)


def test_clipped_and_curved_boundaries_abstain():
    mask = np.zeros((600, 800), np.uint8)
    cv2.rectangle(mask, (0, 150), (650, 599), 255, -1)
    assert edge_normal(mask, np.array([800, 600]), model())[0] is None
    mask[:] = 0
    cv2.ellipse(mask, (400, 350), (300, 150), 0, 0, 360, 255, -1)
    assert edge_normal(mask, np.array([800, 600]), model())[0] is None


def test_height_fusion_does_not_gain_false_certainty_from_repetition():
    single = [(0.43, 0.065, "metric_block_contact_wrists")]
    first = fuse_height(single)
    repeated = fuse_height(single * 100)
    assert repeated["height_m"] == pytest.approx(first["height_m"])
    assert repeated["height_sigma_m"] == pytest.approx(first["height_sigma_m"])
    assert abs(first["height_m"] - 0.43) < 0.02
    assert fuse_height([])["height_method"] == "camera_height_prior"


def test_contact_anchor_uses_same_transform_and_own_fingertip_ray():
    observations, camera, normal, height, table_to_camera, _ = synthetic_scene()
    frame = table_frame(normal, -height * normal)
    camera_pose = frame
    hand = np.zeros((21, 3))
    hand[:] = [0.02, 0.03, 0.06]
    hand[[4, 8, 12, 16, 20]] = [0.02, 0.03, 0.025]
    pixels = model().project(transform_points(hand, np.linalg.inv(camera_pose)))
    keypoints = np.c_[pixels, np.zeros(21), np.ones(21)]
    tip_pixel = pixels[8]
    corners = tip_pixel + np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])
    displaced = hand + [0, 0, 0.05]
    anchored = contact_height_anchor(
        displaced, keypoints, corners, 0.025, model(), camera_pose
    )
    assert anchored is not None
    np.testing.assert_allclose(anchored, hand, atol=1e-8)
    assert contact_distance([anchored], np.array([[0.02, 0.03, 0.0125]])) == pytest.approx(0.0125)
    assert contact_height_anchor(
        hand + [0, 0, 0.4], keypoints, corners, 0.025, model(), camera_pose
    ) is None


def test_both_streams_publish_one_plane_and_honest_missing_metrics():
    observations = {role: empty_stream() for role in ("guider", "builder")}
    cameras = {role: empty_camera() for role in observations}
    output = reconstruct_world(
        observations, cameras, {"offset_s": 0.0, "confidence": 0.0}, 15
    )
    metrics = json.loads(output["metrics_json"].item())
    for role in observations:
        assert output[f"{role}_table_plane_initial"].shape == (4,)
        frame = output[f"{role}_table_from_initial"]
        expected = frame[None] @ cameras[role]["T_initial_from_camera"]
        np.testing.assert_allclose(output[f"{role}_T_world_from_camera_source"], expected)
        report = metrics["streams"][role]["table_plane_consistency"]
        assert report["normal_inlier_count"] == 0
        assert report["normal_method"] == "unobserved_camera_tilt_prior"
        assert report["confidence"] < 0.05
    assert metrics["hand_block_contact_distance_m"] is None
    assert metrics["blocks_within_table_extent_fraction"] is None
    assert not output["verified_fusion"]
