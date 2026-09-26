import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from interaction_recon.fusion.filtering import filter_track, slerp
from interaction_recon.fusion.triangulation import (
    lift_metric, ransac_similarity, transform_points, umeyama,
)
from interaction_recon.fusion.world import (
    REQUIRED_OBSERVATIONS, TRACK_NAMES, reconstruct_world, table_frame,
    validate_observations,
)
from interaction_recon.vision.camera import (
    CameraModel, estimate_intrinsics, match_features, relative_motion,
)


def pinhole():
    return CameraModel(
        np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]]),
        "pinhole", np.array([320, 240, 240]), 105, 10, 0,
    )


def test_umeyama_synthetic_similarity_and_ransac_outliers():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(80, 3))
    rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.6]).as_matrix()
    translation = np.array([0.4, -0.2, 1.1])
    target = 1.7 * source @ rotation.T + translation
    scale, R, t = umeyama(source, target)
    assert scale == pytest.approx(1.7)
    np.testing.assert_allclose(R, rotation, atol=1e-10)
    np.testing.assert_allclose(t, translation, atol=1e-10)
    target[:15] += 5
    fit = ransac_similarity(source, target, threshold=0.01)
    assert fit["inliers"].sum() == 65
    assert fit["rmse_m"] < 1e-8
    assert np.linalg.det(R) == pytest.approx(1)


def test_umeyama_rejects_collinear_geometry():
    points = np.c_[np.arange(5), np.zeros((5, 2))]
    with pytest.raises(ValueError, match="collinear"):
        umeyama(points, points)


def test_camera_rotation_from_synthetic_textured_plane():
    rng = np.random.default_rng(91)
    image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    model = pinhole()
    expected = Rotation.from_euler("y", 3.0, degrees=True).as_matrix()
    H = model.K @ expected @ np.linalg.inv(model.K)
    warped = cv2.warpPerspective(image, H, (640, 480))
    a, b = match_features(image, warped)
    assert len(a) >= 30
    recovered = relative_motion(a, b, model)
    error = Rotation.from_matrix(recovered["R"] @ expected.T).magnitude()
    assert np.rad2deg(error) < 1.0
    assert recovered["confidence"] > 0
    assert recovered["method"] == "homography_rotation"


def test_circle_intrinsics_are_priors_and_roundtrip():
    rgb = np.zeros((1, 400, 400, 3), np.uint8)
    cv2.circle(rgb[0], (198, 203), 184, (100, 120, 130), -1)
    model = estimate_intrinsics(rgb, "guider")
    np.testing.assert_allclose(model.circle, [198, 203, 184], atol=2)
    assert model.model == "equidistant"
    assert model.circle_confidence > 0
    assert not model.arrays()["intrinsics_are_calibration"]
    pixels = np.array([[198, 203], [80, 160], [290, 290]], float)
    np.testing.assert_allclose(model.project(model.rays(pixels)), pixels, atol=1e-8)


def test_metric_lifting_uses_world_meters_not_normalized_z():
    rng = np.random.default_rng(5)
    relative = rng.normal(scale=0.035, size=(21, 3))
    relative -= relative[0]
    actual = relative + [0.1, 0.15, 0.8]
    model = pinhole()
    pixels = np.c_[model.project(actual), np.zeros(21), np.ones(21)]
    lifted, depth, sigma, error = lift_metric(pixels, relative, model)
    np.testing.assert_allclose(lifted, actual, atol=1e-8)
    assert depth == pytest.approx(0.8)
    assert sigma >= 0.16 - 1e-8
    assert error < 1e-7


def test_quaternion_slerp_and_short_gap_filter():
    a = np.array([1.0, 0, 0, 0])
    b = np.array([0.0, 0, 0, 1])
    np.testing.assert_allclose(slerp(a, -b, 0.5), [2 ** -0.5, 0, 0, -2 ** -0.5])
    times = np.array([0.0, 0.1, 0.2])
    values = np.array([a, [np.nan] * 4, b])
    filtered = filter_track(
        times, values, np.array([1.0, 0, 1]),
        np.array([1, 0, 2], np.uint8), quaternion=True,
        min_cutoff=1e10,
    )
    np.testing.assert_allclose(np.linalg.norm(filtered["value"], axis=-1), 1)
    assert filtered["interpolated_mask"].tolist() == [False, True, False]
    assert filtered["observed_mask"].tolist() == [True, False, True]
    assert filtered["source"][1] == 3
    assert filtered["confidence"][1] == 0.5


def test_long_gaps_and_isolated_position_outlier():
    times = np.arange(8) / 10
    values = np.zeros((8, 3))
    values[2] = 100
    values[4:7] = np.nan
    confidence = np.isfinite(values).all(axis=1).astype(float)
    filtered = filter_track(times, values, confidence, confidence.astype(np.uint8))
    assert filtered["rejected_mask"][2]
    assert filtered["interpolated_mask"][2]
    assert np.isnan(filtered["value"][4:7]).all()
    assert not filtered["observed_mask"][2]


def test_table_frame_is_right_handed_z_up_and_metric():
    normal = np.array([0.1, -0.8, -0.6])
    normal /= np.linalg.norm(normal)
    point = np.array([0.3, 0.4, 0.9])
    T = table_frame(normal, point)
    np.testing.assert_allclose(T[:3, :3] @ normal, [0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(transform_points(point[None], T), [[0, 0, 0]], atol=1e-10)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-10)


def empty_stream(count=5):
    times = np.arange(count, dtype=float) / 15
    pose = np.full((count, 33, 4), np.nan)
    hands = np.full((count, 4, 21, 4), np.nan)
    pose[..., 3], hands[..., 3] = 0, 0
    return {
        "timestamps_s": times, "source_timestamps_s": times,
        "duration_s": np.array(count / 15), "image_size": np.array([640, 480]),
        "pose": pose, "hands": hands,
        "pose_world": np.full((count, 33, 3), np.nan),
        "hands_world": np.full((count, 4, 21, 3), np.nan),
        "table_masks": np.zeros((count, 160, 160), np.uint8),
        "table_confidence": np.zeros(count),
        "hand_ids": np.full((count, 4), -1),
        "handedness": np.full((count, 4), -1),
        "hand_owner": np.full((count, 4), -1),
        "hand_owner_confidence": np.zeros((count, 4)),
        "head_transform": np.full((count, 4, 4), np.nan),
        "head_confidence": np.zeros(count),
        "block_ids": np.empty((count, 0), int),
        "block_observed": np.empty((count, 0), bool),
        "block_confidence": np.empty((count, 0)),
        "block_corners": np.empty((count, 0, 4, 2)),
    }


def empty_camera(count=5):
    return {
        **pinhole().arrays(),
        "T_initial_from_camera": np.repeat(np.eye(4)[None], count, axis=0),
        "table_normal_initial": np.full((count, 3), np.nan),
        "table_normal_confidence": np.zeros(count),
        "pose_confidence": np.zeros(count),
        "translation_confidence": np.zeros(count),
        "block_lab": np.empty((count, 0, 3)),
        "metric_scale_prior_json": np.array(json.dumps({
            "depth_m": 1.2, "depth_sigma_m": 1.0, "confidence": 0,
            "method": "unobserved_depth_prior",
        })),
    }


def test_observations_npz_contract_empty_data_and_no_fake_fusion(tmp_path):
    observations = {role: empty_stream() for role in ("guider", "builder")}
    cameras = {role: empty_camera() for role in observations}
    arrays = reconstruct_world(
        observations, cameras,
        {"offset_s": 0.0, "confidence": 0.0, "reliable": False}, 15,
    )
    validate_observations(arrays)
    assert set(REQUIRED_OBSERVATIONS) <= arrays.keys()
    assert arrays["fusion_method"].item() == "role_prior_fallback"
    assert not arrays["verified_fusion"]
    assert arrays["block_poses"].shape == (5, 0, 7)
    assert np.isnan(arrays["guider_left_hand_21"]).all()
    path = tmp_path / "observations.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        validate_observations(dict(archive))
        for name in TRACK_NAMES:
            assert archive[f"{name}_source"].dtype == np.uint8
            assert archive[f"{name}_observed_mask"].dtype == bool
    metrics = json.loads(arrays["metrics_json"].item())
    assert metrics["streams"]["guider"]["lifted_keypoint_reprojection_error_px"] is None
    assert metrics["streams"]["builder"]["camera_pose_coverage"] == 0
