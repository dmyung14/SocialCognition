from types import SimpleNamespace

import numpy as np
import pytest

from interaction_recon.vision.head import (
    HeadDetector,
    head_crop,
    pose_head_fallback,
)


def visible_pose():
    pose = np.full((33, 4), np.nan, np.float32)
    pose[:, 3] = 0
    pose[:11, :3] = [
        [320, 155, -0.03],
        [331, 141, -0.02],
        [335, 140, -0.02],
        [339, 141, -0.01],
        [309, 141, -0.02],
        [305, 140, -0.02],
        [301, 141, -0.01],
        [345, 150, 0],
        [295, 150, 0],
        [330, 173, -0.01],
        [310, 173, -0.01],
    ]
    pose[:11, 3] = 0.9
    return pose


class FakeHead(HeadDetector):
    def __init__(self, result):
        self.fake_result = result
        self.images = []

    def result(self, rgb, timestamp_ms):
        self.images.append(rgb.copy())
        return self.fake_result

    def close(self):
        pass


def empty_result():
    return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])


def test_pose_crop_is_three_ear_distances_and_upscaled():
    rgb = np.zeros((480, 640, 3), np.uint8)
    crop, (x, y, side) = head_crop(rgb, visible_pose())
    assert crop.shape == (256, 256, 3)
    assert side == 150
    assert x <= 295 < 345 < x + side
    assert y <= 140 < 173 < y + side


def test_failed_crop_uses_positive_lower_confidence_pose_fallback():
    detector = FakeHead(empty_result())
    observation = detector.detect_with_pose(
        np.zeros((480, 640, 3), np.uint8), 0, visible_pose()
    )
    assert detector.images[0].shape == (256, 256, 3)
    assert observation.method == "pose_landmarks_fallback"
    assert 0 < observation.confidence <= 0.25
    assert observation.center[3] == pytest.approx(observation.confidence)
    assert np.isfinite(observation.center).all()
    rotation = observation.transformation[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-5)


def test_crop_face_landmarks_map_back_to_source_pixels_and_depth():
    landmarks = [
        SimpleNamespace(x=0.25, y=0.3, z=-0.1),
        SimpleNamespace(x=0.75, y=0.7, z=0.1),
    ]
    detector = FakeHead(SimpleNamespace(
        face_landmarks=[landmarks],
        facial_transformation_matrixes=[np.eye(4)],
    ))
    rgb = np.zeros((480, 640, 3), np.uint8)
    _, (x, y, side) = head_crop(rgb, visible_pose())
    observation = detector.detect_with_pose(rgb, 1, visible_pose())
    assert observation.method == "face_pose_crop"
    assert observation.confidence == 0.5
    np.testing.assert_allclose(
        observation.landmarks[:, :2],
        [[x + side * 0.25, y + side * 0.3],
         [x + side * 0.75, y + side * 0.7]],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        observation.landmarks[:, 2], np.array([-0.1, 0.1]) * side / 640
    )
    np.testing.assert_allclose(observation.center[:2], [x + side / 2, y + side / 2])


def test_crop_clips_to_frame_without_changing_coordinate_mapping():
    rgb = np.zeros((480, 640, 3), np.uint8)
    pose = visible_pose()
    pose[:11, 0] -= 290
    pose[:11, 1] -= 130
    crop, (x, y, side) = head_crop(rgb, pose)
    assert (x, y) == (0, 0)
    assert side == 150
    assert crop.shape == (256, 256, 3)


def test_missing_or_degenerate_pose_does_not_fabricate_a_head():
    detector = FakeHead(empty_result())
    rgb = np.zeros((480, 640, 3), np.uint8)
    missing = visible_pose()
    missing[:, 3] = 0.49
    assert head_crop(rgb, missing) is None
    result = detector.detect_with_pose(rgb, 0, missing)
    assert result.confidence == 0
    assert result.method == "none"
    assert detector.images[0].shape == rgb.shape
    degenerate = visible_pose()
    degenerate[:11, :3] = 0
    assert pose_head_fallback(degenerate, 640).confidence == 0


def test_fallback_works_with_visible_eyes_when_ears_are_occluded():
    pose = visible_pose()
    pose[[7, 8], 3] = 0.1
    result = pose_head_fallback(pose, 640)
    assert result.method == "pose_landmarks_fallback"
    assert result.confidence > 0
