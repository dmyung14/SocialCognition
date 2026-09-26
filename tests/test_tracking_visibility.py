import numpy as np

from interaction_recon.io.media import MediaSequence
from interaction_recon.render.tracking import _skeleton
from interaction_recon.vision.detectors import MODEL_FILES
from interaction_recon.vision.head import pose_head_fallback
from interaction_recon.vision.tracking import track_stream


class Body:
    def detect(self, rgb, timestamp_ms):
        points = np.full((33, 4), np.nan, np.float32)
        points[:, 3] = 0
        points[0] = [50, 35, -0.02, 0.9]
        points[2] = [57, 28, -0.01, 0.9]
        points[5] = [43, 28, -0.01, 0.9]
        points[7] = [62, 32, 0, 0.9]
        points[8] = [38, 32, 0, 0.9]
        points[25] = [40, 70, 0, 0.49]
        points[27] = [40, 85, 0, 0.5]
        return points

    def close(self):
        pass


class Hands:
    def detect(self, rgb, timestamp_ms):
        return []

    def close(self):
        pass


class Head:
    def detect_with_pose(self, rgb, timestamp_ms, pose):
        return pose_head_fallback(pose, rgb.shape[1])

    def close(self):
        pass


def test_tracking_records_fallback_method_and_visibility_threshold(tmp_path):
    for name in MODEL_FILES.values():
        (tmp_path / name).write_bytes(b"injected")
    times = np.array([0.0], np.float64)
    sequence = MediaSequence(
        rgb_frames=np.full((1, 100, 100, 3), 100, np.uint8),
        timestamps_s=times,
        fps=15,
        width=100,
        height=100,
        duration_s=1 / 15,
        sample_times_s=times,
        source_timestamps_s=times,
        source_frame_indices=np.array([0]),
    )
    arrays = track_stream(
        sequence, "guider", tmp_path,
        detector_factory=lambda paths: (Body(), Hands(), Head()),
    )
    assert arrays["head_method"][0] == "pose_landmarks_fallback"
    assert arrays["head_confidence"][0] > 0
    assert not arrays["pose_observed"][0, 25]
    assert arrays["pose_observed"][0, 27]
    # Preserve raw confidence even when a point is not marked observed.
    assert np.isclose(arrays["pose"][0, 25, 3], 0.49)


def test_skeleton_does_not_draw_low_visibility_leg_or_connect_to_it():
    image = np.zeros((100, 100, 3), np.uint8)
    points = np.array([[20, 20, 0, 0.49], [80, 80, 0, 0.5]], np.float32)
    _skeleton(
        image, points, ((0, 1),), (255, 255, 255), minimum_confidence=0.5
    )
    assert not image[18:23, 18:23].any()
    assert not image[45:55, 45:55].any()
    assert image[80, 80].any()
