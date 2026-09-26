from pathlib import Path

import numpy as np

from interaction_recon.vision.detectors import TaskDetector, landmarks_array

POSE_EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
    (15, 17), (15, 19), (16, 18), (16, 20),
)


class BodyDetector(TaskDetector):
    def __init__(self, model: Path):
        if not model.is_file():
            raise RuntimeError(f"Missing MediaPipe Tasks model file: {model}")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        super().__init__(vision.PoseLandmarker.create_from_options(options))
        self.world_landmarks = np.full((33, 3), np.nan, np.float32)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        result = self.result(rgb, timestamp_ms)
        points = np.full((33, 4), np.nan, dtype=np.float32)
        points[:, 3] = 0
        self.world_landmarks = np.full((33, 3), np.nan, np.float32)
        if result.pose_landmarks:
            points[:] = landmarks_array(
                result.pose_landmarks[0], rgb.shape[1], rgb.shape[0]
            )
        if result.pose_world_landmarks:
            self.world_landmarks[:] = [
                (point.x, point.y, point.z)
                for point in result.pose_world_landmarks[0]
            ]
        return points
