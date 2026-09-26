from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from interaction_recon.vision.detectors import TaskDetector, landmarks_array


@dataclass
class HeadObservation:
    center: np.ndarray
    transformation: np.ndarray
    confidence: float
    method: str = "none"
    landmarks: np.ndarray | None = None


def _empty_head() -> HeadObservation:
    return HeadObservation(
        np.array([np.nan, np.nan, np.nan, 0], dtype=np.float32),
        np.full((4, 4), np.nan, dtype=np.float32),
        0.0,
    )


def _valid_pose(pose: np.ndarray | None) -> np.ndarray:
    if pose is None or pose.shape[0] < 11:
        return np.zeros(11, dtype=bool)
    return np.isfinite(pose[:11, :3]).all(axis=1) & (pose[:11, 3] >= 0.5)


def head_crop(
    rgb: np.ndarray, pose: np.ndarray | None
) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    valid = _valid_pose(pose)
    if np.count_nonzero(valid[:9]) < 3:
        return None
    points = pose[:9, :2][valid[:9]]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    if valid[7] and valid[8]:
        span = float(np.linalg.norm(pose[7, :2] - pose[8, :2]))
    else:
        span = float(np.max(np.ptp(points, axis=0))) * 1.5
    if span < 2:
        return None
    height, width = rgb.shape[:2]
    side = min(min(height, width), max(16, int(np.ceil(span * 3))))
    x = int(np.clip(round(center[0] - side / 2), 0, width - side))
    y = int(np.clip(round(center[1] - side / 2), 0, height - side))
    crop = cv2.resize(rgb[y:y + side, x:x + side], (256, 256))
    return crop, (x, y, side)


def pose_head_fallback(pose: np.ndarray | None, width: int) -> HeadObservation:
    """Approximate head axes from pose landmarks, not a metric face transform."""
    valid = _valid_pose(pose)
    if np.count_nonzero(valid) < 3:
        return _empty_head()

    points = pose[:11, :3].astype(np.float64).copy()
    points[:, 1] *= -1
    points[:, 2] *= -width
    pair = next(
        ((a, b) for a, b in ((7, 8), (2, 5), (1, 4), (3, 6))
         if valid[a] and valid[b]),
        None,
    )
    if pair is None:
        return _empty_head()
    a, b = pair
    x_axis = points[a] - points[b]
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 2:
        return _empty_head()
    x_axis /= x_norm

    eyes = [i for i in range(1, 7) if valid[i]]
    mouth = [i for i in (9, 10) if valid[i]]
    if eyes and mouth:
        up = points[eyes].mean(axis=0) - points[mouth].mean(axis=0)
    elif valid[0] and (eyes or (valid[7] and valid[8])):
        upper = points[eyes].mean(axis=0) if eyes else points[[7, 8]].mean(axis=0)
        up = upper - points[0]
    else:
        return _empty_head()
    up -= x_axis * np.dot(up, x_axis)
    up_norm = np.linalg.norm(up)
    if up_norm < 1:
        return _empty_head()
    y_axis = up / up_norm
    z_axis = np.cross(x_axis, y_axis)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    confidence = float(0.25 * np.mean(pose[:11, 3][valid]))
    center = np.empty(4, dtype=np.float32)
    center[:3] = np.mean(pose[:11, :3][valid], axis=0)
    center[3] = confidence
    return HeadObservation(center, transform, confidence, "pose_landmarks_fallback")


class HeadDetector(TaskDetector):
    def __init__(self, model: Path):
        if not Path(model).is_file():
            raise RuntimeError(f"Missing MediaPipe Tasks model file: {model}")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        super().__init__(vision.FaceLandmarker.create_from_options(options))

    def result(self, rgb: np.ndarray, timestamp_ms: int):
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        return self.task.detect(image)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> HeadObservation:
        return self.detect_with_pose(rgb, timestamp_ms, None)

    def detect_with_pose(
        self, rgb: np.ndarray, timestamp_ms: int, pose: np.ndarray | None
    ) -> HeadObservation:
        crop = head_crop(rgb, pose)
        if crop is None:
            image = rgb
            x, y, side = 0, 0, None
            method = "face_full_frame"
        else:
            image, (x, y, side) = crop
            method = "face_pose_crop"
        result = self.result(image, timestamp_ms)
        if result.face_landmarks and result.facial_transformation_matrixes:
            points = landmarks_array(
                result.face_landmarks[0], image.shape[1], image.shape[0]
            )
            if side is not None:
                points[:, 0] = x + points[:, 0] * side / image.shape[1]
                points[:, 1] = y + points[:, 1] * side / image.shape[0]
                points[:, 2] *= side / rgb.shape[1]
            transform = np.asarray(
                result.facial_transformation_matrixes[0], dtype=np.float32
            ).reshape(4, 4)
            if np.isfinite(points[:, :3]).all() and np.isfinite(transform).all():
                center = np.r_[points[:, :3].mean(axis=0), 0.5].astype(np.float32)
                return HeadObservation(center, transform, 0.5, method, points)
        return pose_head_fallback(pose, rgb.shape[1])
