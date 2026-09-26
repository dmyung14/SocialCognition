from pathlib import Path
from typing import Any, Protocol

import numpy as np

MODEL_FILES = {
    "body": "pose_landmarker_full.task",
    "hands": "hand_landmarker.task",
    "head": "face_landmarker.task",
}


class Detector(Protocol):
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> Any: ...
    def close(self) -> None: ...


def require_models(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    paths = {name: root / filename for name, filename in MODEL_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Missing MediaPipe Tasks model file(s): "
            + ", ".join(missing)
            + ". Place the official .task models there or use --model-dir."
        )
    return paths


def landmarks_array(landmarks: list, width: int, height: int, score: float = 1.0) -> np.ndarray:
    result = np.empty((len(landmarks), 4), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        visibility = getattr(landmark, "visibility", None)
        presence = getattr(landmark, "presence", None)
        confidence = min(
            score,
            visibility if visibility is not None else 1.0,
            presence if presence is not None else 1.0,
        )
        result[index] = (
            landmark.x * width, landmark.y * height, landmark.z, confidence
        )
    return result


class TaskDetector:
    def __init__(self, task: Any):
        import mediapipe as mp

        self.mp = mp
        self.task = task
        self.last_timestamp = -1

    def result(self, rgb: np.ndarray, timestamp_ms: int) -> Any:
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("MediaPipe VIDEO timestamps must be strictly increasing")
        self.last_timestamp = timestamp_ms
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        return self.task.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self.task.close()
