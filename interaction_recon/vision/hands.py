from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from interaction_recon.vision.blocks import skin_mask
from interaction_recon.vision.detectors import TaskDetector, landmarks_array

HAND_EDGES = tuple(
    edge
    for finger in (
        (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
    )
    for edge in zip(finger, finger[1:])
) + ((5, 9), (9, 13), (13, 17))


@dataclass
class HandObservation:
    keypoints: np.ndarray
    handedness: int
    handedness_confidence: float
    world_keypoints: np.ndarray | None = None
    method: str = "mediapipe_full"


@dataclass
class ForearmObservation:
    keypoints: np.ndarray
    side: int
    confidence: float = 0.12
    method: str = "border_skin_blob_pca"


def _box(hand: HandObservation) -> np.ndarray | None:
    points = hand.keypoints
    valid = np.isfinite(points[:, :2]).all(axis=1) & (points[:, 3] > 0)
    if valid.sum() < 3:
        return None
    low, high = points[valid, :2].min(axis=0), points[valid, :2].max(axis=0)
    return np.r_[low, np.maximum(high - low, 1.0)]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.prod(np.maximum(
        0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])
    ))
    return float(intersection / max(np.prod(a[2:]) + np.prod(b[2:]) - intersection, 1e-9))


def merge_hands(
    detections: list[HandObservation], iou_threshold: float = 0.45
) -> list[HandObservation]:
    """NMS keeps one complete landmark set; never averages finger identities."""
    retained = []
    boxes = []
    for hand in sorted(
        detections, key=lambda h: h.handedness_confidence, reverse=True
    ):
        box = _box(hand)
        if box is None or any(_iou(box, other) >= iou_threshold for other in boxes):
            continue
        retained.append(hand)
        boxes.append(box)
    return retained


def egocentric_passes(
    rgb: np.ndarray,
    infer: Callable[[np.ndarray], list[HandObservation]],
    padding_fraction: float = 0.25,
) -> list[HandObservation]:
    """Run independent IMAGE inferences and invert each image transformation.

    WORLD landmarks are metric relative geometry and are not resized. Normalized
    landmark z, unlike WORLD z, must be converted back to source-image width.
    """
    height, width = rgb.shape[:2]
    px = max(1, round(width * padding_fraction))
    py = max(1, round(height * padding_fraction))
    padded = cv2.copyMakeBorder(
        rgb, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    small = cv2.resize(
        padded,
        (max(1, round(padded.shape[1] * 0.65)),
         max(1, round(padded.shape[0] * 0.65))),
        interpolation=cv2.INTER_AREA,
    )
    variants = (
        (rgb, 1.0, 1.0, 0, 0, "mediapipe_full"),
        (padded, 1.0, 1.0, px, py, "mediapipe_padded"),
        (
            small, small.shape[1] / padded.shape[1],
            small.shape[0] / padded.shape[0], px, py,
            "mediapipe_padded_downscaled",
        ),
    )
    detections = []
    for image, sx, sy, ox, oy, method in variants:
        for hand in infer(image):
            points = hand.keypoints.copy()
            points[:, 0] = points[:, 0] / sx - ox
            points[:, 1] = points[:, 1] / sy - oy
            points[:, 2] *= image.shape[1] / (sx * width)
            visible = (
                (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height)
            )
            if visible.sum() < 4:
                continue
            # Outside-image landmarks remain available, but are not detections.
            points[~visible, 3] = 0
            detections.append(HandObservation(
                points, hand.handedness, hand.handedness_confidence,
                None if hand.world_keypoints is None else hand.world_keypoints.copy(),
                method,
            ))
    return merge_hands(detections)


def assign_pose_wrists(
    hands: np.ndarray, pose: np.ndarray, width: int, height: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Return hand slot for anatomical left/right pose wrists (15/16).

    Labels and selfie mirroring are deliberately ignored. The assignment is
    one-to-one and gated, including when only one pose wrist is visible.
    """
    result = np.full(2, -1, np.int32)
    hands = np.asarray(hands)
    if not len(hands):
        return result
    if eligible is None:
        eligible = np.ones(len(hands), bool)
    cost = np.full((2, len(hands)), 1e6)
    diagonal = float(np.hypot(width, height))
    for side, joint in enumerate((15, 16)):
        if pose[joint, 3] < 0.5 or not np.isfinite(pose[joint, :2]).all():
            continue
        elbow = 13 + side
        gate = diagonal * 0.10
        if pose[elbow, 3] >= 0.5 and np.isfinite(pose[elbow, :2]).all():
            gate = float(np.clip(
                0.65 * np.linalg.norm(pose[joint, :2] - pose[elbow, :2]),
                diagonal * 0.04, diagonal * 0.12,
            ))
        for slot, hand in enumerate(hands):
            if (
                not eligible[slot] or hand[0, 3] <= 0
                or not np.isfinite(hand[0, :2]).all()
            ):
                continue
            distance = np.linalg.norm(hand[0, :2] - pose[joint, :2])
            if distance <= gate:
                cost[side, slot] = distance / max(gate, 1)
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        if cost[row, col] < 1e6:
            result[row] = col
    return result


def arm_blob_forearms(rgb: np.ndarray) -> list[ForearmObservation]:
    """Low-confidence arm hypotheses, not anatomical elbow/wrist observations.

    Reject broad table-colored regions, nonelongated components and components
    without a lower/side border entry. No fingers are synthesized.
    """
    height, width = rgb.shape[:2]
    if min(height, width) < 16:
        return []
    mask = skin_mask(rgb)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    candidates = []
    border_width = max(2, round(min(width, height) * 0.015))
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if not width * height * 0.003 <= area <= width * height * 0.22:
            continue
        yy, xx = np.where(labels == label)
        entry = (
            (yy >= height - border_width)
            | (((xx < border_width) | (xx >= width - border_width))
               & (yy > height * 0.45))
        )
        if entry.sum() < 3:
            continue
        points = np.column_stack((xx, yy)).astype(float)
        center = points.mean(axis=0)
        covariance = np.cov(points - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] / max(eigenvalues[0], 1) < 2.5:
            continue
        axis = eigenvectors[:, -1]
        elbow = np.median(points[entry], axis=0)
        if np.dot(center - elbow, axis) < 0:
            axis = -axis
        projection = (points - elbow) @ axis
        length = float(np.percentile(projection, 97))
        if length < min(width, height) * 0.12:
            continue
        tip = points[projection >= np.percentile(projection, 92)].mean(axis=0)
        component = (labels == label).astype(np.uint8)
        ring = (cv2.dilate(component, np.ones((9, 9), np.uint8)) != 0) & (component == 0)
        if ring.sum() < 16:
            continue
        contrast = np.linalg.norm(
            np.median(lab[component != 0], axis=0) - np.median(lab[ring], axis=0)
        )
        if contrast < 12:
            continue
        side = int(elbow[0] >= width / 2)
        keypoints = np.array([
            [elbow[0], elbow[1], np.nan, 0.12],
            [tip[0], tip[1], np.nan, 0.12],
        ], np.float32)
        candidates.append((length, ForearmObservation(keypoints, side)))
    result = []
    for side in (0, 1):
        choices = [item for item in candidates if item[1].side == side]
        if choices:
            result.append(max(choices, key=lambda item: item[0])[1])
    return result


class HandsDetector(TaskDetector):
    def __init__(self, model: Path):
        if not model.is_file():
            raise RuntimeError(f"Missing MediaPipe Tasks model file: {model}")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
        )
        super().__init__(vision.HandLandmarker.create_from_options(options))

    def _infer(self, rgb: np.ndarray) -> list[HandObservation]:
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.task.detect(image)
        observations = []
        for index, (points, categories) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            if not categories:
                continue
            category = categories[0]
            handedness = {"left": 0, "right": 1}.get(
                category.category_name.casefold(), -1
            )
            world = None
            if index < len(result.hand_world_landmarks):
                world = np.asarray([
                    (point.x, point.y, point.z)
                    for point in result.hand_world_landmarks[index]
                ], dtype=np.float32)
            observations.append(HandObservation(
                landmarks_array(points, rgb.shape[1], rgb.shape[0], category.score),
                handedness, float(category.score), world,
            ))
        return observations

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        return egocentric_passes(rgb, self._infer)


class OwnershipClassifier:
    """Temporal ownership prior: -1 unknown, 0 other person, 1 wearer."""

    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self.next_id = 0

    def update(
        self, observations: list[HandObservation], width: int, height: int,
        *, pose: np.ndarray | None = None,
        forearms: list[ForearmObservation] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for track in self.tracks.values():
            track["missed"] += 1
        wrists = np.asarray([
            obs.keypoints[0, :2] / [width, height] for obs in observations
        ]).reshape(-1, 2)
        assigned = {}
        ids = list(self.tracks)
        if ids and len(wrists):
            previous = np.array([self.tracks[i]["wrist"] for i in ids])
            distance = np.linalg.norm(previous[:, None] - wrists[None], axis=2)
            finite = np.isfinite(distance) & (distance < 0.25)
            rows, cols = linear_sum_assignment(np.where(finite, distance, 1e6))
            for row, col in zip(rows, cols):
                if finite[row, col]:
                    assigned[int(col)] = ids[row]

        other_slots = set()
        if pose is not None and observations:
            matches = assign_pose_wrists(
                np.asarray([h.keypoints for h in observations]), pose, width, height
            )
            other_slots = {int(slot) for slot in matches if slot >= 0}

        owners, confidence, hand_ids = [], [], []
        for index, (obs, wrist) in enumerate(zip(observations, wrists)):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "wrist": wrist, "missed": 0, "owner": -1, "confidence": 0.0
                }
            track = self.tracks[track_id]
            palm_y = float(np.mean(obs.keypoints[[5, 9, 13, 17], 1]) / height)
            border = (
                wrist[1] > 0.80
                or ((wrist[0] < 0.10 or wrist[0] > 0.90) and wrist[1] > 0.45)
            )
            connected = any(
                np.linalg.norm(
                    (obs.keypoints[0, :2] - arm.keypoints[1, :2]) / [width, height]
                ) < 0.16
                for arm in (forearms or [])
            )
            if index in other_slots:
                track["owner"], track["confidence"] = 0, 0.85
            elif border or connected:
                track["owner"], track["confidence"] = 1, 0.70 if border else 0.45
            elif track["owner"] != 1 and wrist[1] < 0.62 and wrist[1] < palm_y - 0.015:
                track["owner"], track["confidence"] = 0, 0.55
            elif track["owner"] != 0 and wrist[1] > palm_y + 0.01:
                # Egocentric reach: fingers point away from the camera, wrist below knuckles.
                track["owner"], track["confidence"] = 1, 0.50
            else:
                track["confidence"] *= 0.97
            track.update(wrist=wrist, missed=0)
            owners.append(track["owner"])
            confidence.append(track["confidence"])
            hand_ids.append(track_id)
        self.tracks = {
            key: value for key, value in self.tracks.items() if value["missed"] <= 8
        }
        return (
            np.asarray(owners, dtype=np.int8),
            np.asarray(confidence, dtype=np.float32),
            np.asarray(hand_ids, dtype=np.int32),
        )
