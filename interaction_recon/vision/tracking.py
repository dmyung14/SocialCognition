from contextlib import ExitStack
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from interaction_recon.io.media import MediaSequence
from interaction_recon.vision.blocks import BlockDetector, BlockTracker
from interaction_recon.vision.body import BodyDetector
from interaction_recon.vision.detectors import Detector, require_models
from interaction_recon.vision.hands import HandsDetector, OwnershipClassifier
from interaction_recon.vision.head import HeadDetector
from interaction_recon.vision.table import TableDetector

DetectorFactory = Callable[[dict[str, Path]], tuple[Detector, Detector, Detector]]


def create_detectors(paths: dict[str, Path]) -> tuple[Detector, Detector, Detector]:
    detectors = []
    try:
        for cls, name in (
            (BodyDetector, "body"), (HandsDetector, "hands"), (HeadDetector, "head")
        ):
            detectors.append(cls(paths[name]))
    except Exception:
        for detector in detectors:
            detector.close()
        raise
    return tuple(detectors)


def _hand_mask(hands: list, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for hand in hands:
        valid = (
            np.isfinite(hand.keypoints[:, :2]).all(axis=1)
            & (hand.keypoints[:, 3] > 0)
        )
        points = hand.keypoints[valid, :2]
        if len(points) >= 3:
            hull = cv2.convexHull(np.round(points).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
    radius = max(3, round(min(shape[:2]) * 0.015))
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        ),
    )


def track_stream(
    sequence: MediaSequence,
    role: str,
    model_dir: str | Path,
    *,
    detector_factory: DetectorFactory | None = None,
    include_scene: bool = True,
) -> dict[str, np.ndarray]:
    factory = detector_factory or create_detectors
    paths = require_models(model_dir)
    table_detector, block_detector = TableDetector(), BlockDetector()
    block_tracker = BlockTracker(
        max_missing=max(5, round(sequence.fps * 0.5))
    )
    ownership = OwnershipClassifier()
    count = len(sequence.rgb_frames)
    pose = np.full((count, 33, 4), np.nan, np.float32)
    hands = np.full((count, 4, 21, 4), np.nan, np.float32)
    pose[..., 3] = 0
    hands[..., 3] = 0
    handedness = np.full((count, 4), -1, np.int8)
    handedness_confidence = np.zeros((count, 4), np.float32)
    owner = np.full((count, 4), -1, np.int8)
    owner_confidence = np.zeros((count, 4), np.float32)
    hand_ids = np.full((count, 4), -1, np.int32)
    head_center = np.full((count, 4), np.nan, np.float32)
    head_center[:, 3] = 0
    head_matrix = np.full((count, 4, 4), np.nan, np.float32)
    head_confidence = np.zeros(count, np.float32)
    head_method = np.full(count, "none", dtype="<U32")
    table_masks = np.zeros((count, 160, 160), np.uint8)
    table_confidence = np.zeros(count, np.float32)
    table_colors = np.full((count, 3), np.nan, np.float32)
    motion = np.zeros(count, np.float64)
    candidate_count = np.zeros(count, np.int32)
    provisional_count = np.zeros(count, np.int32)
    all_blocks = []
    previous_gray = None
    previous_roi = None

    with ExitStack() as stack:
        body_detector, hands_detector, head_detector = factory(paths)
        for detector in (body_detector, hands_detector, head_detector):
            stack.callback(detector.close)
        last_ms = -1
        for index, rgb in enumerate(sequence.rgb_frames):
            relative = sequence.sample_times_s[index] - sequence.sample_times_s[0]
            timestamp_ms = max(last_ms + 1, int(round(relative * 1000)))
            last_ms = timestamp_ms
            pose[index] = body_detector.detect(rgb, timestamp_ms)
            detected_hands = hands_detector.detect(rgb, timestamp_ms)[:4]
            owners, confidences, ids = ownership.update(
                detected_hands, sequence.width, sequence.height
            )
            for slot, hand in enumerate(detected_hands):
                hands[index, slot] = hand.keypoints
                handedness[index, slot] = hand.handedness
                handedness_confidence[index, slot] = hand.handedness_confidence
                owner[index, slot] = owners[slot]
                owner_confidence[index, slot] = confidences[slot]
                hand_ids[index, slot] = ids[slot]
            if hasattr(head_detector, "detect_with_pose"):
                head = head_detector.detect_with_pose(
                    rgb, timestamp_ms, pose[index]
                )
            else:
                head = head_detector.detect(rgb, timestamp_ms)
            head_center[index] = head.center
            head_matrix[index] = head.transformation
            head_confidence[index] = head.confidence
            head_method[index] = head.method

            if not include_scene:
                all_blocks.append([])
                continue

            exclusion = _hand_mask(detected_hands, rgb.shape)
            table = table_detector.detect(rgb, exclusion)
            detections = block_detector.detect(
                rgb, table.mask, exclusion, table_color_lab=table.color_lab
            )
            tracked = block_tracker.update(detections, rgb.shape)
            confirmed = [block for block in tracked if block.observed]
            all_blocks.append(confirmed)
            candidate_count[index] = len(tracked)
            provisional_count[index] = len(tracked) - len(confirmed)
            table_masks[index] = cv2.resize(
                table.mask, (160, 160), interpolation=cv2.INTER_NEAREST
            )
            table_confidence[index] = table.confidence
            table_colors[index] = table.color_lab

            gray = cv2.cvtColor(
                cv2.resize(rgb, (160, 160)), cv2.COLOR_RGB2GRAY
            ).astype(np.float32)
            roi = (table_masks[index] != 0) | (
                cv2.resize(
                    exclusion, (160, 160), interpolation=cv2.INTER_NEAREST
                ) != 0
            )
            if np.count_nonzero(roi) < 50:
                roi[60:150, 20:140] = True
            if previous_gray is not None:
                difference = np.abs(gray - previous_gray)
                motion[index] = float(
                    np.percentile(difference[roi | previous_roi], 75)
                )
            previous_gray, previous_roi = gray, roi

    slots = max((len(blocks) for blocks in all_blocks), default=0)
    block_boxes = np.full((count, slots, 4), np.nan, np.float32)
    block_corners = np.full((count, slots, 4, 2), np.nan, np.float32)
    block_ids = np.full((count, slots), -1, np.int32)
    block_confidence = np.zeros((count, slots), np.float32)
    for frame, blocks in enumerate(all_blocks):
        for slot, block in enumerate(blocks):
            block_boxes[frame, slot] = block.bbox
            block_corners[frame, slot] = block.corners
            block_ids[frame, slot] = block.track_id
            block_confidence[frame, slot] = block.confidence

    metadata = {
        "schema": "observations_2d_v5",
        "role": role,
        "scene_included": include_scene,
        "keypoint_coordinates": [
            "u_pixels", "v_pixels", "relative_z_normalized", "confidence"
        ],
        "pose_observed": "finite coordinates and min(visibility, presence) >= 0.5",
        "handedness": {
            "-1": "unknown", "0": "MediaPipe Left", "1": "MediaPipe Right"
        },
        "ownership": {
            "-1": "unknown", "0": "other_person", "1": "camera_wearer"
        },
        "ownership_method": "temporal_bottom_entry_heuristic",
        "hand_confidence": "Tasks handedness score proxy; no landmark visibility",
        "head_confidence": "face proxy 0.5; pose fallback <= 0.25; not calibrated",
        "head_method": (
            "face_pose_crop, face_full_frame, pose_landmarks_fallback, or none"
        ),
        "head_transform": (
            "Face: raw Tasks canonical-face transform in inference-image camera "
            "coordinates; crop translation is not full-frame/world translation. "
            "Fallback: pose-derived axes, x right/y up/z toward camera, "
            "zero placeholder translation. Use head_center for image location."
        ),
        "table_mask": "160x160 tabletop occupancy, excluding invalid image",
        "block_persistence": "current detection and >=3 hits in trailing 5 frames",
        "depth": "relative, monocular, non-metric",
        "interpolation": "none",
        "handedness_warning": "Tasks labels retained without assuming mirrored input",
    }
    return {
        "metadata_json": np.array(json.dumps(metadata)),
        "timestamps_s": sequence.sample_times_s,
        "frame_pts_s": sequence.timestamps_s,
        "source_timestamps_s": sequence.source_timestamps_s,
        "source_frame_indices": sequence.source_frame_indices,
        "duration_s": np.array(sequence.duration_s),
        "image_size": np.array([sequence.width, sequence.height]),
        "pose": pose,
        "pose_observed": (
            (pose[..., 3] >= 0.5) & np.isfinite(pose[..., :3]).all(axis=-1)
        ),
        "hands": hands,
        "hands_observed": (
            (hands[..., 3] > 0) & np.isfinite(hands[..., :3]).all(axis=-1)
        ),
        "handedness": handedness,
        "handedness_confidence": handedness_confidence,
        "hand_owner": owner,
        "hand_owner_confidence": owner_confidence,
        "hand_ids": hand_ids,
        "head_center": head_center,
        "head_transform": head_matrix,
        "head_confidence": head_confidence,
        "head_method": head_method,
        "table_masks": table_masks,
        "table_color_lab": table_colors,
        "table_confidence": table_confidence,
        "block_boxes": block_boxes,
        "block_corners": block_corners,
        "block_ids": block_ids,
        "block_confidence": block_confidence,
        "block_observed": block_ids >= 0,
        "block_candidate_count": candidate_count,
        "block_provisional_count": provisional_count,
        "motion_energy": motion,
    }
