"""Record metric landmarks and explicit egocentric fallback provenance."""
import json
from pathlib import Path

import numpy as np

from interaction_recon.vision import tracking
from interaction_recon.vision.hands import (
    HandObservation, OwnershipClassifier, arm_blob_forearms, assign_pose_wrists,
)


class _Recorder:
    def __init__(self, detector, kind: str, records: list, methods: list | None = None):
        self.detector = detector
        self.kind = kind
        self.records = records
        self.methods = methods

    def detect(self, rgb: np.ndarray, timestamp_ms: int):
        result = self.detector.detect(rgb, timestamp_ms)
        if self.kind == "body":
            world = getattr(self.detector, "world_landmarks", None)
            self.records.append(
                np.full((33, 3), np.nan, np.float32)
                if world is None else np.asarray(world, np.float32).copy()
            )
        else:
            world = np.full((4, 21, 3), np.nan, np.float32)
            methods = np.full(4, "none", dtype="<U40")
            for slot, hand in enumerate(result[:4]):
                if hand.world_keypoints is not None:
                    world[slot] = hand.world_keypoints
                methods[slot] = getattr(hand, "method", "mediapipe_full")
            self.records.append(world)
            self.methods.append(methods)
        return result

    def close(self) -> None:
        self.detector.close()


def track_metric_stream(sequence, role: str, model_dir: str | Path, **kwargs) -> dict:
    pose_world, hands_world, detection_methods = [], [], []
    original_factory = kwargs.pop("detector_factory", None) or tracking.create_detectors

    def factory(paths):
        body, hands, head = original_factory(paths)
        return (
            _Recorder(body, "body", pose_world),
            _Recorder(hands, "hands", hands_world, detection_methods),
            head,
        )

    arrays = tracking.track_stream(
        sequence, role, model_dir, detector_factory=factory, **kwargs
    )
    arrays["pose_world"] = np.asarray(pose_world, dtype=np.float32)
    arrays["hands_world"] = np.asarray(hands_world, dtype=np.float32)
    arrays["hand_detection_method"] = np.asarray(detection_methods)
    count = len(sequence.rgb_frames)
    assignment = np.full((count, 4), "unassigned", dtype="<U40")
    forearms = np.full((count, 2, 2, 4), np.nan, np.float32)
    forearms[..., 3] = 0
    forearm_method = np.full((count, 2), "none", dtype="<U40")
    raw_handedness = arrays["handedness"].copy()
    ownership = OwnershipClassifier()

    for i, rgb in enumerate(sequence.rgb_frames):
        blobs = arm_blob_forearms(rgb)
        slots = np.flatnonzero(np.any(arrays["hands_observed"][i], axis=1))
        detected = [
            HandObservation(
                arrays["hands"][i, slot],
                int(raw_handedness[i, slot]),
                float(arrays["handedness_confidence"][i, slot]),
            )
            for slot in slots
        ]
        owners, confidence, ids = ownership.update(
            detected, sequence.width, sequence.height,
            pose=arrays["pose"][i], forearms=blobs,
        )
        arrays["hand_owner"][i, slots] = owners
        arrays["hand_owner_confidence"][i, slots] = confidence
        arrays["hand_ids"][i, slots] = ids
        matches = assign_pose_wrists(
            arrays["hands"][i], arrays["pose"][i],
            sequence.width, sequence.height,
        )
        arrays["handedness"][i] = -1
        for side, slot in enumerate(matches):
            if slot >= 0:
                arrays["handedness"][i, slot] = side
                assignment[i, slot] = "pose_wrist_proximity"
        for slot in slots:
            if arrays["hand_owner"][i, slot] != 1:
                continue
            wrist = arrays["hands"][i, slot, 0, :2]
            connected = [
                blob for blob in blobs
                if np.linalg.norm(
                    (wrist - blob.keypoints[1, :2]) / [sequence.width, sequence.height]
                ) < 0.16
            ]
            if connected:
                blob = min(
                    connected,
                    key=lambda item: np.linalg.norm(wrist - item.keypoints[1, :2]),
                )
                side = blob.side
                method = "wearer_border_arm"
            elif (
                wrist[1] > sequence.height * 0.80
                or wrist[0] < sequence.width * 0.10
                or wrist[0] > sequence.width * 0.90
            ):
                side = int(wrist[0] >= sequence.width / 2)
                method = "wearer_border_side_prior"
            else:
                # Tasks handedness is unreliable on egocentric back-of-hand views.
                side = int(wrist[0] >= sequence.width / 2)
                method = "wearer_image_half_prior"
            arrays["handedness"][i, slot] = side
            assignment[i, slot] = method

        for blob in blobs:
            # Do not replace a detected hand with a color-only joint hypothesis.
            has_hand = np.any(
                (arrays["hand_owner"][i] == 1)
                & (arrays["handedness"][i] == blob.side)
                & np.any(arrays["hands_observed"][i], axis=1)
            )
            if role == "builder" and not has_hand:
                forearms[i, blob.side] = blob.keypoints
                forearm_method[i, blob.side] = blob.method

    arrays["handedness_raw"] = raw_handedness
    arrays["hand_assignment_method"] = assignment
    arrays["wearer_forearms_2d"] = forearms
    arrays["wearer_forearm_method"] = forearm_method
    metadata = json.loads(arrays["metadata_json"].item())
    metadata.update({
        "world_landmarks": {
            "units": "meters",
            "pose_origin": "MediaPipe hip center",
            "hand_origin": "MediaPipe hand geometric center",
            "meaning": "learned metric prior, not measured depth",
            "axes_assumption": "OpenCV camera x right, y down, z forward",
        },
        "hand_passes": "independent IMAGE full, 25% black padding, downscaled padding; IoU NMS",
        "handedness": {
            "-1": "unassigned", "0": "anatomical left hypothesis",
            "1": "anatomical right hypothesis",
        },
        "handedness_raw": "unchanged Tasks selfie labels",
        "ownership_method": "pose wrist association first, then border/arm connection and temporal prior",
        "forearm_fallback": (
            "skin component PCA; border entry and blob tip are joint hypotheses, "
            "not observed anatomical elbow/wrist; no fingers fabricated"
        ),
    })
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    return arrays
