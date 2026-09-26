"""Scene-only recomputation; no MediaPipe inference or human-cache changes."""
import json

import cv2
import numpy as np

from interaction_recon.io.media import MediaSequence
from interaction_recon.vision.blocks import (
    BLOCK_PERCEPTION_VERSION, BlockDetector, PersistentBlockTracker,
    region_fraction, warp_points,
)
from interaction_recon.vision.camera import match_features
from interaction_recon.vision.hands import HandObservation
from interaction_recon.vision.table import TableDetector
from interaction_recon.vision.tracking import _hand_mask


def _metric_pixel_scale(humans, index):
    values = []
    if "hands_world" not in humans:
        return None
    for pixels, world in zip(humans["hands"][index], humans["hands_world"][index]):
        for joint in (5, 9, 13, 17):
            if pixels[[0, joint], 3].min() <= 0:
                continue
            metric = np.linalg.norm(world[joint] - world[0])
            image = np.linalg.norm(pixels[joint, :2] - pixels[0, :2])
            if np.isfinite(metric + image) and 0.02 < metric < 0.15 and image > 2:
                values.append(image / metric)
    return float(np.median(values)) if values else None


def _table_motion(previous, gray, previous_mask, mask):
    a, b = match_features(previous, gray, previous_mask, mask)
    if len(a) < 12:
        return np.eye(3), 0.0
    H, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5, maxIters=800)
    if H is None or inliers is None or inliers.sum() < 10:
        return np.eye(3), 0.0
    H /= H[2, 2]
    if not np.isfinite(H).all() or np.linalg.cond(H) > 1e7:
        return np.eye(3), 0.0
    h, w = gray.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    transformed = warp_points(corners, H)
    ratio = abs(cv2.contourArea(transformed)) / (w * h)
    if not 0.6 < ratio < 1.6:
        return np.eye(3), 0.0
    return H, float(np.mean(inliers))


def refine_scene(
    sequence: MediaSequence, human_observations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = len(sequence.rgb_frames)
    if len(human_observations["hands"]) != count:
        raise ValueError("Human observations and decoded frame counts differ")
    np.testing.assert_allclose(
        human_observations["timestamps_s"], sequence.sample_times_s,
        rtol=0, atol=1e-7,
    )
    table_detector, detector = TableDetector(), BlockDetector()
    tracker = PersistentBlockTracker()
    masks = np.zeros((count, 160, 160), np.uint8)
    exclusions = np.zeros_like(masks)
    colors = np.full((count, 3), np.nan, np.float32)
    table_confidence = np.zeros(count, np.float32)
    candidate_count = np.zeros(count, np.int32)
    motion = np.zeros(count)
    transforms = np.repeat(np.eye(3)[None], count, axis=0)
    motion_confidence = np.zeros(count)
    frames = []
    previous = previous_mask = previous_small = None
    source_ids = np.asarray(sequence.source_frame_indices)
    if source_ids.shape != (count,):
        raise ValueError("Scene tracking requires source_frame_indices")
    unique_source = np.r_[True, source_ids[1:] != source_ids[:-1]]

    for index, rgb in enumerate(sequence.rgb_frames):
        if not unique_source[index]:
            frames.append(frames[-1])
            for array in (
                masks, exclusions, colors, table_confidence, candidate_count,
                transforms, motion_confidence,
            ):
                array[index] = array[index - 1]
            continue
        hands = [
            HandObservation(
                p, int(human_observations["handedness"][index, slot]),
                float(human_observations["handedness_confidence"][index, slot]),
            )
            for slot, p in enumerate(human_observations["hands"][index])
            if np.any((p[:, 3] > 0) & np.isfinite(p[:, :2]).all(axis=1))
        ]
        hand_mask = _hand_mask(hands, rgb.shape)
        forearms = human_observations.get("wearer_forearms_2d")
        if forearms is not None:
            for arm in forearms[index]:
                if np.isfinite(arm[:, :2]).all() and np.all(arm[:, 3] > 0):
                    a, b = np.round(arm[:, :2]).astype(int)
                    cv2.line(
                        hand_mask, tuple(a), tuple(b), 255,
                        max(5, round(min(rgb.shape[:2]) * 0.055)),
                    )
        table = table_detector.detect(rgb, hand_mask)
        detections = detector.detect(
            rgb, table.mask, hand_mask, table_color_lab=table.color_lab,
            pixels_per_meter=_metric_pixel_scale(human_observations, index),
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        background = detector.table_only.astype(np.uint8) * 255
        background = cv2.erode(background, np.ones((5, 5), np.uint8))
        if previous is not None:
            H, confidence = _table_motion(previous, gray, previous_mask, background)
            transforms[index] = H @ transforms[index - 1]
            transforms[index] /= transforms[index, 2, 2]
            motion_confidence[index] = confidence
        else:
            motion_confidence[index] = 1.0
        tracked = tracker.update(
            detections, transforms[index], detector.exclusion, rgb.shape
        )
        frames.append(tracked)
        candidate_count[index] = len(detections)
        masks[index] = cv2.resize(
            detector.support, (160, 160), interpolation=cv2.INTER_NEAREST
        )
        exclusions[index] = cv2.resize(
            detector.exclusion, (160, 160), interpolation=cv2.INTER_NEAREST
        )
        colors[index], table_confidence[index] = table.color_lab, table.confidence
        small = cv2.resize(gray, (160, 160)).astype(float)
        roi = (masks[index] != 0) | (exclusions[index] != 0)
        if previous_small is not None and roi.any():
            motion[index] = np.percentile(np.abs(small - previous_small)[roi], 75)
        previous, previous_mask, previous_small = gray, background, small

    identities = sorted({d.track_id for frame in frames for d in frame})
    accepted, reports = [], []
    scale = np.array([160 / sequence.width, 160 / sequence.height])
    for identity in identities:
        records = {
            i: next(d for d in frame if d.track_id == identity)
            for i, frame in enumerate(frames)
            if any(d.track_id == identity for d in frame)
        }
        observed_indices = [i for i, d in records.items() if d.observed and unique_source[i]]
        if not observed_indices:
            continue
        first = observed_indices[0]
        reference = warp_points(records[first].corners, np.linalg.inv(transforms[first]))
        hits = eligible = 0
        for i in range(count):
            if not unique_source[i]:
                continue
            record = records.get(i)
            if record is not None and record.observed:
                reference = warp_points(record.corners, np.linalg.inv(transforms[i]))
            projected = warp_points(reference, transforms[i])
            inside_image = (
                (projected[:, 0] >= 0) & (projected[:, 0] < sequence.width)
                & (projected[:, 1] >= 0) & (projected[:, 1] < sequence.height)
            ).all()
            small_corners = projected * scale
            visible = (
                inside_image
                and region_fraction(small_corners, masks[i]) >= 0.85
                and region_fraction(small_corners, exclusions[i]) < 0.12
            )
            if visible:
                eligible += 1
                hits += int(record is not None and record.observed)
        fraction = hits / eligible if eligible else 0.0
        keep = hits >= 3 and fraction >= 0.30
        reports.append({
            "id": identity, "eligible_source_frames": eligible,
            "supported_source_frames": hits, "support_fraction": fraction,
            "accepted": keep,
        })
        if keep:
            accepted.append(identity)

    slots = len(accepted)
    boxes = np.full((count, slots, 4), np.nan, np.float32)
    corners = np.full((count, slots, 4, 2), np.nan, np.float32)
    ids = np.full((count, slots), -1, np.int32)
    confidence = np.zeros((count, slots), np.float32)
    observed = np.zeros((count, slots), bool)
    interpolated = np.zeros_like(observed)
    method = np.full((count, slots), "missing", dtype="<U24")
    lookup = {identity: slot for slot, identity in enumerate(accepted)}
    provisional = candidate_count.copy()
    for i, frame in enumerate(frames):
        for d in frame:
            if d.track_id not in lookup:
                continue
            j = lookup[d.track_id]
            boxes[i, j], corners[i, j], ids[i, j] = d.bbox, d.corners, d.track_id
            confidence[i, j] = d.confidence
            observed[i, j], interpolated[i, j], method[i, j] = (
                d.observed, d.interpolated, d.method,
            )
            provisional[i] -= int(d.observed)
    return {
        "scene_metadata_json": np.array(json.dumps({
            "schema": BLOCK_PERCEPTION_VERSION,
            "block_persistence": ">=3 hits and >=30% eligible nonoccluded SOURCE frames",
            "block_observed": "accepted track, current source-frame detection",
            "interpolation": "occluded_hold only; no unoccluded missing-frame filling",
            "coordinates": "pixels; association in homography-stabilized table chart",
            "camera_failure": "identity relative motion, zero confidence; association uncertain",
            "height": "no side-face correspondence inferred by this segmenter",
        })),
        "block_support_json": np.array(json.dumps(reports)),
        "table_masks": masks,
        "table_color_lab": colors,
        "table_confidence": table_confidence,
        "block_boxes": boxes,
        "block_corners": corners,
        "block_ids": ids,
        "block_confidence": confidence,
        "block_observed": observed,
        "block_interpolated": interpolated,
        "block_method": method,
        "block_candidate_count": candidate_count,
        "block_provisional_count": np.maximum(provisional, 0),
        "block_image_from_table": transforms,
        "block_table_motion_confidence": motion_confidence,
        "block_unique_source_frame": unique_source,
        "motion_energy": motion,
    }
