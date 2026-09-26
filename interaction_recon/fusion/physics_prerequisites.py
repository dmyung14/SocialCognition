"""Geometry-only physics prerequisites; inferred contact is never contact evidence."""
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from interaction_recon.fusion.triangulation import ray_plane
from interaction_recon.vision.camera import CameraModel


def footprint(pose: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    corners = np.array([
        [-0.5, -0.5, 0], [0.5, -0.5, 0],
        [0.5, 0.5, 0], [-0.5, 0.5, 0],
    ]) * dimensions
    return cv2.convexHull(
        (corners @ rotation.T + pose[:3])[:, :2].astype(np.float32)
    ).reshape(-1, 2)


def footprint_iou(pose_a, dimensions_a, pose_b, dimensions_b) -> float:
    a, b = footprint(pose_a, dimensions_a), footprint(pose_b, dimensions_b)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(np.clip(intersection / max(union, 1e-12), 0, 1))


def _footprint_match(pa, da, pb, db, threshold):
    score = footprint_iou(pa, da, pb, db)
    distance = float(np.linalg.norm(pa[:2] - pb[:2]))
    gate = 0.5 * float(min(np.min(da[:2]), np.min(db[:2])))
    return score >= threshold or distance < gate, score, distance


def consolidate_blocks(
    arrays: dict, *, minimum_frames: int = 5,
    disjoint_iou: float = 0.25, concurrent_iou: float = 0.25,
) -> dict:
    """Complete-link geometric identity hypotheses, including concurrent fragments.

    Initial overlapping footprints cannot be independent tabletop objects in this
    baseline. Concurrent observations also supply median overlap evidence.
    These are uncertain identity hypotheses, not verified physical identities.
    Held samples never supply independent support.
    """
    ids = np.asarray(arrays["block_ids"])
    poses = arrays["block_poses"]
    dimensions = arrays["block_dimensions"]
    count, slots = poses.shape[:2]
    observed = (
        arrays["block_poses_observed_mask"]
        & (arrays["block_poses_confidence"] > 0)
        & np.isfinite(poses).all(axis=-1)
        & np.isfinite(dimensions).all(axis=-1)
        & (dimensions > 0).all(axis=-1)
    )
    source = np.asarray(
        arrays.get("builder_source_frame_indices", np.arange(count))
    )
    unique = np.zeros(count, bool)
    seen = set()
    for i, identity in enumerate(source):
        if identity >= 0 and int(identity) not in seen:
            unique[i] = True
            seen.add(int(identity))
    observed &= unique[:, None]
    support = [np.flatnonzero(observed[:, j]) for j in range(slots)]
    compatible, scores, reasons = {}, {}, {}

    for a in range(slots):
        for b in range(a + 1, slots):
            key = a, b
            compatible[key], scores[key], reasons[key] = False, 0.0, "no_support"
            ia, ib = support[a], support[b]
            if not len(ia) or not len(ib):
                continue
            da = np.median(dimensions[ia, a], axis=0)
            db = np.median(dimensions[ib, b], axis=0)
            first_match, first_score, _ = _footprint_match(
                poses[ia[0], a], da, poses[ib[0], b], db, disjoint_iou
            )
            common = np.intersect1d(ia, ib)
            if len(common):
                evidence = [
                    _footprint_match(
                        poses[i, a], dimensions[i, a],
                        poses[i, b], dimensions[i, b], concurrent_iou,
                    )
                    for i in common
                ]
                concurrent_match = np.mean([item[0] for item in evidence]) >= 0.5
                score = float(np.median([item[1] for item in evidence]))
                accept = first_match or concurrent_match
                reason = (
                    "initial_footprint_overlap_or_near_center"
                    if first_match else "concurrent_footprint_overlap_or_near_center"
                )
                score = max(score, first_score)
            else:
                # Also handles interleaved missing intervals, without treating
                # held boxes as observed identity evidence.
                distances = np.abs(ia[:, None] - ib[None, :])
                row, col = np.unravel_index(np.argmin(distances), distances.shape)
                boundary_match, boundary_score, _ = _footprint_match(
                    poses[ia[row], a], da, poses[ib[col], b], db, disjoint_iou
                )
                accept = first_match or boundary_match
                score = max(first_score, boundary_score)
                reason = (
                    "initial_footprint_overlap_or_near_center"
                    if first_match else "nearest_observed_reacquisition"
                )
            compatible[key], scores[key], reasons[key] = accept, score, reason

    groups = [[j] for j in range(slots)]
    merge_log = []
    for a, b in sorted(scores, key=lambda key: (-scores[key], key)):
        if not compatible[a, b]:
            continue
        ga = next(group for group in groups if a in group)
        gb = next(group for group in groups if b in group)
        if ga is gb:
            continue
        pairs = [tuple(sorted((x, y))) for x in ga for y in gb]
        if not all(compatible[pair] for pair in pairs):
            continue
        merge_log.append({
            "left_ids": ids[ga].astype(int).tolist(),
            "right_ids": ids[gb].astype(int).tolist(),
            "reason": reasons[a, b],
            "minimum_pair_iou": float(min(scores[pair] for pair in pairs)),
            "identity_verified": False,
        })
        ga.extend(gb)
        ga.sort()
        groups.remove(gb)

    extent = arrays.get("table_extent_xy")
    retained, dropped = [], []
    for group in groups:
        frames = np.any(observed[:, group], axis=1)
        supported = int(frames.sum())
        record = {
            "ids": ids[group].astype(int).tolist(),
            "supported_frames": supported,
        }
        if supported < minimum_frames:
            dropped.append({**record, "reason": "insufficient_source_support"})
            continue
        rank = (
            arrays["block_poses_observed_mask"][:, group].astype(float) * 2
            + arrays["block_poses_confidence"][:, group]
        )
        winners = np.asarray(group)[np.argmax(rank, axis=1)]
        row = np.arange(count)
        confident = (
            arrays["block_poses_observed_mask"][row, winners]
            & (arrays["block_poses_confidence"][row, winners] >= 0.05)
            & np.isfinite(poses[row, winners]).all(axis=1)
        )
        candidates = np.flatnonzero(confident & unique)
        first = int(candidates[0] if len(candidates) else np.flatnonzero(frames)[0])
        center = poses[first, winners[first], :2]
        if extent is not None and np.any(np.abs(center) > np.asarray(extent) / 2):
            dropped.append({
                **record,
                "reason": "initial_center_outside_table",
                "initial_center_xy_m": center.tolist(),
                "source_frame": first,
            })
            continue
        retained.append(group)

    retained.sort(key=lambda group: int(ids[group].min()))
    winners = np.zeros((count, len(retained)), int)
    canonical, mapping = [], {}
    for column, group in enumerate(retained):
        canonical.append(int(ids[group].min()))
        rank = (
            arrays["block_poses_observed_mask"][:, group].astype(float) * 2
            + arrays["block_poses_confidence"][:, group]
        )
        winners[:, column] = np.asarray(group)[np.argmax(rank, axis=1)]
        for old in ids[group]:
            mapping[int(old)] = canonical[-1]

    time_fields = (
        "block_poses", "block_dimensions",
        "block_dimension_outlier_mask", "block_dimension_clamped_mask",
        "block_height_prior_mask",
    )
    selected_names = [
        name for name in arrays
        if any(name == field or name.startswith(field + "_") for field in time_fields)
        and np.shape(arrays[name])[:2] == (count, slots)
        and name != "block_dimensions_raw_source"
    ]
    row = np.arange(count)[:, None]
    for name in selected_names:
        arrays[name] = arrays[name][row, winners].copy()
    for name in ("block_dimension_sigma_m", "block_dimension_variance_m2"):
        if name in arrays:
            original = arrays[name]
            arrays[name] = np.asarray([
                np.max(original[group], axis=0) for group in retained
            ]).reshape(-1, 3)
    for column in range(len(retained)):
        active = arrays["block_dimensions_confidence"][:, column] > 0
        good = active & np.isfinite(arrays["block_dimensions"][:, column]).all(axis=1)
        if good.any():
            dims = np.median(arrays["block_dimensions"][good, column], axis=0)
            arrays["block_dimensions"][active, column] = dims
            arrays["block_poses"][active, column, 2] = dims[2] / 2

    arrays["block_ids_before_consolidation"] = ids.copy()
    arrays["block_ids"] = np.asarray(canonical, dtype=ids.dtype)
    arrays["block_original_to_consolidated"] = np.asarray(
        sorted(mapping.items()), dtype=np.int64
    ).reshape(-1, 2)
    report = {
        "block_count": len(retained),
        "block_count_before_consolidation": len(ids),
        "block_merge_log": merge_log,
        "block_dropped_tracks": dropped,
        "minimum_supported_source_frames": minimum_frames,
        "disjoint_footprint_iou_threshold": disjoint_iou,
        "concurrent_footprint_iou_threshold": concurrent_iou,
        "center_distance_threshold_min_footprint_fraction": 0.5,
        "identity_policy": (
            "complete-link geometric hypothesis; concurrent/initial overlap allowed; "
            "no requested object count; initial off-table centers dropped"
        ),
    }
    arrays["block_consolidation_json"] = np.array(json.dumps(report, allow_nan=False))
    return report


def nearby_block_slots(observations: dict, frame: int, hand: np.ndarray) -> np.ndarray:
    """Observed or explicitly held boxes within 25% of their own image size."""
    visible = np.isfinite(hand[:, :2]).all(axis=1) & (hand[:, 3] > 0)
    if visible.sum() < 3:
        return np.empty(0, int)
    low = hand[visible, :2].min(axis=0)
    high = hand[visible, :2].max(axis=0)
    if "block_boxes" in observations:
        boxes = np.asarray(observations["block_boxes"][frame], float)
    else:
        corners = observations["block_corners"][frame]
        boxes = np.concatenate(
            (np.min(corners, axis=1), np.ptp(corners, axis=1)), axis=1
        )
    active = observations["block_observed"][frame].copy()
    held = observations.get("block_interpolated")
    if held is not None:
        active |= held[frame]
    methods = observations.get("block_method")
    if methods is not None:
        active |= np.isin(methods[frame], ["held", "occluded_hold"])
    padding = 0.25 * boxes[:, 2:]
    box_low = boxes[:, :2] - padding
    box_high = boxes[:, :2] + boxes[:, 2:] + padding
    near = np.all(np.minimum(high, box_high) >= np.maximum(low, box_low), axis=1)
    return np.flatnonzero(
        active & near & np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2:] > 0).all(axis=1)
        & (observations["block_ids"][frame] >= 0)
    )


def surface_distance(points, pose, dimensions) -> float:
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=-1)]
    if not len(points):
        return np.nan
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    local = (points - pose[:3]) @ rotation
    outside = np.maximum(np.abs(local) - dimensions / 2, 0)
    return float(np.min(np.linalg.norm(outside, axis=-1)))


def constrain_hand_depth(points, pixels, corners, top_height, model, camera_pose):
    """Rigidly translate a hand to an index/thumb own-ray height hypothesis.

    This is a depth prior, not an attraction to a block center and not measured
    contact. The original hand geometry and all bone lengths are preserved.
    """
    tips = np.array([4, 8])
    good = (
        np.isfinite(points[tips]).all(axis=1)
        & np.isfinite(pixels[tips, :2]).all(axis=1)
        & (pixels[tips, 3] > 0)
    )
    tips = tips[good]
    if not len(tips) or not np.isfinite(corners).all():
        return None
    anchors = ray_plane(
        pixels[tips, :2], model, camera_pose, np.array([0, 0, 1, -top_height])
    )
    low, high = corners.min(axis=0), corners.max(axis=0)
    outside = np.maximum(low - pixels[tips, :2], 0) + np.maximum(
        pixels[tips, :2] - high, 0
    )
    for k in np.argsort(np.linalg.norm(outside, axis=1)):
        if not np.isfinite(anchors[k]).all():
            continue
        shift = anchors[k] - points[tips[k]]
        return points + shift, shift, int(tips[k])
    return None


def smooth_depth_windows(times, shifts, anchored, *, window_s=0.3, bridge_s=0.5):
    """C1 smoothstep transitions; anchors retain full correction.

    Short anchor gaps bridge smoothly. Longer gaps taper to zero; missing
    landmarks are never synthesized by this function.
    """
    times = np.asarray(times, float)
    result = np.zeros_like(shifts, dtype=float)
    weight = np.zeros(len(times))
    nearest = np.full(len(times), -1, int)
    anchors = np.flatnonzero(anchored)
    if not len(anchors):
        return result, weight, nearest

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    for i, time in enumerate(times):
        right = int(np.searchsorted(times[anchors], time))
        left_anchor = int(anchors[right - 1]) if right else None
        right_anchor = int(anchors[right]) if right < len(anchors) else None
        if right_anchor is not None and abs(times[right_anchor] - time) < 1e-9:
            result[i], weight[i], nearest[i] = shifts[right_anchor], 1, right_anchor
            continue
        if left_anchor is not None and right_anchor is not None:
            gap = times[right_anchor] - times[left_anchor]
            if gap <= bridge_s:
                fraction = smooth((time - times[left_anchor]) / gap)
                result[i] = (
                    (1 - fraction) * shifts[left_anchor]
                    + fraction * shifts[right_anchor]
                )
                weight[i] = 1
                nearest[i] = left_anchor if fraction < 0.5 else right_anchor
                continue
        choices = [a for a in (left_anchor, right_anchor) if a is not None]
        anchor = min(choices, key=lambda a: abs(times[a] - time))
        distance = abs(times[anchor] - time)
        if distance < window_s:
            weight[i] = 1 - smooth(distance / window_s)
            result[i] = weight[i] * shifts[anchor]
            nearest[i] = anchor
    return result, weight, nearest


def prepare_physics_observations(
    arrays: dict, raw_builder: dict, builder_camera: dict, sync: dict,
    *, minimum_frames: int = 5,
) -> dict:
    """Apply geometric prerequisites before scene fitting and human IK."""
    times = arrays["timestamps"]
    raw_times = raw_builder["timestamps_s"]
    queries = times + sync["offset_s"]
    indices = np.clip(
        np.searchsorted(raw_times, queries, side="right") - 1, 0, len(raw_times) - 1
    )
    start = float(raw_builder["source_timestamps_s"][0])
    duration = float(raw_builder["duration_s"])
    valid = (queries >= start) & (queries < start + duration)
    source = raw_builder.get("source_frame_indices", np.arange(len(raw_times)))
    arrays["builder_source_frame_indices"] = np.where(valid, source[indices], -1)
    arrays["builder_clip_start_s"] = np.array(start - sync["offset_s"])
    arrays["builder_clip_end_s"] = np.array(start + duration - sync["offset_s"])
    report = consolidate_blocks(arrays, minimum_frames=minimum_frames)
    mapping = dict(arrays["block_original_to_consolidated"].tolist())
    columns = {int(identity): j for j, identity in enumerate(arrays["block_ids"])}
    model = CameraModel.from_arrays(builder_camera)
    cameras = arrays["builder_T_world_from_camera_source"]
    before, after = np.full(len(times), np.nan), np.full(len(times), np.nan)
    shifts = np.zeros((len(times), 2, 3))
    raw_shifts = shifts.copy()
    anchored_ids = np.full((len(times), 2), -1, int)
    anchored_tips = np.full((len(times), 2), -1, int)
    window_ids = anchored_ids.copy()
    window_sources = anchored_ids.copy()
    weights = np.zeros((len(times), 2))
    methods_used = np.full((len(times), 2), "none", dtype="<U48")
    distances_before, distances_after = [[] for _ in times], [[] for _ in times]
    arrays["builder_forearm_keypoints_before_m5_contact_prior"] = arrays[
        "builder_forearm_keypoints"
    ].copy()

    for side, label in enumerate(("left", "right")):
        name = f"builder_{label}_hand_21"
        original = arrays[name].copy()
        arrays[f"{name}_before_m5_contact_prior"] = original
        original_methods = arrays.get(
            f"{name}_method", np.full(len(times), "unspecified", dtype="<U64")
        )
        arrays[f"{name}_method_before_m5"] = original_methods.copy()
        methods = original_methods.astype("<U64")
        anchors = np.zeros(len(times), bool)

        for k, i in enumerate(indices):
            if not valid[k] or not np.isfinite(original[k, 0]).all():
                continue
            slots = np.flatnonzero(
                (raw_builder["hand_owner"][i] == 1)
                & (raw_builder["handedness"][i] == side)
            )
            if not len(slots):
                continue
            slot = max(slots, key=lambda s: raw_builder["hand_owner_confidence"][i, s])
            pixels = raw_builder["hands"][i, slot]
            choices = []
            for block_slot in nearby_block_slots(raw_builder, i, pixels):
                old = int(raw_builder["block_ids"][i, block_slot])
                if old not in mapping:
                    continue
                j = columns[mapping[old]]
                pose = arrays["block_poses"][k, j]
                dims = arrays["block_dimensions"][k, j]
                if not np.isfinite(np.r_[pose, dims]).all():
                    continue
                anchored = constrain_hand_depth(
                    original[k], pixels, raw_builder["block_corners"][i, block_slot],
                    pose[2] + dims[2] / 2, model, cameras[i],
                )
                if anchored is not None:
                    choices.append((
                        np.linalg.norm(anchored[1]), j, block_slot, anchored
                    ))
            if not choices:
                continue
            _, j, block_slot, (_, shift, tip) = min(choices, key=lambda item: item[0])
            anchors[k] = True
            raw_shifts[k, side] = shift
            anchored_ids[k, side] = int(arrays["block_ids"][j])
            anchored_tips[k, side] = tip
            methods_used[k, side] = (
                "observed_box_proximity"
                if raw_builder["block_observed"][i, block_slot]
                else "held_box_proximity"
            )

        correction, weight, nearest = smooth_depth_windows(
            times, raw_shifts[:, side], anchors
        )
        available = valid & np.isfinite(original[:, 0]).all(axis=1) & (weight > 0)
        correction[~available] = 0
        weight[~available] = 0
        shifts[:, side], weights[:, side] = correction, weight

        for k in np.flatnonzero(available):
            anchor = nearest[k]
            arrays[name][k] = original[k] + correction[k]
            methods[k] = (
                "box_proximity_ray_depth_prior"
                if anchors[k] else "smooth_window_ray_depth_prior"
            )
            arrays[f"{name}_confidence"][k] = np.minimum(
                arrays[f"{name}_confidence"][k], 0.12
            )
            window_ids[k, side] = anchored_ids[anchor, side]
            window_sources[k, side] = arrays["builder_source_frame_indices"][anchor]
            forearm = arrays["builder_forearm_keypoints"][k, side]
            finite = np.isfinite(forearm).all(axis=1)
            forearm[finite] += correction[k]
            arrays["builder_forearm_keypoints_confidence"][k, side] = np.minimum(
                arrays["builder_forearm_keypoints_confidence"][k, side], 0.12
            )
            if anchors[k]:
                j = columns[int(anchored_ids[k, side])]
                pose = arrays["block_poses"][k, j]
                dims = arrays["block_dimensions"][k, j]
                distances_before[k].append(surface_distance(original[k, [4, 8]], pose, dims))
                distances_after[k].append(surface_distance(arrays[name][k, [4, 8]], pose, dims))
        arrays[f"{name}_method"] = methods

    for k in range(len(times)):
        if distances_before[k]:
            before[k] = min(distances_before[k])
            after[k] = min(distances_after[k])
    evaluated = np.zeros(len(times), bool)
    seen = set()
    for k in np.flatnonzero(valid & np.isfinite(after)):
        identity = int(arrays["builder_source_frame_indices"][k])
        if identity not in seen:
            evaluated[k] = True
            seen.add(identity)

    arrays.update({
        "hand_block_contact_distance_before_m5_m": before,
        "hand_block_contact_distance_m": after,
        "builder_hand_contact_depth_shift_m": shifts,
        "builder_hand_contact_anchor_shift_m": raw_shifts,
        "builder_hand_contact_prior_block_ids": anchored_ids,
        "builder_hand_contact_prior_tip_indices": anchored_tips,
        "builder_hand_contact_prior_method": methods_used,
        "builder_hand_contact_window_weight": weights,
        "builder_hand_contact_window_block_ids": window_ids,
        "builder_hand_contact_window_source_frames": window_sources,
        "physics_prerequisites_version": np.array("held-proximity-overlap-v2"),
    })
    report.update({
        "hand_block_contact_distance_m": (
            float(np.mean(after[evaluated])) if evaluated.any() else None
        ),
        "hand_block_contact_distance_before_m5_m": (
            float(np.mean(before[evaluated])) if evaluated.any() else None
        ),
        "hand_block_contact_evaluated_source_frames": int(evaluated.sum()),
        "hand_block_contact_distance_definition": (
            "Index/thumb cuboid-surface distance on unique proximity-anchor source "
            "frames, including held boxes. Post-prior diagnostic, NOT independent "
            "evidence of contact."
        ),
        "hand_depth_prior_max_shift_m": float(np.max(np.linalg.norm(shifts, axis=-1))),
        "hand_depth_prior_window_s": 0.3,
        "hand_depth_prior_bridge_s": 0.5,
        "hand_depth_prior_window_applied_samples": int(np.count_nonzero(weights)),
    })
    metrics = json.loads(arrays["metrics_json"].item())
    metrics.update(report)
    metrics["streams"]["builder"]["block_consolidation"] = report
    arrays["metrics_json"] = np.array(json.dumps(metrics, allow_nan=False))
    return arrays
