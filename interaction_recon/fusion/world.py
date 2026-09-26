"""Uncertain common world. Only builder blocks are physical scene objects."""
import json
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from interaction_recon.fusion.block_geometry import (
    MAX_SIDE_M, export_blocks, lift_blocks, plausible_block_dimensions,
)
from interaction_recon.fusion.filtering import filter_track
from interaction_recon.fusion.table_geometry import (
    contact_distance, contact_height_anchor, fit_table, overlapping_blocks,
    source_unique, table_frame,
)
from interaction_recon.fusion.triangulation import (
    lift_metric, ransac_similarity, ray_plane, transform_points,
)
from interaction_recon.vision.camera import CameraModel, nearest_rotation
from interaction_recon.vision.hands import assign_pose_wrists

REQUIRED_OBSERVATIONS = (
    "timestamps", "guider_head_pose", "guider_torso_pose",
    "guider_arm_keypoints", "guider_left_hand_21", "guider_right_hand_21",
    "builder_forearm_keypoints", "builder_left_hand_21", "builder_right_hand_21",
    "block_poses", "block_dimensions", "table_plane",
)
TRACK_NAMES = REQUIRED_OBSERVATIONS[1:]
MAX_BLOCK_SIDE_M = MAX_SIDE_M


def _finite_median(values, default=np.nan):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float(default)


def _finite_mean(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def _quaternion(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def seated_orientation_invariant(head, shoulders, torso, *, raise_on_failure=False):
    head_z = np.asarray(head)[..., 2]
    shoulder_z = np.mean(np.asarray(shoulders)[..., 2], axis=-1)
    torso_z = np.asarray(torso)[..., 2]
    head_valid = np.isfinite(head_z) & np.isfinite(shoulder_z)
    torso_valid = np.isfinite(torso_z)
    inverted = head_valid & (head_z < shoulder_z)
    below = torso_valid & (torso_z < 0)
    evaluated, failed = head_valid | torso_valid, inverted | below
    report = {
        "status": "failed" if failed.any() else "passed" if evaluated.any() else "not_evaluable",
        "severity": "error" if failed.any() else "info",
        "evaluated_frames": int(evaluated.sum()),
        "head_shoulder_evaluated_frames": int(head_valid.sum()),
        "torso_evaluated_frames": int(torso_valid.sum()),
        "head_below_shoulders_frames": int(inverted.sum()),
        "torso_below_table_frames": int(below.sum()),
        "failed_frames": int(failed.sum()),
        "failure_fraction": float(failed.sum() / evaluated.sum()) if evaluated.any() else None,
        "head_z_median_m": _finite_mean([_finite_median(head_z)]),
        "shoulder_z_median_m": _finite_mean([_finite_median(shoulder_z)]),
        "torso_z_median_m": _finite_mean([_finite_median(torso_z)]),
    }
    if failed.any() and raise_on_failure:
        raise ValueError(f"Seated orientation invariant FAILED: {report}")
    return report


def _pose_invariant(pose):
    return seated_orientation_invariant(
        pose[:, [0, 7, 8]].mean(axis=1), pose[:, [11, 12]],
        pose[:, [11, 12, 23, 24]].mean(axis=1),
    )


def _body_up(pose):
    up = pose[:, [11, 12]].mean(axis=1) - pose[:, [23, 24]].mean(axis=1)
    valid = np.isfinite(up).all(axis=1) & (np.linalg.norm(up, axis=1) > 0.1)
    if not valid.any():
        return None
    result = np.median(up[valid] / np.linalg.norm(up[valid], axis=1)[:, None], axis=0)
    norm = np.linalg.norm(result)
    return result / norm if norm > 0.1 else None


def orient_table_normal(normal, body_up=None):
    """Legacy standalone helper; rejected directions are not estimator evidence."""
    from interaction_recon.fusion.table_geometry import orient_and_gate
    result = orient_and_gate(normal)
    if result is None or (body_up is not None and result @ body_up < 0.15):
        return np.array([0.0, -0.8, -0.6])
    return result


def seated_actor_prior(pose, far_edge_y):
    transform = np.eye(4)
    shoulders = pose[:, [11, 12]].mean(axis=1)
    valid = np.isfinite(shoulders).all(axis=1)
    report = {"method": "unavailable", "confidence": 0.0, "translation_m": [0.0] * 3}
    if not valid.any():
        return transform, report
    center, up = np.median(shoulders[valid], axis=0), _body_up(pose)
    rotation = np.eye(3)
    if up is not None:
        target = np.array([0.0, 0, 1])
        axis = np.cross(up, target)
        sine, cosine = np.linalg.norm(axis), float(np.clip(np.dot(up, target), -1, 1))
        if sine > 1e-8:
            rotation = Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine)).as_matrix()
        elif cosine < 0:
            rotation = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    relative = (pose - center) @ rotation.T
    torso_relative = relative[:, [11, 12, 23, 24]].mean(axis=1)
    shoulder_height = min(0.40, max(0.28, 0.10 - _finite_median(torso_relative[:, 2], -0.18)))
    torso_y = _finite_median(torso_relative[:, 1], 0)
    head_y = _finite_median(relative[:, [0, 7, 8], 1], 0)
    target_center = np.array([
        0.0, far_edge_y + 0.08 - min(0.0, torso_y, head_y), shoulder_height,
    ])
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ center
    report = {
        "method": "seated_upright_actor_only_role_prior", "confidence": 0.12,
        "translation_m": transform[:3, 3].tolist(),
        "rotation_degrees": float(np.rad2deg(Rotation.from_matrix(rotation).magnitude())),
        "shoulder_height_prior_m": shoulder_height, "camera_and_table_changed": False,
        "raw_invariant": _pose_invariant(pose),
    }
    return transform, report


def metric_depth_prior(observations, model):
    depths, uncertainties, bones = [], [], []
    for i in np.flatnonzero(source_unique(observations)):
        for slot in range(observations["hands"].shape[1]):
            pixel, world = observations["hands"][i, slot], observations["hands_world"][i, slot]
            _, depth, uncertainty, _ = lift_metric(pixel, world, model)
            if np.isfinite(depth):
                depths.append(depth)
                uncertainties.append(uncertainty)
                bones.extend(np.linalg.norm(world[[5, 9, 13, 17]] - world[0], axis=1))
        _, depth, uncertainty, _ = lift_metric(
            observations["pose"][i], observations["pose_world"][i], model, (23, 24), 0.5,
        )
        if np.isfinite(depth):
            depths.append(depth)
            uncertainties.append(uncertainty)
            world = observations["pose_world"][i]
            bones.append(np.linalg.norm(world[11] - world[12]))
    if not depths:
        return {
            "depth_m": 1.2, "depth_sigma_m": 1.0, "confidence": 0.0, "samples": 0,
            "method": "unobserved_depth_prior", "median_metric_bone_m": None,
        }
    depth = float(np.median(depths))
    scatter = 1.4826 * float(np.median(np.abs(np.asarray(depths) - depth)))
    return {
        "depth_m": depth,
        "depth_sigma_m": max(float(np.median(uncertainties)), scatter, depth * 0.25),
        "confidence": min(0.65, len(depths) / 60), "samples": len(depths),
        "method": "mediapipe_WORLD_hand_bones_and_pose_shoulders",
        "median_metric_bone_m": _finite_median(bones),
    }


def _lift_stream(observations, camera):
    model = CameraModel.from_arrays(camera)
    count = len(observations["timestamps_s"])
    pose = np.full((count, 33, 3), np.nan)
    hands = np.full((count, 4, 21, 3), np.nan)
    pose_camera, hands_camera = pose.copy(), hands.copy()
    errors, depth_sigma = [], []
    for i in range(count):
        points, _, sigma, error = lift_metric(
            observations["pose"][i], observations["pose_world"][i], model, (23, 24), 0.5,
        )
        pose_camera[i] = points
        pose[i] = transform_points(points, camera["T_initial_from_camera"][i])
        errors.append(error)
        depth_sigma.append(sigma)
        for slot in range(4):
            points, _, sigma, error = lift_metric(
                observations["hands"][i, slot], observations["hands_world"][i, slot], model
            )
            hands_camera[i, slot] = points
            hands[i, slot] = transform_points(points, camera["T_initial_from_camera"][i])
            errors.append(error)
            depth_sigma.append(sigma)
    return {
        "pose": pose, "hands": hands, "pose_camera": pose_camera, "hands_camera": hands_camera,
        "reprojection_px": _finite_mean(errors), "depth_sigma_m": _finite_mean(depth_sigma),
    }


def _fit_table(observations, camera, lifted):
    return fit_table(observations, camera, lifted)


def _blocks(observations, camera, frame):
    return lift_blocks(observations, camera, frame)


def _sample_indices(times, queries, duration, source_start):
    indices = np.searchsorted(times, queries, side="right") - 1
    valid = (queries >= source_start) & (queries < source_start + duration)
    return np.clip(indices, 0, len(times) - 1), valid


def _alignment(streams, observations, sync):
    guider, builder = streams["guider"], streams["builder"]
    ga, ba = observations["guider"], observations["builder"]
    gb, bb = guider["blocks"], builder["blocks"]
    source, target, pairs, frame_ids = [], [], [], []
    gi, inside = _sample_indices(
        ga["timestamps_s"], ba["timestamps_s"] - sync["offset_s"],
        float(ga["duration_s"]), float(ga["source_timestamps_s"][0]),
    )
    for bi, g in enumerate(gi):
        if not inside[bi]:
            continue
        gv, bv = np.flatnonzero(gb["observed"][g]), np.flatnonzero(bb["observed"][bi])
        cost = np.full((len(gv), len(bv)), 1e6)
        for row, a in enumerate(gv):
            for col, b in enumerate(bv):
                d1, d2 = np.sort(gb["dimensions"][g, a]), np.sort(bb["dimensions"][bi, b])
                dimension = np.linalg.norm(np.log(d1 / d2))
                color = np.linalg.norm(gb["colors"][g, a] - bb["colors"][bi, b])
                if dimension < 0.8 and np.isfinite(color) and color < 28:
                    cost[row, col] = dimension + color / 28
        if not cost.size:
            continue
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if cost[row, col] >= 1e6:
                continue
            a, b = gv[row], bv[col]
            source.append(gb["poses"][g, a, :3])
            target.append(bb["poses"][bi, b, :3])
            pairs.append((int(a), int(b)))
            frame_ids.append(bi)
    report = {
        "status": "not_estimated", "verified_fusion": False,
        "fusion_method": "role_prior_fallback", "confidence": 0.05,
        "candidate_correspondences": len(source), "inlier_fraction": 0.0,
        "landmark_rmse_m": None, "landmark_error_normalized": None,
        "shared_human_landmarks": 0,
        "reason": "No independently verified shared object or actor identities.",
    }
    if len(source) < 12 or len(set(frame_ids)) < 5 or len(set(pairs)) < 3:
        return report, np.eye(4), []
    training = np.asarray(frame_ids) % 3 != 0
    source, target = np.asarray(source), np.asarray(target)
    if training.sum() < 6 or (~training).sum() < 3:
        return report, np.eye(4), []
    fit = ransac_similarity(source[training], target[training])
    residual = np.linalg.norm(transform_points(source, fit["transform"]) - target, axis=1)
    error = float(np.sqrt(np.mean(residual[~training] ** 2)))
    report.update(
        status="unverified_geometry_hypothesis",
        inlier_fraction=float(np.mean(residual < 0.06)),
        landmark_rmse_m=error,
        landmark_error_normalized=error / max(_finite_median(bb["dimensions"], 0.1), 0.01),
        reason="Appearance/shape matches are ambiguous; no cross-stream block merge authorized.",
    )
    return report, np.eye(4), []


def _add_track(
    output, name, times, value, confidence, source, observed=None,
    pose=False, fast=False,
):
    arguments = {
        "observed": observed, "max_speed": 20.0 if fast else 8.0,
        "min_cutoff": 3.0 if fast else 1.5, "beta": 5.0 if fast else 2.0,
    }
    if pose:
        position = filter_track(times, value[..., :3], confidence, source, **arguments)
        rotation = filter_track(
            times, value[..., 3:], confidence, source, quaternion=True, **arguments
        )
        result = position
        result["value"] = np.concatenate((position["value"], rotation["value"]), axis=-1)
        result["confidence"] = np.minimum(position["confidence"], rotation["confidence"])
        for key in ("observed_mask", "interpolated_mask", "rejected_mask"):
            result[key] = (
                position[key] & rotation[key] if key == "observed_mask"
                else position[key] | rotation[key]
            )
    else:
        result = filter_track(times, value, confidence, source, **arguments)
    for suffix, array in result.items():
        output[name if suffix == "value" else f"{name}_{suffix}"] = array


def lift_forearm_blob(points, model, camera_pose):
    result = np.full((2, 3), np.nan)
    if not np.isfinite(points[:, :2]).all() or np.any(points[:, 3] <= 0):
        return result
    for joint, height in enumerate((0.18, 0.08)):
        result[joint] = ray_plane(
            points[joint:joint + 1, :2], model, camera_pose, np.array([0.0, 0, 1, -height]),
        )[0]
    if not np.isfinite(result).all():
        return np.full((2, 3), np.nan)
    direction = result[0] - result[1]
    length = np.linalg.norm(direction)
    if length < 0.04:
        return np.full((2, 3), np.nan)
    if length > 0.38:
        result[0] = result[1] + direction * (0.30 / length)
    return result


def _actor_tracks(stream, observations, role):
    pose, hands = stream["lifted"]["pose"], stream["lifted"]["hands"]
    count, result = len(pose), {}
    actor_conf = 0.12 if stream.get("actor_prior_applied", False) else 0.35
    pose_confidence = observations["pose"][..., 3] * actor_conf
    if role == "guider":
        result["guider_arm_keypoints"] = (
            pose[:, [[11, 13, 15], [12, 14, 16]]],
            pose_confidence[:, [[11, 13, 15], [12, 14, 16]]], None,
        )
        torso = np.full((count, 7), np.nan)
        head = torso.copy()
        torso_conf, head_conf = np.zeros(count), np.zeros(count)
        for i, landmarks in enumerate(pose):
            if np.isfinite(landmarks[[11, 12, 23, 24]]).all():
                shoulders = landmarks[[11, 12]].mean(axis=0)
                hips = landmarks[[23, 24]].mean(axis=0)
                x, up = landmarks[12] - landmarks[11], shoulders - hips
                x /= max(np.linalg.norm(x), 1e-9)
                up -= x * np.dot(up, x)
                up /= max(np.linalg.norm(up), 1e-9)
                if np.linalg.norm(np.cross(x, up)) > 0.9:
                    rotation = np.column_stack((x, np.cross(up, x), up))
                    torso[i] = np.r_[(shoulders + hips) / 2, _quaternion(rotation)]
                    torso_conf[i] = np.min(pose_confidence[i, [11, 12, 23, 24]])
            if np.isfinite(landmarks[[0, 7, 8]]).all():
                raw = observations["head_transform"][i, :3, :3]
                if np.isfinite(raw).all() and observations["head_confidence"][i] > 0:
                    rotation = stream["camera_world"][i, :3, :3] @ np.diag([1, -1, -1])
                    rotation = stream["actor_transform"][:3, :3] @ rotation @ nearest_rotation(raw)
                    head[i] = np.r_[landmarks[[0, 7, 8]].mean(axis=0), _quaternion(rotation)]
                    head_conf[i] = observations["head_confidence"][i] * actor_conf
        result["guider_torso_pose"] = torso, torso_conf, None
        result["guider_head_pose"] = head, head_conf, None

    width, height = observations["image_size"]
    selection = np.full((count, 2), -1, np.int32)
    for i in range(count):
        if role == "guider":
            selection[i] = assign_pose_wrists(
                observations["hands"][i], observations["pose"][i], width, height,
                eligible=observations["hand_owner"][i] != 1,
            )
        else:
            for side in (0, 1):
                candidates = np.flatnonzero(
                    (observations["hand_owner"][i] == 1)
                    & (observations["handedness"][i] == side)
                )
                if len(candidates):
                    selection[i, side] = max(
                        candidates, key=lambda s: observations["hand_owner_confidence"][i, s]
                    )
    selected_hands, selected_conf = [], []
    hand_method = np.full((count, 2), "none", dtype="<U64")
    model = stream["model"]
    for side, label in enumerate(("left", "right")):
        value, confidence = np.full((count, 21, 3), np.nan), np.zeros((count, 21))
        for i, slot in enumerate(selection[:, side]):
            if slot < 0:
                continue
            points = hands[i, slot].copy()
            if not np.isfinite(points[0]).all():
                continue
            method = "metric_hand_root"
            if role == "guider" and np.isfinite(pose[i, 15 + side]).all():
                points += pose[i, 15 + side] - points[0]
                method = "pose_wrist_anchored_metric_hand"
            elif role == "builder":
                pixels = observations["hands"][i, slot]
                overlaps = overlapping_blocks(observations, i, pixels)
                for block_slot in overlaps:
                    identity = observations["block_ids"][i, block_slot]
                    matches = np.flatnonzero(stream["blocks"]["ids"] == identity)
                    if not len(matches):
                        continue
                    j = matches[0]
                    dimensions = stream["blocks"]["dimensions"][i, j]
                    if not np.isfinite(dimensions).all():
                        continue
                    anchored = contact_height_anchor(
                        points, pixels, observations["block_corners"][i, block_slot],
                        float(dimensions[2]), model, stream["camera_world"][i],
                    )
                    if anchored is not None:
                        points = anchored
                        method = "overlap_fingertip_contact_height_prior"
                        break
                if method == "metric_hand_root" and not 0.01 <= points[0, 2] <= 0.45:
                    anchor = ray_plane(
                        pixels[0:1, :2], model, stream["camera_world"][i],
                        np.array([0, 0, 1, -0.08]),
                    )[0]
                    if not np.isfinite(anchor).all():
                        continue
                    points += anchor - points[0]
                    method = "metric_hand_table_height_prior"
            value[i] = points
            confidence[i] = (
                observations["hands"][i, slot, :, 3]
                * observations["hand_owner_confidence"][i, slot] * 0.5
            )
            if role == "guider":
                confidence[i] = np.minimum(confidence[i], actor_conf)
            if method.endswith("_prior"):
                confidence[i] = np.minimum(confidence[i], 0.12)
            hand_method[i, side] = method
        result[f"{role}_{label}_hand_21"] = value, confidence, None
        selected_hands.append(value)
        selected_conf.append(confidence)
    stream["hand_lift_method"] = hand_method
    stream["hand_selection"] = selection

    if role == "builder":
        forearms = np.full((count, 2, 2, 3), np.nan)
        confidence, observed = np.zeros((count, 2, 2)), np.zeros((count, 2, 2), bool)
        method = np.full((count, 2), "none", dtype="<U64")
        fallback = observations.get("wearer_forearms_2d")
        for side, hand in enumerate(selected_hands):
            wrist, palm = hand[:, 0], hand[:, [5, 9, 13, 17]].mean(axis=1)
            direction = wrist - palm
            norm = np.linalg.norm(direction, axis=1)
            good = np.isfinite(direction).all(axis=1) & (norm > 0.01)
            forearms[good, side, 1] = wrist[good]
            forearms[good, side, 0] = wrist[good] + direction[good] / norm[good, None] * 0.25
            confidence[good, side, 1] = selected_conf[side][good, 0]
            confidence[good, side, 0] = 0.08
            observed[good, side, 1] = True
            method[good, side] = "hand_axis_forearm_length_prior"
            if fallback is not None:
                for i in np.flatnonzero(~good):
                    points = lift_forearm_blob(fallback[i, side], model, stream["camera_world"][i])
                    if np.isfinite(points).all():
                        forearms[i, side], confidence[i, side] = points, 0.08
                        method[i, side] = "border_skin_blob_pca_table_height_prior"
        stream["forearm_method"] = method
        result["builder_forearm_keypoints"] = forearms, confidence, observed
    return result


def _reject_output(output, name, mask):
    output[name][mask] = np.nan
    output[f"{name}_confidence"][mask] = 0
    output[f"{name}_source"][mask] = 0
    output[f"{name}_observed_mask"][mask] = False
    output[f"{name}_interpolated_mask"][mask] = False
    output[f"{name}_rejected_mask"][mask] = True


def _contact_metrics(output, stream, raw, indices, valid):
    count = len(output["timestamps"])
    distances, unconstrained = np.full(count, np.nan), np.full(count, np.nan)
    evaluated_source = set()
    samples = []
    for k, i in enumerate(indices):
        if not valid[k]:
            continue
        source = int(raw.get("source_frame_indices", np.arange(len(raw["timestamps_s"])))[i])
        final_hands, raw_hands = [], []
        for side, label in enumerate(("left", "right")):
            slot = stream["hand_selection"][i, side]
            if slot < 0 or not len(overlapping_blocks(raw, i, raw["hands"][i, slot])):
                continue
            final_hands.append(output[f"builder_{label}_hand_21"][k])
            raw_hands.append(stream["lifted"]["hands"][i, slot])
        active = output["block_poses_confidence"][k] > 0
        centers = output["block_poses"][k, active, :3]
        distances[k] = contact_distance(final_hands, centers)
        unconstrained[k] = contact_distance(raw_hands, centers)
        if source not in evaluated_source and np.isfinite(distances[k]):
            evaluated_source.add(source)
            samples.append(k)
    output["hand_block_contact_distance_m"] = distances
    output["hand_block_contact_distance_before_hand_priors_m"] = unconstrained
    return {
        "hand_block_contact_distance_m": _finite_mean(distances[samples]),
        "hand_block_contact_distance_before_hand_priors_m": _finite_mean(unconstrained[samples]),
        "hand_block_contact_evaluated_source_frames": len(samples),
        "hand_block_contact_distance_definition": (
            "Mean of source-frame minimum fingertip-to-nearest-centroid distances; "
            "only selected builder hands with 2D block-box overlap. Includes explicitly "
            "held blocks. Post-filter value is not independent validation of contact priors."
        ),
    }


def reconstruct_world(observations, cameras, sync, fps):
    streams = {}
    for role in ("guider", "builder"):
        lifted = _lift_stream(observations[role], cameras[role])
        table = _fit_table(observations[role], cameras[role], lifted)
        frame = table["transform"]
        for name in ("pose", "hands"):
            lifted[name] = transform_points(lifted[name], frame)
        streams[role] = {
            "lifted": lifted, "table": table,
            "blocks": _blocks(observations[role], cameras[role], frame),
            "camera_world": frame[None] @ cameras[role]["T_initial_from_camera"],
            "model": CameraModel.from_arrays(cameras[role]),
            "actor_transform": np.eye(4), "actor_prior_applied": False,
            "layout_transform": np.eye(4),
        }

    consistency, alignment, mapping = _alignment(streams, observations, sync)
    verified = bool(consistency.get("verified_fusion", False))
    if verified and mapping:
        raise NotImplementedError("Verified block fusion requires a correspondence-aware merger")
    table_extent = streams["builder"]["table"]["extent"]
    guider = streams["guider"]
    raw_guider_pose = guider["lifted"]["pose"].copy()
    actor_transform, actor_report = seated_actor_prior(raw_guider_pose, table_extent[1] / 2)
    guider["actor_transform"] = actor_transform
    guider["actor_prior_applied"] = actor_report["method"] != "unavailable"
    for name in ("pose", "hands"):
        guider["lifted"][name] = transform_points(guider["lifted"][name], actor_transform)

    start = min(
        float(observations["guider"]["source_timestamps_s"][0]),
        float(observations["builder"]["source_timestamps_s"][0]) - sync["offset_s"],
    )
    end = max(
        float(observations["guider"]["source_timestamps_s"][0])
        + float(observations["guider"]["duration_s"]),
        float(observations["builder"]["source_timestamps_s"][0])
        + float(observations["builder"]["duration_s"]) - sync["offset_s"],
    )
    times = start + np.arange(max(1, int(np.ceil((end - start) * fps - 1e-8)))) / fps
    output, samples = {"timestamps": times}, {}
    camera_poses = np.full((len(times), 2, 4, 4), np.nan)
    camera_confidence = np.zeros((len(times), 2))
    for role_index, role in enumerate(("guider", "builder")):
        raw, stream = observations[role], streams[role]
        query = times + (sync["offset_s"] if role == "builder" else 0)
        indices, valid = _sample_indices(
            raw["timestamps_s"], query, float(raw["duration_s"]),
            float(raw["source_timestamps_s"][0]),
        )
        samples[role] = indices, valid
        camera_poses[valid, role_index] = stream["camera_world"][indices[valid]]
        camera_confidence[valid, role_index] = cameras[role]["pose_confidence"][indices[valid]]
        for name, (value, confidence, observed) in _actor_tracks(stream, raw, role).items():
            value, confidence = value[indices].copy(), confidence[indices].copy()
            value[~valid], confidence[~valid] = np.nan, 0
            finite = np.isfinite(value).all(axis=-1)
            confidence[~finite] = 0
            source = np.where(confidence > 0, 1 << role_index, 0).astype(np.uint8)
            if observed is not None:
                observed = observed[indices].copy()
                observed[~valid] = False
                observed &= finite
            _add_track(
                output, name, times, value, confidence, source, observed,
                pose=name.endswith("_pose"), fast="_hand_21" in name,
            )
        output[f"{role}_T_world_from_camera_source"] = stream["camera_world"]
        output[f"{role}_table_from_initial"] = stream["table"]["transform"]
        output[f"{role}_layout_transform"] = stream["layout_transform"]
        output[f"{role}_actor_prior_transform"] = stream["actor_transform"]
        output[f"{role}_table_plane_initial"] = stream["table"]["plane_initial"]
        output[f"{role}_table_extent_xy"] = stream["table"]["extent"]
        for field in (
            "candidate_normals_camera", "candidate_frames", "candidate_weights",
            "candidate_methods", "candidate_inliers",
        ):
            output[f"{role}_table_{field}"] = stream["table"][field]
        table_report = {
            name: value for name, value in stream["table"].items()
            if not isinstance(value, np.ndarray)
        }
        output[f"{role}_table_geometry_json"] = np.array(json.dumps(table_report, allow_nan=False))
        method = stream["hand_lift_method"][indices].copy()
        method[~valid] = "none"
        for side, label in enumerate(("left", "right")):
            name = f"{role}_{label}_hand_21"
            published = method[:, side].copy()
            published[np.any(output[f"{name}_interpolated_mask"], axis=1)] = "temporal_interpolation"
            output[f"{name}_method"] = published
        if role == "builder":
            method = stream["forearm_method"][indices].copy()
            method[~valid] = "none"
            method[np.any(
                output["builder_forearm_keypoints_interpolated_mask"], axis=2
            )] = "temporal_interpolation"
            output["builder_forearm_keypoints_method"] = method

    output["guider_pose_raw_table_source"] = raw_guider_pose
    output["guider_actor_prior_applied"] = np.array(guider["actor_prior_applied"])
    output["guider_actor_prior_json"] = np.array(json.dumps(actor_report, allow_nan=False))
    for role, prefix, bit in (("builder", "", 2), ("guider", "guider_", 1)):
        indices, valid = samples[role]
        export_blocks(
            output, streams[role]["blocks"], indices, valid, prefix=prefix, source_bit=bit
        )
        output[f"{role}_block_source_timestamps"] = observations[role]["timestamps_s"]
        output[f"{role}_block_support_json"] = observations[role].get(
            "block_support_json", np.array("[]")
        )
    output["table_extent_xy"] = table_extent
    output["camera_poses"], output["camera_confidence"] = camera_poses, camera_confidence
    output["table_plane"] = np.repeat([[0.0, 0, 1, 0]], len(times), axis=0)
    table_conf = min(streams[role]["table"]["confidence"] for role in streams)
    output["table_plane_confidence"] = np.full(len(times), table_conf)
    output["table_plane_observed_mask"] = np.zeros(len(times), bool)
    output["table_plane_interpolated_mask"] = np.zeros(len(times), bool)
    output["table_plane_source"] = np.full(len(times), 3, np.uint8)
    output["table_plane_rejected_mask"] = np.zeros(len(times), bool)

    invariant = seated_orientation_invariant(
        output["guider_head_pose"][..., :3], output["guider_arm_keypoints"][:, :, 0],
        output["guider_torso_pose"][..., :3],
    )
    if invariant["status"] == "failed":
        warnings.warn(
            f"Seated orientation invariant FAILED: {invariant}", RuntimeWarning, stacklevel=2
        )
    metrics = {
        "scope": "Uncertain metric reconstruction; no physics",
        "cross_view_consistency": consistency,
        "cross_view_landmark_consistency": {
            "rmse_m": consistency["landmark_rmse_m"],
            "normalized_by_block_length": consistency["landmark_error_normalized"],
            "kind": "unverified candidate block centers, never used to merge physical sets",
        },
        "seated_orientation_invariant": invariant,
        "raw_seated_orientation_invariant": _pose_invariant(raw_guider_pose),
        "guider_actor_layout": actor_report,
        "canonical_layout_valid": invariant["status"] == "passed",
        "block_dimension_policy": {
            "minimum_side_m": 0.005, "maximum_side_m": MAX_BLOCK_SIDE_M,
            "maximum_height_m": MAX_BLOCK_SIDE_M,
            "oversize_action": "per-track median clamped to priors; raw evidence and flags retained",
            "height": "matched side-face edge only, otherwise explicit 25mm prior",
        },
        "streams": {},
    }
    metrics.update(_contact_metrics(
        output, streams["builder"], observations["builder"], *samples["builder"]
    ))
    for role, stream in streams.items():
        table, blocks = stream["table"], stream["blocks"]
        raw = observations[role]
        unique = source_unique(raw)
        per_track = []
        for j, identity in enumerate(blocks["ids"]):
            per_track.append({
                "id": int(identity),
                "observed_source_frames": int(np.sum(blocks["observed"][unique, j])),
                "occluded_hold_source_frames": int(np.sum(blocks["interpolated"][unique, j])),
                "source_frames": int(unique.sum()),
                "observed_source_coverage": float(np.mean(blocks["observed"][unique, j])),
                "available_source_coverage": float(np.mean(
                    blocks["observed"][unique, j] | blocks["interpolated"][unique, j]
                )),
            })
        eligible = unique[:, None] & blocks["observed"]
        finite = np.isfinite(blocks["poses"][..., :3]).all(axis=-1)
        within = np.all(
            np.abs(blocks["poses"][..., :2]) <= table["extent"] / 2 + 1e-6, axis=-1
        )
        evaluated = eligible & finite
        fraction = float(np.mean(within[evaluated])) if evaluated.any() else None
        table_report = json.loads(output[f"{role}_table_geometry_json"].item())
        metrics["streams"][role] = {
            "camera_pose_coverage": float(np.mean(cameras[role]["pose_confidence"] > 0)),
            "camera_translation_coverage": float(np.mean(
                cameras[role]["translation_confidence"] > 0
            )),
            "table_plane_consistency": table_report,
            "blocks_within_table_extent_fraction": fraction,
            "blocks_within_table_extent_evaluated_samples": int(evaluated.sum()),
            "blocks_within_table_extent_definition": (
                "Observed finite centroids within independently estimated centered table "
                "bounds; unique source frames only; no coordinate clipping."
            ),
            "blocks_on_builder_half_fraction": (
                float(np.mean(blocks["poses"][..., 1][evaluated] < 0))
                if evaluated.any() else None
            ),
            "block_dimension_variance_m2": _finite_mean(blocks["dimension_variance"]),
            "block_dimension_outlier_count": int(blocks["dimension_outlier"].sum()),
            "block_dimension_clamped_count": int(blocks["dimension_clamped"].sum()),
            "block_tracks": per_track,
            "block_support": json.loads(raw.get("block_support_json", np.array("[]")).item()),
            "scale": json.loads(cameras[role]["metric_scale_prior_json"].item()),
            "camera_height_m": table["height_m"],
            "camera_height_sigma_m": table["height_sigma_m"],
            "lifted_keypoint_reprojection_error_px": stream["lifted"]["reprojection_px"],
            "reprojection_scope": "raw metric lifting BEFORE actor/contact/height priors and filtering",
            "depth_sigma_m": stream["lifted"]["depth_sigma_m"],
        }
    metrics["blocks_within_table_extent_fraction"] = metrics["streams"]["builder"][
        "blocks_within_table_extent_fraction"
    ]
    output["fusion_method"] = np.array(consistency["fusion_method"])
    output["fusion_confidence"] = np.array(consistency["confidence"])
    output["verified_fusion"] = np.array(verified)
    output["metrics_json"] = np.array(json.dumps(metrics, allow_nan=False))
    output["metadata_json"] = np.array(json.dumps({
        "schema": "observations_3d_global_table_v4",
        "units": "meters, seconds, radians", "pose_layout": "xyz,wxyz",
        "coordinates": "builder canonical table z-up; guider blocks in separate guider table gauge",
        "source_bits": {"0": "missing", "1": "guider", "2": "builder", "3": "verified_both"},
        "block_set": "builder only; guider_block_* are nonphysical separate-view observations",
        "table": "one robust initial-frame normal and plane offset per stream",
        "table_center": "midpoint of median projected hull bounds; never block-position fitting",
        "block_hold": "last metric pose, interpolated_mask true, method occluded_hold",
        "block_height": "matched side-face correspondence or explicitly flagged prior",
        "block_dimensions": "track median, clamped 5mm..25cm; raw evidence retained",
        "block_filter": "no generic interpolation or smoothing that relabels holds",
        "human_filter": "short-gap One-Euro/SLERP",
        "hand_contact": "weak overlap and metric-height-gated fingertip own-ray height prior",
        "observed_mask": "detection-derived target, not ground-truth metric measurement",
        "limitations": (
            "Uncalibrated intrinsics, learned metric depth/gravity, camera drift, "
            "clipped or curved tables, and ambiguous contact remain uncertain. "
            "No source-specific dimensions or forced builder-half block placement."
        ),
    }))
    validate_observations(output)
    return output


def validate_observations(arrays):
    missing = set(REQUIRED_OBSERVATIONS) - arrays.keys()
    names = list(TRACK_NAMES)
    names.extend(n for n in ("guider_block_poses", "guider_block_dimensions") if n in arrays)
    for name in names:
        for suffix in ("confidence", "observed_mask", "interpolated_mask", "source"):
            if f"{name}_{suffix}" not in arrays:
                missing.add(f"{name}_{suffix}")
    if missing:
        raise ValueError(f"observations.npz lacks keys: {sorted(missing)}")
    count = len(arrays["timestamps"])
    if np.any(np.diff(arrays["timestamps"]) <= 0):
        raise ValueError("Observation timestamps must increase")
    for name in names:
        value = arrays[name]
        if len(value) != count:
            raise ValueError(f"{name}: timeline mismatch")
        for suffix in ("confidence", "observed_mask", "interpolated_mask", "source"):
            if arrays[f"{name}_{suffix}"].shape != value.shape[:-1]:
                raise ValueError(f"{name}_{suffix}: shape mismatch")
        observed = arrays[f"{name}_observed_mask"]
        interpolated = arrays[f"{name}_interpolated_mask"]
        if np.any(observed & interpolated):
            raise ValueError(f"{name}: interpolated values cannot be observed")
        if np.any((observed | interpolated) & ~np.isfinite(value).all(axis=-1)):
            raise ValueError(f"{name}: nonfinite published values")
    for name in ("block_dimensions", "guider_block_dimensions"):
        if name in arrays:
            dimensions = arrays[name]
            finite = np.isfinite(dimensions).all(axis=-1)
            if np.any(finite & ~plausible_block_dimensions(dimensions)):
                raise ValueError("Published block dimensions violate the metric size prior")
    if not bool(arrays.get("verified_fusion", False)):
        if np.any(arrays["block_poses_source"] & 1):
            raise ValueError("Unverified guider observations cannot enter the physical block set")
