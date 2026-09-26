"""Block-only metric lifting and provenance-preserving export."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from interaction_recon.fusion.triangulation import ray_plane, transform_points
from interaction_recon.vision.camera import CameraModel

MIN_SIDE_M = 0.005
MAX_SIDE_M = 0.25
HEIGHT_PRIOR_M = 0.025


def plausible_block_dimensions(dimensions):
    dimensions = np.asarray(dimensions)
    return (
        np.isfinite(dimensions).all(axis=-1)
        & (dimensions >= MIN_SIDE_M).all(axis=-1)
        & (dimensions <= MAX_SIDE_M).all(axis=-1)
    )


def side_face_height(bottom_pixel, top_pixel, model, camera_pose):
    """Fit a vertical segment only when the caller supplies a matched side edge.

    Merely having a bounding box is NOT a side-face correspondence.
    """
    bottom = ray_plane(
        np.asarray(bottom_pixel)[None], model, camera_pose, np.array([0, 0, 1, 0])
    )[0]
    if not np.isfinite(bottom).all():
        return np.nan
    ray = model.rays(np.asarray(top_pixel)[None])[0] @ camera_pose[:3, :3].T
    origin = camera_pose[:3, 3]
    matrix = np.column_stack((ray, -np.array([0.0, 0, 1])))
    if np.linalg.cond(matrix) > 100:
        return np.nan
    distance, height = np.linalg.lstsq(matrix, bottom - origin, rcond=None)[0]
    residual = np.linalg.norm(origin + distance * ray - bottom - [0, 0, height])
    if distance <= 0 or height <= 0 or residual > 0.01:
        return np.nan
    reprojection = model.project(
        transform_points((bottom + [0, 0, height])[None], np.linalg.inv(camera_pose))
    )[0]
    return float(height) if np.linalg.norm(reprojection - top_pixel) < 3 else np.nan


def lift_blocks(observations, camera, frame):
    model = CameraModel.from_arrays(camera)
    identities = np.unique(observations["block_ids"])
    identities = identities[identities >= 0]
    count, slots = len(observations["timestamps_s"]), len(identities)
    shape = count, slots
    poses = np.full(shape + (7,), np.nan)
    dimensions = np.full(shape + (3,), np.nan)
    raw = dimensions.copy()
    colors = dimensions.copy()
    confidence = np.zeros(shape)
    observed = np.zeros(shape, bool)
    interpolated = np.zeros(shape, bool)
    outlier = np.zeros(shape, bool)
    clamped = np.zeros(shape, bool)
    height_prior = np.ones(shape, bool)
    method = np.full(shape, "missing", dtype="<U24")
    side_edges = observations.get("block_side_face_edges")
    side_valid = observations.get("block_side_face_observed")
    held_input = observations.get(
        "block_interpolated", np.zeros_like(observations["block_observed"])
    )
    last = {}
    for i in range(count):
        transform = frame @ camera["T_initial_from_camera"][i]
        for slot, identity in enumerate(observations["block_ids"][i]):
            if identity < 0:
                continue
            j = int(np.searchsorted(identities, identity))
            if held_input[i, slot]:
                if j in last:
                    poses[i, j] = last[j][0]
                    confidence[i, j] = min(
                        last[j][1] * 0.25,
                        float(observations["block_confidence"][i, slot]) * 0.35,
                    )
                    interpolated[i, j] = True
                    method[i, j] = "occluded_hold"
                continue
            if not observations["block_observed"][i, slot]:
                continue
            footprint = ray_plane(
                observations["block_corners"][i, slot], model, transform,
                np.array([0, 0, 1, 0]),
            )
            if not np.isfinite(footprint).all():
                outlier[i, j] = True
                continue
            (x, y), (sx, sy), angle = cv2.minAreaRect(footprint[:, :2].astype(np.float32))
            if sx < sy:
                sx, sy, angle = sy, sx, angle + 90
            angle %= 180
            height = np.nan
            if side_edges is not None and side_valid is not None and side_valid[i, slot]:
                height = side_face_height(
                    side_edges[i, slot, 0], side_edges[i, slot, 1], model, transform
                )
            if np.isfinite(height):
                height_prior[i, j] = False
            else:
                height = HEIGHT_PRIOR_M
            raw[i, j] = [sx, sy, height]
            if sx <= 0 or sy <= 0:
                outlier[i, j] = True
                continue
            clamped[i, j] = not bool(plausible_block_dimensions(raw[i, j]))
            q = Rotation.from_euler("z", angle, degrees=True).as_quat()[[3, 0, 1, 2]]
            poses[i, j] = [x, y, height / 2, *q]
            confidence[i, j] = float(observations["block_confidence"][i, slot]) * 0.35
            observed[i, j] = True
            method[i, j] = "observed"
            colors[i, j] = camera["block_lab"][i, slot]
            last[j] = poses[i, j].copy(), confidence[i, j]

    sigma = np.full((slots, 3), np.nan)
    variance = sigma.copy()
    unique = np.asarray(observations.get("source_frame_indices", np.arange(count)))
    unique_mask = np.r_[True, unique[1:] != unique[:-1]] if count else np.zeros(0, bool)
    for j in range(slots):
        good = observed[:, j] & np.isfinite(raw[:, j]).all(axis=-1) & unique_mask
        if not good.any():
            continue
        threshold = np.median(confidence[good, j])
        robust = good & (confidence[:, j] >= threshold)
        values = raw[robust, j]
        median = np.median(values, axis=0)
        measured_heights = good & ~height_prior[:, j]
        if measured_heights.any():
            median[2] = np.median(raw[measured_heights, j, 2])
        else:
            median[2] = HEIGHT_PRIOR_M
        variance[j] = np.var(values, axis=0)
        sigma[j] = np.maximum(
            1.4826 * np.median(np.abs(values - median), axis=0),
            median * [0.25, 0.25, 0.75],
        )
        final = np.clip(median, MIN_SIDE_M, MAX_SIDE_M)
        valid = confidence[:, j] > 0
        clamped[valid, j] |= np.any(final != median)
        dimensions[valid, j] = final
        poses[valid, j, 2] = final[2] / 2
        confidence[clamped[:, j], j] *= 0.5
    return {
        "ids": identities, "poses": poses, "dimensions": dimensions,
        "raw_dimensions": raw, "dimension_sigma": sigma,
        "dimension_variance": variance, "dimension_clamped": clamped,
        "dimension_outlier": outlier, "height_prior": height_prior,
        "confidence": confidence, "colors": colors, "observed": observed,
        "interpolated": interpolated, "method": method,
    }


def export_blocks(output, blocks, indices, valid, *, prefix="", source_bit=2):
    """No generic interpolation: an occluded hold never becomes observed."""
    confidence = blocks["confidence"][indices].copy()
    confidence[~valid] = 0
    observed = blocks["observed"][indices].copy()
    interpolated = blocks["interpolated"][indices].copy()
    observed[~valid] = False
    interpolated[~valid] = False
    source = np.where(confidence > 0, source_bit, 0).astype(np.uint8)
    method = blocks["method"][indices].copy()
    method[~valid] = "missing"
    rejected = blocks["dimension_outlier"][indices].copy()
    rejected[~valid] = False
    for suffix, field in (("poses", "poses"), ("dimensions", "dimensions")):
        name = f"{prefix}block_{suffix}"
        value = blocks[field][indices].copy()
        value[~valid] = np.nan
        output[name] = value
        output[f"{name}_confidence"] = confidence.copy()
        output[f"{name}_observed_mask"] = observed.copy()
        output[f"{name}_interpolated_mask"] = interpolated.copy()
        output[f"{name}_source"] = source.copy()
        output[f"{name}_method"] = method.copy()
        output[f"{name}_rejected_mask"] = rejected.copy()
    output[f"{prefix}block_ids"] = blocks["ids"]
    output[f"{prefix}block_dimensions_raw_source"] = blocks["raw_dimensions"]
    output[f"{prefix}block_dimension_sigma_m"] = blocks["dimension_sigma"]
    output[f"{prefix}block_dimension_variance_m2"] = blocks["dimension_variance"]
    for suffix, field in (
        ("dimension_outlier_mask", "dimension_outlier"),
        ("dimension_clamped_mask", "dimension_clamped"),
        ("height_prior_mask", "height_prior"),
    ):
        value = blocks[field][indices].copy()
        value[~valid] = False
        output[f"{prefix}block_{suffix}"] = value
