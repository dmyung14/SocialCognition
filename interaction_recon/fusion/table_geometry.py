"""Sequence-wide table geometry in the initial-camera gauge.

WORLD pose landmarks supply learned metric/gravity evidence, not calibrated
depth. Repeated source frames never contribute additional evidence.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from interaction_recon.fusion.triangulation import ray_plane, transform_points
from interaction_recon.vision.camera import CameraModel


@dataclass(frozen=True)
class NormalCandidate:
    normal_camera: np.ndarray
    frame: int
    confidence: float
    method: str


def unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1e-10:
        return None
    return vector / length


def orient_and_gate(normal: np.ndarray) -> np.ndarray | None:
    """OpenCV axes: x right, y down, z forward; normal points above table.

    A sign ambiguity may be resolved, but an invalid direction is rejected,
    never replaced with a plausible direction and counted as evidence.
    """
    normal = unit(normal)
    if normal is None:
        return None
    if normal[1] > 0:
        normal = -normal
    if -normal[1] <= np.cos(np.deg2rad(80)):
        return None
    if normal[2] >= -1e-4:
        return None
    if abs(normal[0]) >= np.cos(np.deg2rad(25)):
        return None
    return normal


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def robust_normal(
    candidates: list[NormalCandidate], poses: np.ndarray,
    threshold_deg: float = 12.0,
) -> dict:
    """Weighted deterministic spherical RANSAC followed by a spherical median."""
    normals, weights, accepted_indices = [], [], []
    for index, candidate in enumerate(candidates):
        if not np.isfinite(candidate.confidence) or candidate.confidence <= 0:
            continue
        normal = orient_and_gate(candidate.normal_camera)
        if normal is None:
            continue
        rotation = poses[candidate.frame, :3, :3]
        normal = unit(rotation @ normal)
        if normal is not None:
            normals.append(normal)
            weights.append(candidate.confidence)
            accepted_indices.append(index)

    retained = np.zeros(len(candidates), bool)
    if not normals:
        return {
            "normal": None, "inliers": retained, "dispersion_deg": None,
            "inlier_fraction": 0.0, "accepted_candidates": 0,
        }

    normals, weights = np.asarray(normals), np.asarray(weights)
    threshold = np.deg2rad(threshold_deg)
    # Bound the consensus matrix for long recordings.
    seeds = np.unique(np.linspace(0, len(normals) - 1, min(512, len(normals))).astype(int))
    scores = (normals[seeds] @ normals.T >= np.cos(threshold)) @ weights
    normal = normals[seeds[np.argmax(scores)]].copy()
    selected = normals @ normal >= np.cos(threshold)
    for _ in range(30):
        angles = np.arccos(np.clip(normals @ normal, -1, 1))
        influence = weights * selected / np.maximum(angles, np.deg2rad(0.1))
        updated = unit(np.sum(normals * influence[:, None], axis=0))
        if updated is None:
            break
        difference = np.linalg.norm(updated - normal)
        normal = updated
        if difference < 1e-9:
            break

    angles = np.arccos(np.clip(normals @ normal, -1, 1))
    selected = angles <= threshold
    retained[np.asarray(accepted_indices)[selected]] = True
    return {
        "normal": normal,
        "inliers": retained,
        "dispersion_deg": float(np.rad2deg(weighted_median(
            angles[selected], weights[selected]
        ))),
        "inlier_fraction": float(weights[selected].sum() / weights.sum()),
        "accepted_candidates": len(normals),
    }


def edge_normal(
    mask: np.ndarray, image_size: np.ndarray, model: CameraModel,
) -> tuple[np.ndarray | None, float]:
    """Fit four actual hull-boundary lines in undistorted coordinates.

    Opposite lines define two vanishing points, including points at infinity.
    Their cross product is the vanishing line. In normalized coordinates K=I,
    so its coefficients are the normal (equivalent to n ~ K.T @ l).

    Clipped, curved, degenerate, and nonrectangular boundaries abstain. No
    minAreaRect is used: it would invent parallel edges in the image.
    """
    binary = (np.asarray(mask) != 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 32:
        return None, 0.0
    mh, mw = mask.shape
    if (
        (contour[:, 0] <= 1).any() or (contour[:, 0] >= mw - 2).any()
        or (contour[:, 1] <= 1).any() or (contour[:, 1] >= mh - 2).any()
    ):
        return None, 0.0

    pixels = contour.astype(float) * (np.asarray(image_size) / [mw, mh])
    points = model.normalized(pixels)
    if not np.isfinite(points).all():
        return None, 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    perimeter = cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, 0.012 * perimeter, True).reshape(-1, 2)
    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
        return None, 0.0

    # Quantization in a cached mask is larger than one full-resolution pixel.
    pixel_step = float(np.max(np.asarray(image_size) / [mw, mh]))
    tolerance = 1.5 * pixel_step / min(model.K[0, 0], model.K[1, 1])
    distances, fractions = [], []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = b - a
        length = np.linalg.norm(delta)
        if length < 8 * tolerance:
            return None, 0.0
        distances.append(np.abs(np.cross(
            np.c_[points - a, np.zeros(len(points))],
            np.r_[delta / length, 0],
        )[:, 2]))
        fractions.append((points - a) @ delta / (length * length))
    assignment = np.argmin(np.asarray(distances), axis=0)

    lines, residuals = [], []
    for side in range(4):
        selected = (
            (assignment == side)
            & (np.asarray(fractions[side]) > 0.08)
            & (np.asarray(fractions[side]) < 0.92)
        )
        support = points[selected]
        if len(support) < 6:
            return None, 0.0
        center = support.mean(axis=0)
        _, _, vt = np.linalg.svd(support - center, full_matrices=False)
        perpendicular = vt[-1]
        error = np.abs((support - center) @ perpendicular)
        if np.percentile(error, 90) > tolerance:
            return None, 0.0
        lines.append(np.r_[perpendicular, -perpendicular @ center])
        residuals.append(float(np.sqrt(np.mean(error ** 2))))

    first = unit(np.cross(lines[0], lines[2]))
    second = unit(np.cross(lines[1], lines[3]))
    if first is None or second is None or abs(first @ second) > 0.15:
        return None, 0.0
    normal = orient_and_gate(np.cross(first, second))
    if normal is None:
        return None, 0.0
    confidence = 0.9 * np.exp(-np.mean(residuals) / max(tolerance, 1e-9))
    return normal, float(confidence)


def source_unique(observations: dict) -> np.ndarray:
    count = len(observations["timestamps_s"])
    source = np.asarray(observations.get("source_frame_indices", np.arange(count)))
    return np.r_[True, source[1:] != source[:-1]] if count else np.zeros(0, bool)


def normal_candidates(observations: dict, camera: dict) -> list[NormalCandidate]:
    model = CameraModel.from_arrays(camera)
    poses = camera["T_initial_from_camera"]
    unique = source_unique(observations)
    candidates = []
    for i in np.flatnonzero(unique):
        tracking = 1.0 if i == 0 else float(camera["pose_confidence"][i])
        if tracking <= 0:
            continue
        tracking = min(1.0, tracking / 0.5)
        normal, confidence = edge_normal(
            observations["table_masks"][i], observations["image_size"], model
        )
        if normal is not None:
            candidates.append(NormalCandidate(
                normal, i, confidence * tracking * float(observations["table_confidence"][i]),
                "table_edge_vanishing_line",
            ))

        # The existing camera backend ALREADY transports homography normals
        # from camera i-1 into the initial-camera gauge. Undo that rotation for
        # the view-specific gate; robust_normal transports it exactly once.
        homography = camera["table_normal_initial"][i]
        confidence = float(camera["table_normal_confidence"][i])
        previous = max(0, i - 1)
        if np.isfinite(homography).all() and confidence > 0:
            candidates.append(NormalCandidate(
                poses[previous, :3, :3].T @ homography, previous,
                confidence * tracking, "table_region_homography",
            ))

        pose = observations["pose_world"][i]
        visibility = observations["pose"][i, :, 3]
        for upper, lower, weight, method in (
            ([11, 12], [23, 24], 0.40, "WORLD_spine_gravity"),
            ([7, 8], [11, 12], 0.20, "WORLD_head_up_gravity"),
        ):
            selected = upper + lower
            if (
                not np.isfinite(pose[selected]).all()
                or not np.isfinite(visibility[selected]).all()
                or np.min(visibility[selected]) < 0.5
            ):
                continue
            direction = pose[upper].mean(axis=0) - pose[lower].mean(axis=0)
            if np.linalg.norm(direction) < 0.05:
                continue
            candidates.append(NormalCandidate(
                direction, i, weight * tracking * float(np.min(visibility[selected])),
                method,
            ))
    return candidates


def overlapping_blocks(observations: dict, frame: int, hand: np.ndarray) -> np.ndarray:
    """Source slots whose current or explicitly held box overlaps a visible hand."""
    visible = np.isfinite(hand[:, :2]).all(axis=1) & (hand[:, 3] > 0)
    if visible.sum() < 3:
        return np.empty(0, int)
    low, high = hand[visible, :2].min(axis=0), hand[visible, :2].max(axis=0)
    boxes = observations.get("block_boxes")
    if boxes is None:
        corners = observations["block_corners"][frame]
        valid = np.isfinite(corners).all(axis=(1, 2))
        boxes_frame = np.full((len(corners), 4), np.nan)
        boxes_frame[valid, :2] = corners[valid].min(axis=1)
        boxes_frame[valid, 2:] = np.ptp(corners[valid], axis=1)
    else:
        boxes_frame = boxes[frame]
    active = observations["block_observed"][frame].copy()
    active |= observations.get(
        "block_interpolated", np.zeros_like(observations["block_observed"])
    )[frame]
    overlap = np.all(
        np.minimum(high, boxes_frame[:, :2] + boxes_frame[:, 2:])
        > np.maximum(low, boxes_frame[:, :2]), axis=1,
    )
    return np.flatnonzero(
        active & overlap & np.isfinite(boxes_frame).all(axis=1)
        & (observations["block_ids"][frame] >= 0)
    )


def _on_table(pixel: np.ndarray, mask: np.ndarray, image_size: np.ndarray) -> bool:
    if not np.isfinite(pixel).all():
        return False
    xy = np.floor(pixel * np.array(mask.shape[::-1]) / image_size).astype(int)
    return bool(
        0 <= xy[0] < mask.shape[1] and 0 <= xy[1] < mask.shape[0]
        and mask[xy[1], xy[0]] != 0
    )


def height_evidence(
    observations: dict, camera: dict, lifted: dict, normal: np.ndarray,
) -> list[tuple[float, float, str]]:
    evidence = []
    unique = source_unique(observations)
    times = observations["timestamps_s"]
    poses = camera["T_initial_from_camera"]
    for i in np.flatnonzero(unique):
        if i and camera["pose_confidence"][i] <= 0:
            continue
        pose = lifted["pose"][i]
        selected = [11, 12]
        if (
            np.isfinite(pose[selected]).all()
            and np.min(observations["pose"][i, selected, 3]) >= 0.5
        ):
            shoulder = pose[selected].mean(axis=0)
            width = np.linalg.norm(pose[11] - pose[12])
            if 0.20 < width < 0.65:
                # Includes the seated shoulder-above-table prior and learned
                # depth uncertainty; observations are not repeated independent
                # anthropometric measurements.
                distance = np.linalg.norm(shoulder - poses[i, :3, 3])
                sigma = np.hypot(0.075, 0.20 * distance)
                evidence.append((0.325 - shoulder @ normal, sigma, "metric_shoulders"))

        if observations["table_confidence"][i] <= 0.1:
            continue
        for slot, hand in enumerate(observations["hands"][i]):
            wrist = lifted["hands"][i, slot, 0]
            if (
                not np.isfinite(wrist).all() or hand[0, 3] <= 0
                or not _on_table(hand[0, :2], observations["table_masks"][i],
                                 observations["image_size"])
            ):
                continue
            overlaps = overlapping_blocks(observations, i, hand)
            height, method = 0.035, "slow_metric_resting_wrists"
            if len(overlaps):
                height, method = 0.045, "metric_block_contact_wrists"
            else:
                previous_indices = np.flatnonzero(unique[:i])
                if not len(previous_indices):
                    continue
                previous = int(previous_indices[-1])
                identity = observations["hand_ids"][i, slot]
                matches = np.flatnonzero(observations["hand_ids"][previous] == identity)
                if identity < 0 or not len(matches):
                    continue
                before = lifted["hands"][previous, matches[0], 0]
                dt = float(times[i] - times[previous])
                if dt <= 0 or not np.isfinite(before).all():
                    continue
                if np.linalg.norm(wrist - before) / dt > 0.20:
                    continue
            distance = np.linalg.norm(wrist - poses[i, :3, 3])
            sigma = np.hypot(0.045, 0.20 * distance)
            evidence.append((height - wrist @ normal, sigma, method))
    return evidence


def fuse_height(
    evidence: list[tuple[float, float, str]],
    prior_m: float = 0.65, prior_sigma_m: float = 0.40,
    minimum_height_m: float = 0.05,
) -> dict:
    """Fuse independent evidence families, retaining correlated scale uncertainty."""
    groups = {}
    for value, sigma, method in evidence:
        if (
            np.isfinite(value + sigma) and value > minimum_height_m
            and sigma > 0
        ):
            groups.setdefault(method, []).append((value, sigma))
    summaries = []
    for method, measurements in groups.items():
        values = np.asarray(measurements)
        weights = 1 / values[:, 1] ** 2
        center = weighted_median(values[:, 0], weights)
        scatter = 1.4826 * weighted_median(np.abs(values[:, 0] - center), weights)
        sigma = max(float(np.median(values[:, 1])), scatter)
        summaries.append((center, sigma, method))

    if prior_m <= minimum_height_m:
        raise ValueError("Camera trajectory crosses the table-height prior; inspect camera drift")
    values = np.array([prior_m] + [item[0] for item in summaries])
    sigmas = np.array([prior_sigma_m] + [item[1] for item in summaries])
    base_weights = 1 / sigmas ** 2
    center = weighted_median(values, base_weights)
    for _ in range(12):
        standardized = np.abs(values - center) / sigmas
        weights = base_weights * np.minimum(1.0, 1.5 / np.maximum(standardized, 1e-9))
        center = float(np.sum(values * weights) / weights.sum())
    disagreement = float(np.sum(weights * (values - center) ** 2) / weights.sum())
    # Do not average away the common monocular scale systematic.
    sigma = max(float(np.sqrt(1 / weights.sum() + disagreement)), 0.15 * center)
    return {
        "height_m": center, "height_sigma_m": sigma,
        "height_method": "metric_evidence_fused_with_prior" if summaries else "camera_height_prior",
        "height_evidence": [
            {"method": method, "height_m": float(value), "sigma_m": float(spread),
             "samples": len(groups[method])}
            for value, spread, method in summaries
        ],
    }


def table_frame(normal: np.ndarray, point: np.ndarray, x_hint=None) -> np.ndarray:
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    hint = np.array([1.0, 0, 0]) if x_hint is None else np.asarray(x_hint, float)
    x = hint - normal * (hint @ normal)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross([0, 1, 0], normal)
    x /= np.linalg.norm(x)
    rotation = np.stack((x, np.cross(normal, x), normal))
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, -rotation @ point
    return result


def fit_table(observations: dict, camera: dict, lifted: dict) -> dict:
    poses = camera["T_initial_from_camera"]
    candidates = normal_candidates(observations, camera)
    fit = robust_normal(candidates, poses)
    normal = fit["normal"]
    normal_method = "weighted_initial_frame_spherical_consensus"
    if normal is None:
        normal = np.array([0.0, -0.8, -0.6])
        normal_method = "unobserved_camera_tilt_prior"

    evidence = height_evidence(observations, camera, lifted, normal)
    minimum_height = max(0.05, float(-np.min(poses[:, :3, 3] @ normal) + 0.02))
    height = fuse_height(evidence, minimum_height_m=minimum_height)
    distance = height["height_m"]
    plane = np.r_[normal, distance]
    above = poses[:, :3, 3] @ normal + distance
    if np.any(above <= 0):
        raise ValueError("Recovered table plane places a tracked camera below the table")

    model = CameraModel.from_arrays(camera)
    provisional = table_frame(normal, -distance * normal)
    bounds = []
    indices = np.flatnonzero(source_unique(observations))
    indices = indices[::max(1, len(indices) // 40)]
    for i in indices:
        if observations["table_confidence"][i] <= 0:
            continue
        mask = observations["table_masks"][i]
        points = cv2.findNonZero((mask != 0).astype(np.uint8))
        if points is None:
            continue
        hull = cv2.convexHull(points).reshape(-1, 2).astype(float)
        pixels = hull * (observations["image_size"] / np.array(mask.shape[::-1]))
        rays = model.rays(pixels) @ poses[i, :3, :3].T
        incidence = -(rays @ normal)
        points3 = ray_plane(pixels, model, poses[i], plane)
        good = np.isfinite(points3).all(axis=1) & (incidence > 0.05)
        # A horizon-crossing hull is not a metric extent observation.
        if len(hull) < 3 or not good.all():
            continue
        local = transform_points(points3, provisional)[:, :2]
        bounds.append(np.stack((local.min(axis=0), local.max(axis=0))))

    extent = np.array([1.2, 0.8])
    extent_method = "unobserved_extent_prior"
    frame = provisional.copy()
    if bounds:
        low, high = np.median(np.asarray(bounds), axis=0)
        if np.all(high - low > 0.05):
            extent = high - low
            # The rendered table is centered at zero. A median of hull vertices
            # is not the center of its bounding rectangle.
            frame[:2, 3] -= (low + high) / 2
            extent_method = "median_projected_hull_bounds"

    inliers = fit["inliers"]
    supported = int(inliers.sum())
    confidence = (
        min(0.65, 0.55 * fit["inlier_fraction"] * min(1.0, supported / 8))
        if fit["normal"] is not None else 0.03
    )
    if height["height_method"] == "camera_height_prior":
        confidence *= 0.5
    methods = {}
    for keep, candidate in zip(inliers, candidates):
        if keep:
            methods[candidate.method] = methods.get(candidate.method, 0) + 1
    projections = [
        abs(value - distance) for value, _, method in evidence if "wrists" in method
    ]
    return {
        "transform": frame, "plane_initial": plane, "extent": extent,
        "confidence": float(confidence), **height,
        "normal_method": normal_method,
        "normal_dispersion_deg": fit["dispersion_deg"],
        "rest_contact_residual_m": float(np.median(projections)) if projections else None,
        "rest_candidates": len(projections),
        "normal_candidate_count": len(candidates),
        "normal_gate_accepted_count": fit["accepted_candidates"],
        "normal_inlier_count": supported,
        "normal_inlier_methods": methods,
        "extent_method": extent_method,
        "minimum_camera_height_m": float(above.min()),
        "candidate_normals_camera": np.asarray(
            [c.normal_camera for c in candidates], float
        ).reshape(-1, 3),
        "candidate_frames": np.asarray([c.frame for c in candidates], int),
        "candidate_weights": np.asarray([c.confidence for c in candidates], float),
        "candidate_methods": np.asarray([c.method for c in candidates], dtype="<U40"),
        "candidate_inliers": inliers,
    }


def contact_height_anchor(
    points: np.ndarray, pixels: np.ndarray, block_corners: np.ndarray,
    top_height: float, model: CameraModel, camera_pose: np.ndarray,
) -> np.ndarray | None:
    """Weak contact hypothesis: anchor a fingertip on its OWN ray, not a centroid.

    A 2D overlap is not proof of contact. The caller must label this as a prior
    and preserve a diagnostic computed before this constraint.
    """
    tips = np.array([4, 8, 12, 16, 20])
    low, high = block_corners.min(axis=0), block_corners.max(axis=0)
    valid = (
        np.isfinite(points[tips]).all(axis=1)
        & np.isfinite(pixels[tips, :2]).all(axis=1)
        & (pixels[tips, 3] > 0)
    )
    inside = (
        (pixels[tips, :2] >= low).all(axis=1)
        & (pixels[tips, :2] <= high).all(axis=1)
    )
    eligible = tips[valid & inside]
    if not len(eligible):
        return None
    joint = eligible[np.argmin(np.abs(points[eligible, 2] - top_height))]
    # Reject a clear noncontact metric hypothesis instead of pulling a raised
    # hand onto every object it happens to overlap in the image.
    if abs(points[joint, 2] - top_height) > 0.12:
        return None
    anchor = ray_plane(
        pixels[joint:joint + 1, :2], model, camera_pose,
        np.array([0.0, 0, 1, -top_height]),
    )[0]
    if not np.isfinite(anchor).all():
        return None
    return points + anchor - points[joint]


def contact_distance(
    hands: list[np.ndarray], centroids: np.ndarray,
) -> float:
    if not hands or not len(centroids):
        return np.nan
    tips = np.concatenate([hand[[4, 8, 12, 16, 20]] for hand in hands])
    tips = tips[np.isfinite(tips).all(axis=1)]
    centroids = centroids[np.isfinite(centroids).all(axis=1)]
    if not len(tips) or not len(centroids):
        return np.nan
    return float(np.min(np.linalg.norm(tips[:, None] - centroids[None], axis=-1)))
