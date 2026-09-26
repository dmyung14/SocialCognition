import cv2
import numpy as np


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target = scale * R * source + translation; reflections are forbidden."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must have matching (N,3) shapes")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity needs at least three finite correspondences")
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2:
        raise ValueError("Similarity correspondences are collinear or coincident")
    covariance = b.T @ a / len(a)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(a * a, axis=1))
    scale = float(np.dot(singular, sign) / max(variance, 1e-12))
    if scale <= 0:
        raise ValueError("Nonpositive similarity scale")
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    return scale, rotation, translation


def similarity_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray):
    result = np.eye(4)
    result[:3, :3] = scale * rotation
    result[:3, 3] = translation
    return result


def ransac_similarity(
    source: np.ndarray, target: np.ndarray,
    threshold: float = 0.06, iterations: int = 256, seed: int = 23,
) -> dict:
    source, target = np.asarray(source, float), np.asarray(target, float)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    indices = np.flatnonzero(valid)
    best = np.zeros(len(source), bool)
    rng = np.random.default_rng(seed)
    if len(indices) >= 3:
        for _ in range(iterations):
            sample = rng.choice(indices, 3, replace=False)
            try:
                scale, rotation, translation = umeyama(source[sample], target[sample])
            except ValueError:
                continue
            residual = np.linalg.norm(
                scale * source @ rotation.T + translation - target, axis=1
            )
            inliers = valid & (residual <= threshold)
            if inliers.sum() > best.sum():
                best = inliers
    if best.sum() < 3:
        return {"transform": np.eye(4), "inliers": best, "rmse_m": None, "scale": None}
    scale, rotation, translation = umeyama(source[best], target[best])
    residual = np.linalg.norm(
        scale * source @ rotation.T + translation - target, axis=1
    )
    best = valid & (residual <= threshold)
    return {
        "transform": similarity_matrix(scale, rotation, translation),
        "inliers": best,
        "rmse_m": float(np.sqrt(np.mean(residual[best] ** 2))) if best.any() else None,
        "scale": scale,
    }


def lift_metric(
    pixels: np.ndarray, metric: np.ndarray, model,
    root: int | tuple[int, ...] = 0, minimum_confidence: float = 0.2,
) -> tuple[np.ndarray, float, float, float]:
    """Root a WORLD skeleton on its image ray, solving metric root z robustly.

    MediaPipe relative geometry, including its camera-axis assumption, is a
    learned prior. This is not triangulation or independently measured depth.
    """
    empty = np.full(np.asarray(metric).shape, np.nan, float)
    roots = np.atleast_1d(root)
    good = (
        np.isfinite(metric).all(axis=-1)
        & np.isfinite(pixels[:, :2]).all(axis=-1)
        & (pixels[:, 3] >= minimum_confidence)
    )
    if not np.all(good[roots]) or good.sum() < 4:
        return empty, np.nan, np.nan, np.nan
    xy = model.normalized(pixels[:, :2])
    anchor = np.mean(xy[roots], axis=0)
    relative = metric - np.mean(metric[roots], axis=0)
    A = (xy - anchor)[good].reshape(-1)
    B = (relative[:, :2] - xy * relative[:, 2:3])[good].reshape(-1)
    usable = np.isfinite(A) & np.isfinite(B) & (np.abs(A) > 0.002)
    A, B = A[usable], B[usable]
    if len(A) < 4:
        return empty, np.nan, np.nan, np.nan
    depth = float(np.dot(A, B) / max(np.dot(A, A), 1e-12))
    for _ in range(4):
        residual = B - A * depth
        spread = max(0.003, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = np.minimum(1, 1.5 * spread / np.maximum(np.abs(residual), 1e-9))
        depth = float(np.dot(A * weights, B) / max(np.dot(A * weights, A), 1e-12))
    if not 0.12 <= depth <= 6:
        return empty, np.nan, np.nan, np.nan
    lifted = relative + np.r_[anchor * depth, depth]
    lifted[~good] = np.nan
    error = np.linalg.norm(model.project(lifted)[good] - pixels[good, :2], axis=1)
    sigma = max(depth * 0.20, spread / max(np.sqrt(np.mean(A * A)), 1e-6))
    return lifted, depth, float(sigma), float(np.median(error))


def ray_plane(pixels: np.ndarray, model, camera_pose: np.ndarray, plane: np.ndarray):
    rays = model.rays(pixels) @ camera_pose[:3, :3].T
    origin = camera_pose[:3, 3]
    denominator = rays @ plane[:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -(origin @ plane[:3] + plane[3]) / denominator
        result = origin + rays * distance[..., None]
    valid = np.isfinite(distance) & (distance > 0) & (distance < 10)
    return np.where(valid[..., None], result, np.nan)


def triangulate(
    first_pixels: np.ndarray, second_pixels: np.ndarray,
    first_model, second_model, first_pose: np.ndarray, second_pose: np.ndarray,
    max_reprojection_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Only call with independently established cross-view correspondences."""
    a, b = first_model.normalized(first_pixels), second_model.normalized(second_pixels)
    A, B = np.linalg.inv(first_pose)[:3], np.linalg.inv(second_pose)[:3]
    homogeneous = cv2.triangulatePoints(A, B, a.T, b.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        points = (homogeneous[:3] / homogeneous[3]).T
    ac = transform_points(points, np.linalg.inv(first_pose))
    bc = transform_points(points, np.linalg.inv(second_pose))
    error = np.maximum(
        np.linalg.norm(first_model.project(ac) - first_pixels, axis=1),
        np.linalg.norm(second_model.project(bc) - second_pixels, axis=1),
    )
    da = points - first_pose[:3, 3]
    db = points - second_pose[:3, 3]
    cosine = np.sum(da * db, axis=1) / np.maximum(
        np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1), 1e-12
    )
    valid = (
        np.isfinite(points).all(axis=1) & (ac[:, 2] > 0) & (bc[:, 2] > 0)
        & (error <= max_reprojection_px)
        & (np.arccos(np.clip(cosine, -1, 1)) > np.deg2rad(1))
    )
    return np.where(valid[:, None], points, np.nan), valid
