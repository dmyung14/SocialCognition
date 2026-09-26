"""OpenCV moving-camera baseline. All intrinsics are explicitly uncertain priors."""
from dataclasses import dataclass
import json

import cv2
import numpy as np


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


@dataclass
class CameraModel:
    K: np.ndarray
    model: str
    circle: np.ndarray
    fov_deg: float
    fov_sigma_deg: float
    circle_confidence: float

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, float)
        xy = (pixels - self.K[:2, 2]) / [self.K[0, 0], self.K[1, 1]]
        if self.model == "equidistant":
            theta = np.linalg.norm(xy, axis=-1)
            factor = np.sin(theta) / np.maximum(theta, 1e-12)
            rays = np.concatenate(
                (xy * factor[..., None], np.cos(theta)[..., None]), axis=-1
            )
        else:
            rays = np.concatenate((xy, np.ones(xy.shape[:-1] + (1,))), axis=-1)
        return rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)

    def normalized(self, pixels: np.ndarray) -> np.ndarray:
        rays = self.rays(pixels)
        return rays[..., :2] / rays[..., 2:3]

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float)
        if self.model == "equidistant":
            radius = np.linalg.norm(points[..., :2], axis=-1)
            theta = np.arctan2(radius, points[..., 2])
            xy = points[..., :2] * (
                theta / np.maximum(radius, 1e-12)
            )[..., None]
        else:
            xy = points[..., :2] / points[..., 2:3]
        result = xy * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        return np.where((points[..., 2] > 1e-6)[..., None], result, np.nan)

    def arrays(self) -> dict:
        return {
            "K": self.K,
            "model": np.array(self.model),
            "image_circle": self.circle,
            "fov_prior_deg": np.array(self.fov_deg),
            "fov_sigma_deg": np.array(self.fov_sigma_deg),
            "circle_confidence": np.array(self.circle_confidence),
            "intrinsics_are_calibration": np.array(False),
            "intrinsics_relative_sigma": np.array(0.20),
        }

    @classmethod
    def from_arrays(cls, arrays: dict):
        return cls(
            arrays["K"], str(arrays["model"].item()), arrays["image_circle"],
            float(arrays["fov_prior_deg"]), float(arrays["fov_sigma_deg"]),
            float(arrays["circle_confidence"]),
        )


def estimate_intrinsics(frames: np.ndarray, role: str) -> CameraModel:
    height, width = frames.shape[1:3]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    radius = min(width, height) / 2
    circle_confidence = 0.0
    if role == "guider":
        indices = np.linspace(0, len(frames) - 1, min(7, len(frames))).astype(int)
        image = np.median(frames[indices].max(axis=-1), axis=0)
        binary = (image > 20).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            dark_corners = np.mean(image[
                [0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]
            ] < 25)
            if area > width * height * 0.25 and circularity > 0.65 and dark_corners >= 0.75:
                center, radius = np.array([cx, cy]), float(r)
                circle_confidence = float(min(1, circularity))
        # Circular-image angular diameter prior, not a manufacturer calibration.
        fov, sigma = 110.0, 15.0
        focal = radius / np.deg2rad(fov / 2)
        model = "equidistant"
    else:
        fov, sigma = 105.0, 10.0
        focal = np.hypot(width, height) / (2 * np.tan(np.deg2rad(fov / 2)))
        model = "pinhole"
    K = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1.0]])
    return CameraModel(
        K, model, np.r_[center, radius], fov, sigma, circle_confidence
    )


def background_mask(rgb: np.ndarray, observations: dict, index: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    mask = (rgb.max(axis=-1) > 20).astype(np.uint8) * 255
    excluded = np.zeros((height, width), np.uint8)
    groups = [observations["pose"][index], *observations["hands"][index]]
    for points in groups:
        valid = np.isfinite(points[:, :2]).all(axis=1) & (points[:, 3] >= 0.25)
        if np.count_nonzero(valid) >= 3:
            hull = cv2.convexHull(np.round(points[valid, :2]).astype(np.int32))
            cv2.fillConvexPoly(excluded, hull, 255)
    for corners, observed in zip(
        observations["block_corners"][index], observations["block_observed"][index]
    ):
        if observed and np.isfinite(corners).all():
            cv2.fillConvexPoly(excluded, np.round(corners).astype(np.int32), 255)
    radius = max(3, round(min(height, width) * 0.025))
    excluded = cv2.dilate(
        excluded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    )
    mask[excluded != 0] = 0
    return mask


def match_features(
    first: np.ndarray, second: np.ndarray,
    first_mask: np.ndarray | None = None, second_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.ORB_create(nfeatures=1400, fastThreshold=12)
    a, da = detector.detectAndCompute(first, first_mask)
    b, db = detector.detectAndCompute(second, second_mask)
    if da is None or db is None or min(len(a), len(b)) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(da, db, k=2)
    reverse = matcher.match(db, da)
    back = {m.queryIdx: m.trainIdx for m in reverse}
    matches = [
        pair[0] for pair in forward
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
        and back.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    return (
        np.asarray([a[m.queryIdx].pt for m in matches], float).reshape(-1, 2),
        np.asarray([b[m.trainIdx].pt for m in matches], float).reshape(-1, 2),
    )


def relative_motion(
    first: np.ndarray, second: np.ndarray, model: CameraModel,
    depth_prior_m: float = 1.2,
) -> dict:
    """Return camera2_from_camera1. Translation scale is a weak depth prior."""
    result = {
        "R": np.eye(3), "t": np.zeros(3), "confidence": 0.0,
        "translation_confidence": 0.0, "method": "untracked",
        "scale_m_per_unit": np.nan, "scale_sigma": np.nan,
        "inlier_fraction": 0.0,
    }
    if len(first) < 8:
        return result
    a, b = model.normalized(first), model.normalized(second)
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[good], b[good]
    if len(a) < 8:
        return result
    threshold = 2.0 / model.K[0, 0]
    H, hm = cv2.findHomography(a, b, cv2.RANSAC, threshold, maxIters=1000)
    hratio = float(np.mean(hm != 0)) if hm is not None else 0.0
    if H is not None:
        H = H / np.cbrt(max(np.linalg.det(H), 1e-12))
        rotation = nearest_rotation(H)
        anisotropy = np.linalg.norm(H - rotation)
        if hratio > 0.65 and anisotropy < 0.035:
            result.update(
                R=rotation, confidence=0.6 * hratio,
                method="homography_rotation", inlier_fraction=hratio,
            )
            return result
    E, em = cv2.findEssentialMat(
        a, b, np.eye(3), method=cv2.RANSAC, prob=0.999,
        threshold=threshold, maxIters=1000,
    )
    if E is not None and em is not None:
        for candidate in np.asarray(E).reshape(-1, 3, 3):
            count, rotation, direction, mask = cv2.recoverPose(
                candidate, a, b, np.eye(3), mask=em.copy()
            )
            if count < max(12, len(a) * 0.45):
                continue
            selected = mask.ravel() != 0
            P1 = np.c_[np.eye(3), np.zeros(3)]
            P2 = np.c_[rotation, direction]
            homogeneous = cv2.triangulatePoints(P1, P2, a[selected].T, b[selected].T)
            xyz = (homogeneous[:3] / homogeneous[3]).T
            positive = xyz[:, 2][np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)]
            if len(positive) < 8:
                continue
            scale = float(depth_prior_m / np.median(positive))
            if not 1e-5 < scale < depth_prior_m * 0.5:
                continue
            fraction = count / len(a)
            result.update(
                R=rotation, t=direction.ravel() * scale,
                confidence=min(0.75, fraction),
                translation_confidence=0.25 * fraction,
                method="essential_depth_prior", scale_m_per_unit=scale,
                scale_sigma=scale * 0.75, inlier_fraction=fraction,
            )
            return result
    if H is not None and hratio > 0.4:
        result.update(
            R=nearest_rotation(H), confidence=0.2 * hratio,
            method="homography_ambiguous_rotation_only", inlier_fraction=hratio,
        )
    return result


def _table_membership(pixels: np.ndarray, mask: np.ndarray, width: int, height: int):
    if not len(pixels):
        return np.zeros(0, bool)
    ij = np.floor(pixels * [mask.shape[1] / width, mask.shape[0] / height]).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, mask.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, mask.shape[0] - 1)
    return mask[ij[:, 1], ij[:, 0]] != 0


def estimate_camera(sequence, observations: dict, role: str, depth_prior_m: float) -> dict:
    model = estimate_intrinsics(sequence.rgb_frames, role)
    count = len(sequence.rgb_frames)
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    confidence = np.zeros(count)
    translation_confidence = np.zeros(count)
    method = np.full(count, "untracked", dtype="<U48")
    scale = np.full(count, np.nan)
    scale_sigma = np.full(count, np.nan)
    normals = np.full((count, 3), np.nan)
    normal_confidence = np.zeros(count)
    lab = np.full(observations["block_ids"].shape + (3,), np.nan)
    previous = previous_mask = None

    for i, rgb in enumerate(sequence.rgb_frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = background_mask(rgb, observations, i)
        for slot, corners in enumerate(observations["block_corners"][i]):
            if not observations["block_observed"][i, slot] or not np.isfinite(corners).all():
                continue
            region = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(region, corners.astype(np.int32), 255)
            selected = region != 0
            if selected.any():
                pixels = rgb[selected].reshape(-1, 1, 3)
                lab[i, slot] = np.median(
                    cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3), axis=0
                )
        if i == 0:
            method[i] = "gauge_origin"
            previous, previous_mask = gray, mask
            continue
        poses[i] = poses[i - 1]
        if observations["source_frame_indices"][i] == observations["source_frame_indices"][i - 1]:
            confidence[i] = confidence[i - 1]
            translation_confidence[i] = translation_confidence[i - 1]
            method[i] = "repeated_source_frame"
            previous, previous_mask = gray, mask
            continue
        a, b = match_features(previous, gray, previous_mask, mask)
        relative = relative_motion(a, b, model, depth_prior_m)
        R, t = relative["R"], relative["t"]
        inverse = np.eye(4)
        inverse[:3, :3] = R.T
        inverse[:3, 3] = -R.T @ t
        poses[i] = poses[i - 1] @ inverse
        confidence[i] = relative["confidence"]
        translation_confidence[i] = relative["translation_confidence"]
        method[i] = relative["method"]
        scale[i], scale_sigma[i] = relative["scale_m_per_unit"], relative["scale_sigma"]

        selected = (
            _table_membership(a, observations["table_masks"][i - 1],
                              sequence.width, sequence.height)
            & _table_membership(b, observations["table_masks"][i],
                                sequence.width, sequence.height)
        )
        if selected.sum() >= 12:
            H, inliers = cv2.findHomography(
                model.normalized(a[selected]), model.normalized(b[selected]),
                cv2.RANSAC, 2 / model.K[0, 0],
            )
            if H is not None and inliers is not None:
                number, rotations, translations, candidates = cv2.decomposeHomographyMat(
                    H, np.eye(3)
                )
                choices = []
                for j in range(number):
                    if np.linalg.norm(translations[j]) < 0.005:
                        continue
                    normal = candidates[j].ravel()
                    if normal[1] > 0:
                        normal = -normal
                    if normal[2] > -0.05:
                        continue
                    score = np.linalg.norm(rotations[j] - R)
                    choices.append((score, normal))
                if choices:
                    score, normal = min(choices, key=lambda entry: entry[0])
                    if score < 0.3:
                        normals[i] = poses[i - 1, :3, :3] @ normal
                        normal_confidence[i] = float(np.mean(inliers)) * 0.5
        previous, previous_mask = gray, mask

    return {
        **model.arrays(),
        "timestamps": observations["timestamps_s"],
        "T_initial_from_camera": poses,
        "pose_confidence": confidence,
        "translation_confidence": translation_confidence,
        "pose_observed": confidence > 0,
        "pose_method": method,
        "translation_scale_m_per_unit": scale,
        "translation_scale_sigma": scale_sigma,
        "table_normal_initial": normals,
        "table_normal_confidence": normal_confidence,
        "block_lab": lab,
        "metadata_json": np.array(json.dumps({
            "pose_convention": "T_initial_from_camera; OpenCV camera axes",
            "intrinsics": "uncertain FOV and vignette priors, NOT calibration",
            "translation": (
                "essential-matrix direction scaled by learned human metric depth; "
                "background depth equality is an uncertain prior"
            ),
            "failure": "held pose with zero confidence; no fabricated observed motion",
            "homography": "rotation-only fallback; planar translation can remain unobservable",
        })),
    }
