from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("left", "right")
HAND_PREFIXES = tuple(
    f"{actor}_{side}" for actor in ("guider", "builder") for side in SIDES
)


@dataclass(frozen=True)
class RetargetConfig:
    iterations: int = 32
    damping: float = 0.012
    regularizer: float = 0.00001
    finger_regularizer: float = 0.03
    orientation_weight: float = 0.08
    hold_s: float = 0.5
    width: int = 640
    height: int = 480
    render_fps: int = 30

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("IK iterations must be positive")
        for name in (
            "damping", "regularizer", "finger_regularizer",
            "orientation_weight", "hold_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.hold_s <= 0.5:
            raise ValueError("Retarget holds must not exceed 0.5 seconds")
        if self.width < 64 or self.height < 64 or self.width % 2 or self.height % 2:
            raise ValueError("Render dimensions must be even and at least 64")
        if self.render_fps != 30:
            raise ValueError("The M4 reference render uses 30 FPS")


def quaternion(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def rotation(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def valid_pose(pose: np.ndarray) -> bool:
    return bool(
        np.shape(pose) == (7,) and np.isfinite(pose).all()
        and np.linalg.norm(pose[3:]) > 1e-8
    )


def palm_frame(points: np.ndarray, side: str) -> np.ndarray | None:
    """Local +y points wrist->middle MCP; +x is mirrored across the two hands."""
    p = np.asarray(points, float)
    if p.shape != (21, 3) or not np.isfinite(p[[0, 5, 9, 17]]).all():
        return None
    y = p[9] - p[0]
    if np.linalg.norm(y) < 1e-5:
        return None
    y /= np.linalg.norm(y)
    sign = -1 if side == "left" else 1
    x = sign * (p[5] - p[17])
    x -= y * np.dot(x, y)
    if np.linalg.norm(x) < 1e-5:
        return None
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def axis_frame(direction: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    y = np.asarray(direction, float)
    if not np.isfinite(y).all() or np.linalg.norm(y) < 1e-6:
        raise ValueError("Cannot orient a zero-length forearm")
    y = y / np.linalg.norm(y)
    x = np.asarray(x_hint, float) - y * np.dot(x_hint, y)
    if np.linalg.norm(x) < 1e-5:
        basis = np.eye(3)[np.argmin(np.abs(y))]
        x = basis - y * np.dot(basis, y)
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def usable_points(observations: dict, name: str) -> np.ndarray:
    points = np.asarray(observations[name], float).copy()
    confidence = observations.get(f"{name}_confidence")
    if confidence is not None:
        points[np.asarray(confidence) <= 0] = np.nan
    return points


def _measure(values, prior, low, high) -> dict:
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values) & (values > 1e-5)]
    median = float(np.median(values)) if len(values) else float(prior)
    value = float(np.clip(median, low, high))
    return {
        "value_m": value,
        "raw_median_m": median if len(values) else None,
        "samples": len(values),
        "fallback_prior": not bool(len(values)),
        "clipped_to_adult_prior": bool(len(values) and value != median),
        "confidence": 0.25 if len(values) else 0.0,
    }


def estimate_anthropometry(observations: dict) -> dict:
    """Sequence-wide metric medians, not per-frame skeleton rescaling."""
    measurements = {}

    def measure(name, values, prior, low, high):
        result = _measure(values, prior, low, high)
        measurements[name] = result
        return result["value_m"]

    arms = usable_points(observations, "guider_arm_keypoints")
    torso = usable_points(observations, "guider_torso_pose")
    head = usable_points(observations, "guider_head_pose")
    width = measure(
        "shoulder_width", np.linalg.norm(arms[:, 0, 0] - arms[:, 1, 0], axis=-1),
        0.38, 0.28, 0.50,
    )
    shoulder_center = np.mean(arms[:, :, 0], axis=1)
    shoulder_height = measure(
        "torso_to_shoulders",
        np.linalg.norm(shoulder_center - torso[:, :3], axis=-1),
        0.20, 0.12, 0.30,
    )
    neck_length = measure(
        "shoulders_to_head",
        np.linalg.norm(head[:, :3] - shoulder_center, axis=-1),
        0.16, 0.10, 0.23,
    )
    result = {
        "shoulder_width": width,
        "shoulder_height": shoulder_height,
        "neck_length": max(0.06, neck_length - 0.025),
        "measurements": measurements,
        "hands": {},
        "upper_arm": {},
        "forearm": {},
    }
    forearms = usable_points(observations, "builder_forearm_keypoints")
    for side_index, side in enumerate(SIDES):
        prefix = f"guider_{side}"
        result["upper_arm"][prefix] = measure(
            f"{prefix}_upper_arm",
            np.linalg.norm(arms[:, side_index, 1] - arms[:, side_index, 0], axis=-1),
            0.29, 0.22, 0.38,
        )
        result["forearm"][prefix] = measure(
            f"{prefix}_forearm",
            np.linalg.norm(arms[:, side_index, 2] - arms[:, side_index, 1], axis=-1),
            0.25, 0.20, 0.34,
        )
        prefix = f"builder_{side}"
        result["forearm"][prefix] = measure(
            f"{prefix}_forearm",
            np.linalg.norm(forearms[:, side_index, 1] - forearms[:, side_index, 0], axis=-1),
            0.25, 0.20, 0.34,
        )

    for prefix in HAND_PREFIXES:
        side = prefix.split("_")[1]
        sign = -1 if side == "left" else 1
        points = usable_points(observations, f"{prefix}_hand_21")
        length = measure(
            f"{prefix}_palm_length", np.linalg.norm(points[:, 9] - points[:, 0], axis=-1),
            0.085, 0.065, 0.11,
        )
        width = measure(
            f"{prefix}_palm_width", np.linalg.norm(points[:, 5] - points[:, 17], axis=-1),
            0.075, 0.055, 0.10,
        )
        anchors = np.array([
            [sign * width * 0.48, length * 0.30, 0],
            [sign * width * 0.40, length * 0.88, 0],
            [0, length, 0],
            [-sign * width * 0.24, length * 0.91, 0],
            [-sign * width * 0.48, length * 0.75, 0],
        ])
        local = []
        for p in points:
            frame = palm_frame(p, side)
            if frame is not None:
                local.append((p - p[0]) @ frame)
        anchor_prior = []
        for f, start in enumerate((1, 5, 9, 13, 17)):
            candidates = np.asarray([p[start] for p in local]).reshape(-1, 3)
            candidates = candidates[np.isfinite(candidates).all(axis=-1)]
            anchor_prior.append(not bool(len(candidates)))
            if len(candidates):
                anchors[f] = np.clip(
                    np.median(candidates, axis=0),
                    [-width * 0.7, length * 0.15, -0.01],
                    [width * 0.7, length * 1.15, 0.01],
                )
        lengths = np.zeros((5, 3))
        for f, finger in enumerate(FINGERS):
            start = 1 + 4 * f
            priors = (0.032, 0.025, 0.020) if f == 0 else (0.040, 0.025, 0.019)
            for k in range(3):
                lengths[f, k] = measure(
                    f"{prefix}_{finger}_phalanx_{k}",
                    np.linalg.norm(points[:, start + k + 1] - points[:, start + k], axis=-1),
                    priors[k],
                    (0.020, 0.014, 0.010)[k],
                    (0.055, 0.038, 0.032)[k],
                )
        result["hands"][prefix] = {
            "palm_length": length, "palm_width": width,
            "anchors": anchors.tolist(), "lengths": lengths.tolist(),
            "anchor_fallback_prior": anchor_prior,
            "thumb_yaw": -sign * 0.7,
        }
    return result


def finger_limits(finger: str) -> np.ndarray:
    if finger == "thumb":
        return np.array([[-0.8, 0.8], [-0.35, 1.15], [-0.15, 1.5], [-0.15, 1.5]])
    return np.array([[-0.45, 0.45], [-0.20, 1.65], [0, 1.95], [0, 1.55]])


def finger_joint_names(prefix: str, finger: str) -> list[str]:
    return [
        f"{prefix}_{finger}_{suffix}"
        for suffix in ("abduction", "flexion_0", "flexion_1", "flexion_2")
    ]
