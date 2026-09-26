import numpy as np
from scipy.spatial.transform import Rotation

from interaction_recon.retarget.models import FINGERS, finger_limits, palm_frame


def hand_angles(
    landmarks: np.ndarray,
    side: str,
    thumb_yaw: float,
    previous: np.ndarray | None = None,
    regularizer: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Return five (spread, proximal, middle, distal) angle vectors.

    MCP spread is measured in the palm frame. Flexion is signed around the
    transported flexion axis, not an unsigned angle that would fold extensions.
    A missing finger returns NaNs; the timeline manager decides hold/rest.
    """
    result = np.full((5, 4), np.nan)
    valid = np.zeros(5, bool)
    frame = palm_frame(landmarks, side)
    if frame is None:
        return result, valid
    points = (np.asarray(landmarks, float) - landmarks[0]) @ frame
    for f, finger in enumerate(FINGERS):
        start = 1 + 4 * f
        bones = np.diff(points[start:start + 4], axis=0)
        norms = np.linalg.norm(bones, axis=1)
        if not np.isfinite(bones).all() or np.any(norms < 1e-6):
            continue
        bones /= norms[:, None]
        yaw = np.arctan2(-bones[0, 0], bones[0, 1])
        spread = yaw - (thumb_yaw if f == 0 else 0)
        spread = (spread + np.pi) % (2 * np.pi) - np.pi
        proximal = np.arctan2(-bones[0, 2], np.linalg.norm(bones[0, :2]))
        axis = Rotation.from_euler("z", yaw).apply([1.0, 0, 0])
        flexion = [
            np.arctan2(-np.dot(axis, np.cross(a, b)), np.dot(a, b))
            for a, b in zip(bones, bones[1:])
        ]
        angles = np.array([spread, proximal, *flexion])
        if previous is not None and np.isfinite(previous[f]).all():
            angles = (angles + regularizer * previous[f]) / (1 + regularizer)
        limits = finger_limits(finger)
        result[f] = np.clip(angles, limits[:, 0], limits[:, 1])
        valid[f] = True
    return result, valid
