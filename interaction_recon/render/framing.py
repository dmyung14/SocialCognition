"""Render-only camera/lighting changes. No solver state or scene dynamics change."""
from itertools import product

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FRAME_VERSION = "rollout-bounds-builder-three-quarter-v1"


def output_size(config) -> tuple[int, int]:
    """Upgrade the legacy default; retain explicitly smaller diagnostic sizes."""
    if (config.width, config.height) == (640, 480):
        return 960, 720
    return config.width, config.height


def corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray(list(product(*zip(low, high))), dtype=float)


def geometry_bounds(model, data) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        name = model.geom(geom).name or ""
        if name == "table" or body.startswith(("guider_", "builder_", "block_")):
            selected.append(geom)
    if not selected:
        raise ValueError("Cannot frame a scene without table/actor/block geometry")
    selected = np.asarray(selected, int)
    # Conservative, orientation-independent primitive bounds.
    radius = model.geom_rbound[selected, None]
    positions = data.geom_xpos[selected]
    return np.min(positions - radius, axis=0), np.max(positions + radius, axis=0)


def table_corners(model, data) -> np.ndarray:
    geom = model.geom("table").id
    size = model.geom_size[geom]
    local = corners(-size, size)
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def configure_view(
    model, low: np.ndarray, high: np.ndarray, table: np.ndarray,
    width: int, height: int,
) -> dict:
    if not np.isfinite(np.r_[low, high]).all():
        raise ValueError("Nonfinite scene bounds cannot be framed")
    center = (low + high) / 2
    direction = np.array([0.65, -1.0, 0.72])
    direction /= np.linalg.norm(direction)
    right = np.cross([0.0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    axes = np.column_stack((right, up, direction))
    fovy = 42.0
    tangent_y = np.tan(np.deg2rad(fovy / 2))
    tangent_x = tangent_y * width / height
    local = (corners(low, high) - center) @ axes

    # Fit all actor bounds with room for the top label and image margins.
    fit_distance = max(
        float(np.max(local[:, 2] + np.abs(local[:, 0]) / (0.92 * tangent_x))),
        float(np.max(local[:, 2] + np.abs(local[:, 1]) / (0.78 * tangent_y))),
    )
    table_local = (table - center) @ axes

    def table_fraction(distance: float) -> float:
        depth = distance - table_local[:, 2]
        if np.any(depth <= 0):
            return np.inf
        x = table_local[:, 0] / (depth * tangent_x)
        return float(np.ptp(x) / 2)

    lower = float(np.max(table_local[:, 2]) + 1e-5)
    upper = max(1.0, fit_distance)
    while table_fraction(upper) > 0.70:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if table_fraction(middle) > 0.70:
            lower = middle
        else:
            upper = middle
    distance = max(upper, fit_distance, 0.1)
    camera = model.camera("overview").id
    model.cam_pos[camera] = center + direction * distance
    model.cam_quat[camera] = Rotation.from_matrix(axes).as_quat()[[3, 0, 1, 2]]
    model.cam_fovy[camera] = fovy
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.55
    model.vis.headlight.specular[:] = 0.08
    if model.nlight:
        model.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        model.light_dir[0] = [-0.35, 0.45, -1]
        model.light_diffuse[0] = [0.65, 0.65, 0.65]
        model.light_ambient[0] = [0.12, 0.12, 0.12]
    model.geom_rgba[model.geom("table").id] = [0.32, 0.27, 0.22, 1]
    for geom in range(model.ngeom):
        if (model.body(int(model.geom_bodyid[geom])).name or "").startswith("block_"):
            model.geom_rgba[geom] = [0.83, 0.89, 0.96, 1]
    return {
        "version": FRAME_VERSION,
        "bounds_min_m": low.tolist(),
        "bounds_max_m": high.tolist(),
        "camera_position_m": model.cam_pos[camera].tolist(),
        "table_width_fraction": table_fraction(distance),
        "requested_table_width_fraction": 0.70,
        "policy": "70% table width unless complete actor bounds require a wider view",
    }
