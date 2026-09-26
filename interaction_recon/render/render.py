"""KINEMATIC visualization only. Never import this module into the solver."""
from pathlib import Path

import cv2
import mujoco
import numpy as np

from interaction_recon.render.framing import (
    configure_view, geometry_bounds, output_size, table_corners,
)
from interaction_recon.render.video import write_mp4
from interaction_recon.retarget.models import RetargetConfig, valid_pose


def human_qpos_at(model, targets: dict, time: float) -> np.ndarray:
    times = targets["timestamps"]
    left = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 1))
    right = min(left + 1, len(times) - 1)
    fraction = (
        float(np.clip((time - times[left]) / (times[right] - times[left]), 0, 1))
        if right != left else 0.0
    )
    first, second = model.qpos0.copy(), model.qpos0.copy()
    for actor in ("guider", "builder"):
        indices = targets[f"{actor}_qpos_indices"]
        first[indices] = targets[f"{actor}_qpos_targets"][left]
        second[indices] = targets[f"{actor}_qpos_targets"][right]
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1.0, first, second)
    mujoco.mj_integratePos(model, first, velocity, fraction)
    return first


def set_kinematic_block_poses(
    model: mujoco.MjModel, data: mujoco.MjData, observations: dict, index: int,
) -> None:
    """Prescribe blocks FOR VISUALIZATION ONLY; never call during a rollout."""
    for j, identity in enumerate(observations["block_ids"]):
        name = f"block_{int(identity)}"
        geom = model.geom(f"{name}_geom").id
        pose = observations["block_poses"][index, j]
        confidence = observations["block_poses_confidence"][index, j]
        valid = valid_pose(pose) and confidence > 0
        model.geom_rgba[geom, 3] = 1.0 if valid else 0.0
        if valid:
            address = model.joint(f"{name}_free").qposadr[0]
            data.qpos[address:address + 3] = pose[:3]
            data.qpos[address + 3:address + 7] = pose[3:] / np.linalg.norm(pose[3:])


def kinematic_frame(
    model, data, renderer, targets: dict, observations: dict, time: float,
) -> np.ndarray:
    data.qpos[:] = human_qpos_at(model, targets, time)
    index = int(np.clip(
        np.searchsorted(observations["timestamps"], time, side="right") - 1,
        0, len(observations["timestamps"]) - 1,
    ))
    set_kinematic_block_poses(model, data, observations, index)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="overview")
    frame = renderer.render().copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 65), (15, 15, 15), -1)
    cv2.putText(
        frame, "KINEMATIC REFERENCE (not physics)", (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 110), 1, cv2.LINE_AA,
    )
    target_index = int(np.clip(
        np.searchsorted(targets["timestamps"], time, side="right") - 1,
        0, len(targets["timestamps"]) - 1,
    ))
    valid = float(np.mean(targets["validity_mask"][target_index]))
    cv2.putText(
        frame, f"t={time:.3f}s | target coverage={valid:.0%} | missing chains hold/rest",
        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA,
    )
    return frame


def render_kinematic_reference(
    output: Path, xml: str, targets: dict, observations: dict,
    config: RetargetConfig = RetargetConfig(),
) -> int:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    width, height = output_size(config)
    times = targets["timestamps"]
    tail = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    count = max(1, int(np.ceil((times[-1] + tail - times[0]) * 30 - 1e-8)))
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for stamp in np.unique(np.r_[times, times[0] + np.arange(count) / 30]):
        data.qpos[:] = human_qpos_at(model, targets, float(stamp))
        index = int(np.clip(
            np.searchsorted(observations["timestamps"], stamp, side="right") - 1,
            0, len(observations["timestamps"]) - 1,
        ))
        set_kinematic_block_poses(model, data, observations, index)
        mujoco.mj_kinematics(model, data)
        a, b = geometry_bounds(model, data)
        low, high = np.minimum(low, a), np.maximum(high, b)
    configure_view(model, low, high, table_corners(model, data), width, height)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        frames = (
            kinematic_frame(
                model, data, renderer, targets, observations, float(times[0] + i / 30)
            )
            for i in range(count)
        )
        return write_mp4(output, frames, width, height, 30)
