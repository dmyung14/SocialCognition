"""Rendering-only replay; this module never runs or controls a physics rollout."""
from pathlib import Path

import cv2
import mujoco
import numpy as np

from interaction_recon.render.framing import (
    configure_view, geometry_bounds, output_size, table_corners,
)
from interaction_recon.render.video import write_mp4
from interaction_recon.simulation.physics_scene import PhysicsConfig


def render_physics(
    output: Path, xml: str, rollout: dict,
    config: PhysicsConfig = PhysicsConfig(),
) -> int:
    model = mujoco.MjModel.from_xml_string(xml)
    replay = mujoco.MjData(model)
    times = rollout["timestamps"]
    width, height = output_size(config)
    if not np.isfinite(rollout["qpos"]).all():
        raise ValueError("Cannot render a nonfinite physics trajectory")
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    # Inspect every recorded substep, not just observation or movie frames.
    for qpos in rollout["qpos"]:
        replay.qpos[:] = qpos
        mujoco.mj_kinematics(model, replay)
        a, b = geometry_bounds(model, replay)
        low, high = np.minimum(low, a), np.maximum(high, b)
    configure_view(model, low, high, table_corners(model, replay), width, height)
    duration = max(1 / 30, float(times[-1] - times[0]))
    count = max(1, int(np.ceil(duration * 30 - 1e-8)))

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        def frames():
            for index in range(count):
                stamp = float(times[0] + index / 30)
                k = int(np.clip(
                    np.searchsorted(times, stamp, side="right") - 1, 0, len(times) - 1
                ))
                replay.qpos[:] = rollout["qpos"][k]
                replay.qvel[:] = rollout["qvel"][k]
                replay.ctrl[:] = rollout["ctrl"][k]
                replay.mocap_pos[:] = rollout["mocap_pos"][k]
                replay.mocap_quat[:] = rollout["mocap_quat"][k]
                mujoco.mj_kinematics(model, replay)
                mujoco.mj_camlight(model, replay)
                renderer.update_scene(replay, camera="overview")
                image = renderer.render().copy()
                cv2.rectangle(image, (0, 0), (width, 54), (15, 15, 15), -1)
                cv2.putText(
                    image, "PHYSICS: selected contact rollout", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 110), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    image, f"t={stamp:.3f}s | contact-driven blocks | fusion unverified",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (235, 235, 235), 1, cv2.LINE_AA,
                )
                yield image
        return write_mp4(output, frames(), width, height, 30)
