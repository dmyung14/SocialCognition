import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from interaction_recon.retarget.models import (
    FINGERS, HAND_PREFIXES, SIDES, estimate_anthropometry,
    finger_joint_names, finger_limits, usable_points, valid_pose,
)


def _text(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(f"{float(x):.10g}" for x in np.asarray(value).ravel())
    return str(value)


def _add(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: _text(v) for k, v in attributes.items()})


def _joint(body, name, axis, limits):
    _add(body, "joint", name=name, type="hinge", axis=axis, range=limits,
         limited="true", damping=0.1, armature=0.0002)


def _site(body, name, pos=(0, 0, 0)):
    _add(body, "site", name=name, pos=pos, size=0.003, rgba="0 0 0 0")


def _capsule(body, length, radius, collision, color):
    _add(
        body, "geom", type="capsule", fromto=[0, 0, 0, 0, length, 0],
        size=radius, contype=collision[0], conaffinity=collision[1], rgba=color,
    )


def _hand(parent, prefix, dimensions, collision, color):
    palm = _add(parent, "body", name=f"{prefix}_palm")
    for axis, name, limits in zip(
        np.eye(3), ("x", "y", "z"),
        ((-1.3, 1.3), (-1.4, 1.4), (-0.65, 0.65)),
    ):
        _joint(palm, f"{prefix}_wrist_{name}", axis, limits)
    _site(palm, f"{prefix}_wrist")
    length, width = dimensions["palm_length"], dimensions["palm_width"]
    _add(
        palm, "geom", type="box", size=[width / 2, length / 2, 0.012],
        pos=[0, length / 2, 0], contype=collision[0],
        conaffinity=collision[1], rgba=color,
    )
    _site(palm, f"{prefix}_landmark_0")
    for f, finger in enumerate(FINGERS):
        lengths = dimensions["lengths"][f]
        limits = finger_limits(finger)
        names = finger_joint_names(prefix, finger)
        yaw = dimensions["thumb_yaw"] if f == 0 else 0
        base = _add(
            palm, "body", name=f"{prefix}_{finger}_0",
            pos=dimensions["anchors"][f], euler=[0, 0, yaw],
        )
        _joint(base, names[0], [0, 0, 1], limits[0])
        _joint(base, names[1], [-1, 0, 0], limits[1])
        _site(base, f"{prefix}_landmark_{1 + 4 * f}")
        current = base
        for k in range(3):
            if k:
                current = _add(
                    current, "body", name=f"{prefix}_{finger}_{k}",
                    pos=[0, lengths[k - 1], 0],
                )
                _joint(current, names[k + 1], [-1, 0, 0], limits[k + 1])
            _capsule(current, lengths[k], 0.007 if f == 0 else 0.006, collision, color)
            _site(current, f"{prefix}_landmark_{2 + 4 * f + k}", [0, lengths[k], 0])


def _camera(world, name, position, target, fovy):
    z = np.asarray(position, float) - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    _add(world, "camera", name=name, pos=position, xyaxes=np.r_[x, y], fovy=fovy)


def build_scene(observations: dict, output: Path | None = None) -> tuple[str, dict]:
    """Generate MJCF; block joints have no actuators, constraints, or mocap parents."""
    anthropometry = estimate_anthropometry(observations)
    extent = np.asarray(observations["table_extent_xy"], float)
    if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("table_extent_xy must contain two positive metric extents")
    root = ET.Element("mujoco", model="interaction_kinematic_reference")
    _add(root, "compiler", angle="radian", autolimits="true")
    _add(root, "option", timestep=0.002, gravity="0 0 -9.81", integrator="implicitfast")
    visual = _add(root, "visual")
    _add(visual, "global", offwidth=1920, offheight=1080)
    _add(visual, "quality", shadowsize=1024, offsamples=0)
    default = _add(root, "default")
    _add(
        default, "geom", density=650, friction="0.9 0.02 0.002",
        condim=4, solref="0.01 1", solimp="0.9 0.95 0.001",
    )
    world = _add(root, "worldbody")
    _add(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    _add(world, "light", pos="1 2 2", dir="-0.2 -0.2 -1", diffuse="0.5 0.5 0.5")
    # Collision bits: table=1, blocks=2, builder=4. Guider is visual-only.
    _add(
        world, "geom", name="table", type="box",
        size=[extent[0] / 2, extent[1] / 2, 0.035], pos=[0, 0, -0.035],
        contype=1, conaffinity=6, rgba="0.30 0.23 0.16 1",
    )
    neutral_root = np.array([0, extent[1] / 2 + 0.14, 0.14, 1, 0, 0, 0], float)
    poses = usable_points(observations, "guider_torso_pose")
    available = [p for p in poses if valid_pose(p)]
    if available:
        neutral_root[:3] = np.median(np.asarray(available)[:, :3], axis=0)
        neutral_root[3:] = available[0][3:] / np.linalg.norm(available[0][3:])
    torso = _add(
        world, "body", name="guider_torso",
        pos=neutral_root[:3], quat=neutral_root[3:],
    )
    _add(torso, "freejoint", name="guider_root")
    _site(torso, "guider_torso")
    h = anthropometry["shoulder_height"]
    _add(
        torso, "geom", type="box",
        size=[anthropometry["shoulder_width"] * 0.42, 0.085, h],
        contype=0, conaffinity=0, rgba="0.20 0.48 0.65 1",
    )
    neck = _add(torso, "body", name="guider_neck", pos=[0, 0, h + 0.025])
    for axis, label, limits in zip(
        np.eye(3), ("x", "y", "z"), ((-0.8, 0.8), (-0.7, 0.7), (-1.4, 1.4))
    ):
        _joint(neck, f"guider_neck_{label}", axis, limits)
    head = _add(
        neck, "body", name="guider_head", pos=[0, 0, anthropometry["neck_length"]]
    )
    _add(
        head, "geom", type="sphere", size=0.085,
        contype=0, conaffinity=0, rgba="0.75 0.59 0.44 1",
    )
    _add(
        head, "geom", type="sphere", size=0.014, pos="0 -0.081 0",
        contype=0, conaffinity=0, rgba="0.5 0.35 0.25 1",
    )
    _site(head, "guider_head")
    for side_index, side in enumerate(SIDES):
        prefix = f"guider_{side}"
        upper = anthropometry["upper_arm"][prefix]
        lower = anthropometry["forearm"][prefix]
        shoulder = _add(
            torso, "body", name=f"{prefix}_upper_arm",
            pos=[(-1 if side_index == 0 else 1) * anthropometry["shoulder_width"] / 2, 0, h],
        )
        for axis, label, limits in zip(
            np.eye(3), ("x", "y", "z"),
            ((-2.8, 2.8), (-1.7, 1.7), (-2.4, 2.4)),
        ):
            _joint(shoulder, f"{prefix}_shoulder_{label}", axis, limits)
        _site(shoulder, f"{prefix}_shoulder")
        _capsule(shoulder, upper, 0.028, (0, 0), "0.25 0.55 0.72 1")
        elbow = _add(
            shoulder, "body", name=f"{prefix}_forearm", pos=[0, upper, 0]
        )
        _joint(elbow, f"{prefix}_elbow", [1, 0, 0], [0, 2.65])
        _site(elbow, f"{prefix}_elbow")
        _capsule(elbow, lower, 0.023, (0, 0), "0.75 0.59 0.44 1")
        wrist = _add(elbow, "body", name=f"{prefix}_wrist_mount", pos=[0, lower, 0])
        _hand(wrist, prefix, anthropometry["hands"][prefix], (0, 0), "0.78 0.63 0.48 1")

        prefix = f"builder_{side}"
        forearm = _add(
            world, "body", name=f"{prefix}_forearm",
            pos=[(-1 if side_index == 0 else 1) * 0.22, -extent[1] / 2 - 0.18, 0.13],
        )
        _add(forearm, "freejoint", name=f"{prefix}_root")
        _site(forearm, f"{prefix}_elbow")
        length = anthropometry["forearm"][prefix]
        _capsule(forearm, length, 0.024, (4, 3), "0.85 0.52 0.24 1")
        wrist = _add(forearm, "body", name=f"{prefix}_wrist_mount", pos=[0, length, 0])
        _hand(wrist, prefix, anthropometry["hands"][prefix], (4, 3), "0.90 0.64 0.35 1")

    block_report = []
    block_poses = usable_points(observations, "block_poses")
    dimensions = usable_points(observations, "block_dimensions")
    for j, identity in enumerate(observations["block_ids"]):
        values = dimensions[:, j]
        good = (
            np.isfinite(values).all(axis=-1)
            & (values >= 0.005).all(axis=-1)
            & (values <= 0.25).all(axis=-1) & (values[:, 2] <= 0.18)
        )
        dims = np.median(values[good], axis=0) if good.any() else np.array([0.06, 0.03, 0.025])
        valid = [p for p in block_poses[:, j] if valid_pose(p)]
        pose = valid[0].copy() if valid else np.array([
            0.08 * (j % 6 - 2.5), 0.08 * (j // 6), dims[2] / 2, 1, 0, 0, 0
        ])
        pose[3:] /= np.linalg.norm(pose[3:])
        body_name = f"block_{int(identity)}"
        body = _add(world, "body", name=body_name, pos=pose[:3], quat=pose[3:])
        _add(body, "freejoint", name=f"{body_name}_free")
        _add(
            body, "geom", name=f"{body_name}_geom", type="box", size=dims / 2,
            contype=2, conaffinity=7, density=550, rgba="0.82 0.83 0.85 1",
        )
        block_report.append({
            "id": int(identity), "body": body_name, "dimensions_m": dims.tolist(),
            "dimension_fallback_prior": not bool(good.any()),
            "initial_pose_fallback_prior": not bool(valid),
        })

    points = [
        np.array([[-extent[0] / 2, -extent[1] / 2, 0],
                  [extent[0] / 2, extent[1] / 2, 0]]),
        neutral_root[None, :3] + [[0, 0, h + anthropometry["neck_length"] + 0.12]],
    ]
    for name in (
        "guider_arm_keypoints", "builder_forearm_keypoints",
        *(f"{p}_hand_21" for p in HAND_PREFIXES),
    ):
        p = usable_points(observations, name).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if len(p):
            points.append(np.percentile(p, [1, 99], axis=0))
    bounds = np.concatenate(points)
    low, high = bounds.min(axis=0) - 0.2, bounds.max(axis=0) + 0.2
    center = (low + high) / 2
    radius = max(0.8, np.linalg.norm(high - low) / 2)
    direction = np.array([0.95, -1.25, 0.95])
    direction /= np.linalg.norm(direction)
    _camera(world, "overview", center + direction * radius * 3.0, center, 48)
    _camera(world, "top", center + [0, 0, radius * 3.0], center, 48)

    metadata = {
        "anthropometry": anthropometry, "blocks": block_report,
        "collision_bits": {"table": 1, "block": 2, "builder": 4, "guider": 0},
        "builder_roots": "free joints; no kinematic weld or mocap parent",
        "physics_performed": False,
    }
    custom = _add(root, "custom")
    _add(custom, "text", name="reconstruction_metadata", data=json.dumps(metadata))
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    audit_blocks(model, block_report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(xml + "\n", encoding="utf-8")
    return xml, metadata


def audit_blocks(model: mujoco.MjModel, blocks: list[dict]) -> None:
    for item in blocks:
        body = model.body(item["body"]).id
        if model.body_jntnum[body] != 1:
            raise ValueError(f"{item['body']} must have exactly one joint")
        joint = model.body_jntadr[body]
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{item['body']} must have a free joint")
        if model.body_parentid[body] != 0 or model.body_mocapid[body] != -1:
            raise ValueError("Blocks must be independent dynamic world children")
    # M4 intentionally generates no controllers or equalities of any kind.
    if model.nu or model.neq:
        raise ValueError("The M4 scene must not contain actuators or equalities")
