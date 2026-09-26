"""Contact-only block rollout. Only MuJoCo may change locked block state."""
import json
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from interaction_recon.simulation.controllers import (
    HumanController, ROOT_SPEED_M_S, ROOT_ANGULAR_SPEED_RAD_S,
    JOINT_TARGET_SPEED_RAD_S,
)
from interaction_recon.simulation.guard import BlockWriteGuard
from interaction_recon.simulation.physics_scene import (
    PhysicsConfig, audit_physics_scene, builder_hand_geom,
)
from interaction_recon.simulation.physics_metrics import (
    MIN_REFERENCE_SAMPLES, MIN_REFERENCE_SAMPLES_PER_BLOCK,
    MIN_REFERENCE_COVERAGE, physics_metrics,
)

MAX_KINETIC_ENERGY_J = 1e6


def _projected_corners(pose, half_size):
    corners = np.array([
        [x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
    ]) * half_size
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return cv2.convexHull(
        (corners @ rotation.T + pose[:3])[:, :2].astype(np.float32)
    ).reshape(-1, 2).astype(float)


def _axes(polygon):
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    return normals[lengths > 1e-10] / lengths[lengths > 1e-10, None]


def _travel_capacity(polygon, direction, half_extent, clearance):
    capacity = np.inf
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    for axis in range(2):
        if direction[axis] > 1e-10:
            capacity = min(
                capacity, (half_extent[axis] - clearance - high[axis]) / direction[axis]
            )
        elif direction[axis] < -1e-10:
            capacity = min(
                capacity, (-half_extent[axis] + clearance - low[axis]) / direction[axis]
            )
    return max(0.0, float(capacity))


def _separating_translation(a, b, half_extent, clearance):
    axes = np.vstack((_axes(a), _axes(b)))
    projections = [(a @ axis, b @ axis) for axis in axes]
    if any(
        pa.max() <= pb.min() - clearance or pb.max() <= pa.min() - clearance
        for pa, pb in projections
    ):
        return None
    choices = []
    for axis, (pa, pb) in zip(axes, projections):
        for direction, distance in (
            (axis, pa.max() - pb.min() + clearance),
            (-axis, pb.max() - pa.min() + clearance),
        ):
            distance = max(0.0, float(distance))
            ca = _travel_capacity(a, -direction, half_extent, clearance)
            cb = _travel_capacity(b, direction, half_extent, clearance)
            if ca + cb + 1e-9 < distance:
                continue
            move_a = float(np.clip(distance / 2, max(0.0, distance - cb), ca))
            choices.append((distance, -direction * move_a, direction * (distance - move_a)))
    if not choices:
        raise ValueError("No nonpenetrating tabletop initialization fits; inspect block geometry")
    _, move_a, move_b = min(choices, key=lambda item: item[0])
    return move_a, move_b


def separate_initial_poses(poses, half_sizes, extent, *, clearance=0.0002):
    """Initialization-only correction, never called after locking the guard."""
    result = np.asarray(poses, float).reshape(-1, 7).copy()
    half_extent = np.asarray(extent, float) / 2
    for j, pose in enumerate(result):
        if np.any(np.abs(pose[:2]) > half_extent):
            raise ValueError(
                "Off-table block survived prerequisites; rerun geometric consolidation"
            )
        polygon = _projected_corners(pose, half_sizes[j])
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        lower = -half_extent + clearance - low
        upper = half_extent - clearance - high
        if np.any(lower > upper):
            raise ValueError("Block footprint cannot fit within the estimated table")
        pose[:2] += np.clip(np.zeros(2), lower, upper)
    for _ in range(200):
        changed = False
        for a in range(len(result)):
            for b in range(a + 1, len(result)):
                pa = _projected_corners(result[a], half_sizes[a])
                pb = _projected_corners(result[b], half_sizes[b])
                translation = _separating_translation(pa, pb, half_extent, clearance * 0.9)
                if translation is not None:
                    da, db = translation
                    result[a, :2] += da
                    result[b, :2] += db
                    changed = True
        if not changed:
            return result
    raise ValueError("Residual block overlap did not converge during initialization")


def initialize_blocks(model, guard, observations, config):
    records, addresses, bodies, poses, half_sizes, velocities = [], [], [], [], [], []
    for j, identity in enumerate(observations["block_ids"]):
        name = f"block_{int(identity)}"
        pose = observations["block_poses"][:, j]
        confidence = observations["block_poses_confidence"][:, j]
        observed = observations["block_poses_observed_mask"][:, j]
        good = (
            observed & (confidence >= config.initial_confidence)
            & np.isfinite(pose).all(axis=1)
            & (np.linalg.norm(pose[:, 3:], axis=1) > 1e-8)
        )
        indices = np.flatnonzero(good)
        if not len(indices):
            raise ValueError(f"{name}: no confident observed initialization pose")
        first = int(indices[0])
        initial = pose[first].copy()
        initial[3:] /= np.linalg.norm(initial[3:])
        geom = model.geom(f"{name}_geom").id
        size = model.geom_size[geom].copy()
        rotation = Rotation.from_quat(initial[[4, 5, 6, 3]]).as_matrix()
        initial[2] = float(np.abs(rotation[2]) @ size) + 0.0002
        joint = model.joint(f"{name}_free").id
        qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        addresses.append(np.arange(qa, qa + 7))
        velocities.append(np.arange(va, va + 6))
        bodies.append(model.body(name).id)
        poses.append(initial)
        half_sizes.append(size)
        records.append({
            "id": int(identity), "source_frame": first,
            "source_timestamp_s": float(observations["timestamps"][first]),
            "confidence": float(confidence[first]),
            "method": "first_confident_observation_table_height_minimal_separation",
            "original_pose": pose[first].tolist(),
        })
    poses = np.asarray(poses, float).reshape(-1, 7)
    half_sizes = np.asarray(half_sizes, float).reshape(-1, 3)
    separated = separate_initial_poses(poses, half_sizes, observations["table_extent_xy"])
    for j, initial in enumerate(separated):
        guard.write_qpos(addresses[j], initial)
        guard.write_qvel(velocities[j], 0)
        records[j].update({
            "initial_pose": initial.tolist(), "initial_z_m": float(initial[2]),
            "original_z_m": float(records[j]["original_pose"][2]),
            "separation_shift_xy_m": (initial[:2] - poses[j, :2]).tolist(),
        })
    return np.asarray(addresses, int).reshape(-1, 7), np.asarray(bodies, int), records


def contact_records(model, data):
    pairs, positions, forces, depth = [], [], [], []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        pairs.append([contact.geom1, contact.geom2])
        positions.append(contact.pos.copy())
        forces.append(force)
        depth.append(max(0.0, -float(contact.dist)))
    return (
        np.asarray(pairs, int).reshape(-1, 2),
        np.asarray(positions, float).reshape(-1, 3),
        np.asarray(forces, float).reshape(-1, 6),
        np.asarray(depth, float),
    )


def _block_penetration(model, data, bodies):
    bodies = set(map(int, bodies))
    return max((
        max(0.0, -float(contact.dist))
        for contact in data.contact
        if (
            int(model.geom_bodyid[contact.geom1]) in bodies
            or int(model.geom_bodyid[contact.geom2]) in bodies
        )
    ), default=0.0)


def fatal_instability(model, data, previous_time):
    if not (
        np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all() and np.isfinite(data.time)
    ):
        return True, "nonfinite_state"
    if data.time < previous_time - 1e-9:
        return True, "engine_reset_after_numerical_failure"
    for warning in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
    ):
        if data.warning[int(warning)].number:
            return True, "engine_invalid_state_warning"
    mujoco.mj_energyVel(model, data)
    kinetic = float(data.energy[1])
    if not np.isfinite(kinetic) or kinetic > MAX_KINETIC_ENERGY_J:
        return True, "kinetic_energy_explosion"
    return False, ""


def run_rollout(
    xml: str, targets: dict, observations: dict,
    config: PhysicsConfig = PhysicsConfig(), *,
    stop_time_s: float | None = None, deadline: float | None = None,
) -> dict:
    """Short search rollouts simulate the original prefix, never reset midclip."""
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Physics candidate exceeded refinement deadline")

    check_deadline()
    model = audit_physics_scene(xml)
    if not np.isclose(model.opt.timestep, config.timestep):
        raise ValueError("PhysicsConfig timestep differs from compiled scene")
    data = mujoco.MjData(model)
    guard = BlockWriteGuard(model, data)
    addresses, bodies, initialization = initialize_blocks(model, guard, observations, config)
    controller = HumanController(model, targets, config)
    controller.initialize_humans(guard)
    guard.forward()
    initial_penetration = _block_penetration(model, data, bodies)
    if initial_penetration > 0.001:
        raise ValueError(f"Unsafe initial block penetration: {initial_penetration:.6f} m")
    guard.lock()

    times = np.asarray(targets["timestamps"], float)
    tail = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 15
    start = min(float(times[0]), float(observations.get("builder_clip_start_s", times[0])))
    end = max(
        float(times[-1] + tail),
        float(observations.get("builder_clip_end_s", times[-1] + tail)),
    )
    if stop_time_s is not None:
        end = min(end, float(stop_time_s))
    if end <= start:
        raise ValueError("Rollout end must follow its start")
    preroll_warning = np.array([warning.number for warning in data.warning], int)
    previous_time = float(data.time)
    for step in range(int(np.ceil(config.preroll_s / config.timestep))):
        if step % 32 == 0:
            check_deadline()
        controller.apply(guard, start, 0, parked=True)
        guard.step()
        fatal, reason = fatal_instability(model, data, previous_time)
        if fatal:
            raise RuntimeError(f"Numerical failure during settling: {reason}")
        previous_time = float(data.time)
    guard.forward()
    settled_penetration = _block_penetration(model, data, bodies)
    if settled_penetration > 0.001:
        raise ValueError(f"Settled block penetration exceeds 1 mm: {settled_penetration:.6f} m")
    preroll_warning = np.array([warning.number for warning in data.warning], int) - preroll_warning
    origin = float(data.time)
    steps = int(np.ceil((end - start) / config.timestep - 1e-9))
    records = {name: [] for name in (
        "timestamps", "qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat",
        "block_poses", "hand_block_contact", "nan_flags", "instability_flags",
        "joint_limit_overshoot_rad", "warning_counts", "block_fallen_flags",
        "kinetic_energy_j",
    )}
    contact_offsets = [0]
    contacts, positions, forces, depths = [], [], [], []
    body_to_block = {int(body): j for j, body in enumerate(bodies)}
    hand_geoms = np.array([builder_hand_geom(model, g) for g in range(model.ngeom)])
    limited = np.flatnonzero(
        model.jnt_limited & (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    )
    failed, termination_reason = False, ""
    previous_time = float(data.time)
    fallen = np.zeros(len(bodies), bool)
    for step in range(steps + 1):
        if step % 32 == 0:
            check_deadline()
        elapsed = step * config.timestep
        timestamp = start + elapsed
        controller.apply(guard, timestamp, elapsed)
        guard.forward()
        state = guard.state()
        pairs, points, force, depth = contact_records(model, data)
        touching = np.zeros(len(bodies), bool)
        for g1, g2 in pairs:
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if b1 in body_to_block and hand_geoms[g2]:
                touching[body_to_block[b1]] = True
            if b2 in body_to_block and hand_geoms[g1]:
                touching[body_to_block[b2]] = True
        nan = not all(np.isfinite(value).all() for value in state.values())
        warnings = np.array([warning.number for warning in data.warning], int)
        instability, reason = fatal_instability(model, data, previous_time)
        instability |= nan
        block_poses = data.qpos[addresses].copy()
        fallen |= block_poses[:, 2] < -0.05
        overshoot = 0.0
        if len(limited):
            values = data.qpos[model.jnt_qposadr[limited]]
            low, high = model.jnt_range[limited].T
            overshoot = float(np.max(np.maximum(0, np.maximum(low - values, values - high))))
        records["timestamps"].append(timestamp)
        for name, value in state.items():
            records[name].append(value)
        records["block_poses"].append(block_poses)
        records["hand_block_contact"].append(touching)
        records["nan_flags"].append(nan)
        records["instability_flags"].append(instability)
        records["joint_limit_overshoot_rad"].append(overshoot)
        records["warning_counts"].append(warnings)
        records["block_fallen_flags"].append(fallen.copy())
        records["kinetic_energy_j"].append(float(data.energy[1]))
        contacts.append(pairs)
        positions.append(points)
        forces.append(force)
        depths.append(depth)
        contact_offsets.append(contact_offsets[-1] + len(pairs))
        if instability:
            failed = True
            termination_reason = reason or "nonfinite_control_state"
            break
        if step < steps:
            previous_time = float(data.time)
            guard.step()
    output = {name: np.asarray(value) for name, value in records.items()}
    output.update({
        "block_ids": np.asarray(observations["block_ids"]),
        "block_qpos_addresses": addresses,
        "block_qvel_addresses": guard.block_qvel_addresses,
        "contact_offsets": np.asarray(contact_offsets, np.int64),
        "contacts": np.concatenate(contacts, axis=0),
        "contact_positions": np.concatenate(positions, axis=0),
        "contact_forces": np.concatenate(forces, axis=0),
        "penetration_depths": np.concatenate(depths),
        "builder_hand_geom_mask": hand_geoms,
        "geom_names": np.array([model.geom(g).name or f"geom_{g}" for g in range(model.ngeom)]),
        "geom_body_names": np.array([
            model.body(int(body)).name or "world" for body in model.geom_bodyid
        ]),
        "preroll_warning_counts": preroll_warning,
        "initial_max_block_penetration_m": np.array(initial_penetration),
        "settled_max_block_penetration_m": np.array(settled_penetration),
        "requested_start_s": np.array(start), "requested_end_s": np.array(end),
        "initialization_json": np.array(json.dumps(initialization, allow_nan=False)),
        "guard_log_json": np.array(json.dumps(guard.report(), allow_nan=False)),
        "metadata_json": np.array(json.dumps({
            "schema": "physics_rollout_v4", "pose_layout": "xyz,wxyz",
            "units": "meters, seconds, radians, Newtons",
            "sampling": "every physics substep, including initial settled state",
            "contact_layout": "CSR: contact_offsets[k]:contact_offsets[k+1]",
            "contact_forces": "mj_contactForce: contact-frame force xyz and torque xyz",
            "contacts": "geom-id pairs; positions in world coordinates",
            "block_policy": "initialized once, then only mj_step changes block qpos/qvel",
            "guider": "kinematic qpos targets each substep, zero collision masks",
            "startup_transition_s": config.startup_s,
            "root_target_speed_limit_m_s": ROOT_SPEED_M_S,
            "root_target_angular_speed_limit_rad_s": ROOT_ANGULAR_SPEED_RAD_S,
            "joint_target_speed_limit_rad_s": JOINT_TARGET_SPEED_RAD_S,
            "builder_translation_offset_m": list(config.hand_offset_m),
            "builder_timing_offset_s": config.timing_offset_s,
            "contact_timeconst_requested_s": config.contact_timeconst,
            "contact_timeconst_effective_lower_bound_s": 2 * config.timestep,
            "preroll_s": origin, "terminated_early": failed,
            "termination_reason": termination_reason,
            "fall_policy": "persistent flagged event; never terminate solely for falling",
            "kinetic_energy_explosion_threshold_j": MAX_KINETIC_ENERGY_J,
        }, allow_nan=False)),
    })
    check_deadline()
    output["metrics_json"] = np.array(json.dumps(
        physics_metrics(output, observations, model), allow_nan=False
    ))
    return output
