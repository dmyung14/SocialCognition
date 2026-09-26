import json

import mujoco
import numpy as np

from interaction_recon.retarget.arm_ik import (
    forward_kinematics, joint_addresses, solve_ik,
)
from interaction_recon.retarget.hand_ik import hand_angles
from interaction_recon.retarget.models import (
    FINGERS, HAND_PREFIXES, SIDES, RetargetConfig, axis_frame,
    finger_joint_names, palm_frame, quaternion, rotation, usable_points, valid_pose,
)


def actor_qpos_indices(model, actor: str) -> np.ndarray:
    indices = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name and name.startswith(f"{actor}_"):
            address = model.jnt_qposadr[j]
            count = 7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            indices.extend(range(address, address + count))
    return np.asarray(indices, int)


def neutral_qpos(model) -> np.ndarray:
    qpos = model.qpos0.copy()
    for side in SIDES:
        qpos[model.joint(f"guider_{side}_shoulder_x").qposadr[0]] = -1.57
        qpos[model.joint(f"guider_{side}_elbow").qposadr[0]] = 0.75
    return qpos


def _free_indices(model, name):
    address = int(model.joint(name).qposadr[0])
    return np.arange(address, address + 7)


def _hinges(model, names):
    return joint_addresses(model, names)[0]


def _site_pose(model, data, name):
    site = model.site(name).id
    return np.r_[data.site_xpos[site], quaternion(data.site_xmat[site].reshape(3, 3))]


def _evidence(observations, name, frame, selection=None):
    confidence = np.asarray(observations[f"{name}_confidence"][frame])
    observed = np.asarray(observations[f"{name}_observed_mask"][frame])
    interpolated = np.asarray(observations[f"{name}_interpolated_mask"][frame])
    source = np.asarray(observations[f"{name}_source"][frame], np.uint8)
    if selection is not None:
        confidence, observed, interpolated, source = (
            a[selection] for a in (confidence, observed, interpolated, source)
        )
    finite = confidence[np.isfinite(confidence) & (confidence > 0)]
    return (
        float(np.min(finite)) if len(finite) else 0.0,
        bool(np.all(observed)), bool(np.any(interpolated)),
        int(np.bitwise_or.reduce(source.ravel(), initial=np.uint8(0))),
    )


class _Channels:
    def __init__(self, model, neutral, count, hold_s):
        self.model = model
        self.neutral = neutral
        self.hold_s = hold_s
        self.names = []
        self.indices = {}
        self.last_time = {}
        self.last_confidence = {}
        self.last_values = {}
        self.records = {}
        self.count = count

    def add(self, name, indices):
        self.names.append(name)
        self.indices[name] = np.asarray(indices, int)
        self.last_time[name] = -np.inf
        self.last_confidence[name] = 0.0
        self.last_values[name] = self.neutral[indices].copy()
        self.records[name] = {
            "valid": np.zeros(self.count, bool),
            "observed": np.zeros(self.count, bool),
            "interpolated": np.zeros(self.count, bool),
            "held": np.zeros(self.count, bool),
            "rest": np.zeros(self.count, bool),
            "confidence": np.zeros(self.count),
            "source": np.zeros(self.count, np.uint8),
            "residual": np.full(self.count, np.nan),
            "orientation_residual": np.full(self.count, np.nan),
            "method": np.full(self.count, "neutral_rest", dtype="<U64"),
        }

    def apply(self, data, name, i, time, available, evidence, update, method):
        record = self.records[name]
        indices = self.indices[name]
        if available and evidence[0] > 0:
            update()
            self.last_values[name] = data.qpos[indices].copy()
            self.last_time[name] = time
            self.last_confidence[name] = evidence[0]
            record["valid"][i] = True
            record["confidence"][i] = evidence[0]
            record["observed"][i], record["interpolated"][i] = evidence[1:3]
            record["source"][i] = evidence[3]
            record["method"][i] = method
        else:
            age = time - self.last_time[name]
            if age <= self.hold_s + 1e-9:
                data.qpos[indices] = self.last_values[name]
                record["held"][i] = True
                record["confidence"][i] = self.last_confidence[name] * np.exp(
                    -age / max(self.hold_s, 1e-6)
                )
                record["method"][i] = "held_local_pose"
            else:
                data.qpos[indices] = self.neutral[indices]
                record["rest"][i] = True
        return bool(record["valid"][i])


def retarget_observations(
    observations: dict, xml: str, scene_metadata: dict,
    config: RetargetConfig = RetargetConfig(),
) -> dict:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    times = np.asarray(observations["timestamps"], float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Retargeting requires a nonempty increasing timeline")
    count = len(times)
    neutral = neutral_qpos(model)
    data.qpos[:] = neutral
    anthropometry = scene_metadata["anthropometry"]
    tracks = {
        name: usable_points(observations, name)
        for name in (
            "guider_torso_pose", "guider_head_pose",
            "guider_arm_keypoints", "builder_forearm_keypoints",
            *(f"{prefix}_hand_21" for prefix in HAND_PREFIXES),
        )
    }
    channels = _Channels(model, neutral, count, config.hold_s)
    channels.add("guider_root", _free_indices(model, "guider_root"))
    neck_names = [f"guider_neck_{a}" for a in "xyz"]
    channels.add("guider_head", _hinges(model, neck_names))
    arm_names = {}
    for prefix in HAND_PREFIXES:
        if prefix.startswith("guider"):
            arm_names[prefix] = [
                *(f"{prefix}_shoulder_{a}" for a in "xyz"), f"{prefix}_elbow"
            ]
            channels.add(f"{prefix}_arm", _hinges(model, arm_names[prefix]))
        else:
            channels.add(f"{prefix}_root", _free_indices(model, f"{prefix}_root"))
        channels.add(
            f"{prefix}_wrist", _hinges(model, [f"{prefix}_wrist_{a}" for a in "xyz"])
        )
        for finger in FINGERS:
            channels.add(
                f"{prefix}_{finger}", _hinges(model, finger_joint_names(prefix, finger))
            )

    indices = {actor: actor_qpos_indices(model, actor) for actor in ("guider", "builder")}
    qpos_targets = {
        actor: np.empty((count, len(addresses))) for actor, addresses in indices.items()
    }
    wrist_poses = np.empty((count, 4, 7))
    wrist_targets = np.full_like(wrist_poses, np.nan)
    root_names = ("guider_root", "builder_left_root", "builder_right_root")
    root_poses = np.empty((count, 3, 7))
    root_targets = np.full_like(root_poses, np.nan)
    settings = {
        "iterations": config.iterations, "damping": config.damping,
        "regularizer": config.regularizer,
        "orientation_weight": config.orientation_weight,
    }

    for i, time in enumerate(times):
        position_evaluations = {}
        orientation_evaluations = {}
        root_pose = tracks["guider_torso_pose"][i]
        root_indices = channels.indices["guider_root"]

        def set_torso():
            value = root_pose.copy()
            value[3:] /= np.linalg.norm(value[3:])
            data.qpos[root_indices] = value
            root_targets[i, 0] = value

        if channels.apply(
            data, "guider_root", i, time, valid_pose(root_pose),
            _evidence(observations, "guider_torso_pose", i), set_torso,
            "observed_torso_pose",
        ):
            position_evaluations["guider_root"] = {"guider_torso": root_pose[:3]}
            orientation_evaluations["guider_root"] = ("guider_torso", rotation(root_pose[3:]))

        head_pose = tracks["guider_head_pose"][i]

        def set_head():
            solve_ik(
                model, data, neck_names, {"guider_head": head_pose[:3]},
                ("guider_head", rotation(head_pose[3:])), **settings,
            )

        if channels.apply(
            data, "guider_head", i, time, valid_pose(head_pose),
            _evidence(observations, "guider_head_pose", i), set_head,
            "neck_position_orientation_dls",
        ):
            position_evaluations["guider_head"] = {"guider_head": head_pose[:3]}
            orientation_evaluations["guider_head"] = ("guider_head", rotation(head_pose[3:]))

        for pindex, prefix in enumerate(HAND_PREFIXES):
            side = prefix.split("_")[1]
            side_index = SIDES.index(side)
            hand_name = f"{prefix}_hand_21"
            hand = tracks[hand_name][i]
            palm = palm_frame(hand, side)
            if prefix.startswith("guider"):
                arm = tracks["guider_arm_keypoints"][i, side_index]
                targets = {
                    f"{prefix}_{joint}": arm[k]
                    for k, joint in enumerate(("shoulder", "elbow", "wrist"))
                    if np.isfinite(arm[k]).all()
                }
                available = any(name.endswith(("elbow", "wrist")) for name in targets)
                evidence = _evidence(observations, "guider_arm_keypoints", i, side_index)

                def set_arm():
                    solve_ik(model, data, arm_names[prefix], targets, **settings)

                channel = f"{prefix}_arm"
                if channels.apply(
                    data, channel, i, time, available, evidence, set_arm,
                    "shoulder_elbow_wrist_dls",
                ):
                    position_evaluations[channel] = targets
            else:
                forearm = tracks["builder_forearm_keypoints"][i, side_index]
                length = anthropometry["forearm"][prefix]
                pose = None
                evidence = _evidence(
                    observations, "builder_forearm_keypoints", i, side_index
                )
                method = "forearm_keypoints"
                if np.isfinite(forearm).all() and np.linalg.norm(forearm[1] - forearm[0]) > 1e-5:
                    frame = axis_frame(
                        forearm[1] - forearm[0],
                        palm[:, 0] if palm is not None else np.array([1.0, 0, 0]),
                    )
                    # Preserve observed wrist position despite clipped forearm length.
                    pose = np.r_[forearm[1] - length * frame[:, 1], quaternion(frame)]
                elif palm is not None:
                    frame = palm
                    pose = np.r_[hand[0] - length * frame[:, 1], quaternion(frame)]
                    evidence = _evidence(observations, hand_name, i, [0, 5, 9, 17])
                    evidence = (min(evidence[0], 0.12), False, evidence[2], evidence[3])
                    method = "hand_axis_forearm_prior"
                root_channel = f"{prefix}_root"
                root_slot = side_index + 1

                def set_forearm():
                    data.qpos[channels.indices[root_channel]] = pose
                    root_targets[i, root_slot] = pose

                if channels.apply(
                    data, root_channel, i, time, pose is not None,
                    evidence, set_forearm, method,
                ):
                    targets = {}
                    if np.isfinite(forearm[0]).all():
                        targets[f"{prefix}_elbow"] = forearm[0]
                    if np.isfinite(forearm[1]).all():
                        targets[f"{prefix}_wrist"] = forearm[1]
                    elif palm is not None:
                        targets[f"{prefix}_wrist"] = hand[0]
                    position_evaluations[root_channel] = targets

            wrist_channel = f"{prefix}_wrist"

            def set_wrist():
                solve_ik(
                    model, data, [f"{prefix}_wrist_{a}" for a in "xyz"], {},
                    (wrist_channel, palm), **settings,
                )
                wrist_targets[i, pindex] = np.r_[hand[0], quaternion(palm)]

            if channels.apply(
                data, wrist_channel, i, time, palm is not None,
                _evidence(observations, hand_name, i, [0, 5, 9, 17]),
                set_wrist, "palm_orientation_dls",
            ):
                position_evaluations[wrist_channel] = {wrist_channel: hand[0]}
                orientation_evaluations[wrist_channel] = (wrist_channel, palm)

            previous = np.array([
                data.qpos[channels.indices[f"{prefix}_{finger}"]]
                for finger in FINGERS
            ])
            angles, finger_valid = hand_angles(
                hand, side, anthropometry["hands"][prefix]["thumb_yaw"],
                previous, config.finger_regularizer,
            )
            for f, finger in enumerate(FINGERS):
                channel = f"{prefix}_{finger}"
                selected = np.arange(1 + 4 * f, 5 + 4 * f)

                def set_finger(f=f, channel=channel):
                    data.qpos[channels.indices[channel]] = angles[f]

                if channels.apply(
                    data, channel, i, time, bool(finger_valid[f]),
                    _evidence(observations, hand_name, i, selected),
                    set_finger, "bone_direction_angles",
                ):
                    position_evaluations[channel] = {
                        f"{prefix}_landmark_{k}": hand[k] for k in selected
                    }

        forward_kinematics(model, data)
        for name, targets in position_evaluations.items():
            errors = [
                np.linalg.norm(data.site_xpos[model.site(site).id] - target)
                for site, target in targets.items()
            ]
            if errors:
                channels.records[name]["residual"][i] = np.sqrt(np.mean(np.square(errors)))
        from scipy.spatial.transform import Rotation
        for name, (site, target) in orientation_evaluations.items():
            actual = data.site_xmat[model.site(site).id].reshape(3, 3)
            channels.records[name]["orientation_residual"][i] = Rotation.from_matrix(
                target @ actual.T
            ).magnitude()
        for actor, addresses in indices.items():
            qpos_targets[actor][i] = data.qpos[addresses]
        for k, prefix in enumerate(HAND_PREFIXES):
            wrist_poses[i, k] = _site_pose(model, data, f"{prefix}_wrist")
        for k, name in enumerate(root_names):
            root_poses[i, k] = data.qpos[channels.indices[name]]

    output = {
        "timestamps": times,
        "chain_names": np.array(channels.names),
        "wrist_names": np.array([f"{p}_wrist" for p in HAND_PREFIXES]),
        "root_names": np.array(root_names),
        "wrist_poses": wrist_poses,
        "root_poses": root_poses,
        "wrist_pose_targets": wrist_targets,
        "root_pose_targets": root_targets,
        "neutral_human_qpos_indices": np.r_[indices["guider"], indices["builder"]],
        "neutral_human_qpos": neutral[np.r_[indices["guider"], indices["builder"]]],
        "metadata_json": np.array(json.dumps({
            "schema": "human_retarget_v1", "pose_layout": "xyz,wxyz",
            "units": "meters, radians, seconds",
            "block_qpos_included": False,
            "validity": "usable upstream target, including explicit upstream priors",
            "observed_mask": "all contributing selected landmarks detection-derived",
            "hold": "local joint pose, <=0.5 seconds; confidence decays exponentially",
            "missing_after_hold": "neutral rest, zero confidence",
            "residual": "world-space landmark RMS per chain, evaluated after all updates",
            "hand_method": "signed bone-direction angles with bounded MCP spread",
            "self_collision_penalty": "not optimized in this kinematic baseline",
        })),
    }
    key_mapping = {
        "valid": "validity_mask", "observed": "observed_mask",
        "interpolated": "interpolated_mask", "held": "held_mask", "rest": "rest_mask",
        "confidence": "confidence", "source": "source",
        "residual": "ik_residuals_m", "orientation_residual": "orientation_residuals_rad",
        "method": "method",
    }
    for source, target in key_mapping.items():
        output[target] = np.stack(
            [channels.records[name][source] for name in channels.names], axis=1
        )
    joint_names, joint_addresses_out = [], []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.startswith(("guider_", "builder_")):
            joint_names.append(name)
            joint_addresses_out.append(int(model.jnt_qposadr[j]))
    output["joint_names"] = np.array(joint_names)
    output["joint_qpos_addresses"] = np.array(joint_addresses_out)
    for actor in indices:
        output[f"{actor}_qpos_targets"] = qpos_targets[actor]
        output[f"{actor}_qpos_indices"] = indices[actor]
    output["metrics_json"] = np.array(json.dumps(retarget_metrics(model, output), allow_nan=False))
    return output


def retarget_metrics(model, arrays: dict) -> dict:
    def stats(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        return {
            "samples": len(values),
            "mean": float(np.mean(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
        }

    result = {
        "ik_residual_m": {},
        "orientation_residual_rad": {},
        "retarget_coverage": {},
        "joint_limit_violations": 0,
        "max_joint_limit_overshoot_rad": 0.0,
        "physics_performed": False,
        "visual_pass": "not_evaluated_automatically",
    }
    for k, name in enumerate(arrays["chain_names"]):
        name = str(name)
        result["ik_residual_m"][name] = stats(arrays["ik_residuals_m"][:, k])
        result["orientation_residual_rad"][name] = stats(arrays["orientation_residuals_rad"][:, k])
        result["retarget_coverage"][name] = {
            key: float(np.mean(arrays[key][:, k]))
            for key in ("validity_mask", "observed_mask", "interpolated_mask", "held_mask", "rest_mask")
        }
    for actor in ("guider", "builder"):
        lookup = {
            address: i for i, address in enumerate(arrays[f"{actor}_qpos_indices"])
        }
        for j in range(model.njnt):
            if not model.jnt_limited[j] or model.jnt_qposadr[j] not in lookup:
                continue
            values = arrays[f"{actor}_qpos_targets"][:, lookup[model.jnt_qposadr[j]]]
            low, high = model.jnt_range[j]
            overshoot = np.maximum(0, np.maximum(low - values, values - high))
            result["joint_limit_violations"] += int(np.count_nonzero(overshoot > 1e-7))
            result["max_joint_limit_overshoot_rad"] = max(
                result["max_joint_limit_overshoot_rad"], float(np.max(overshoot))
            )
    return result
