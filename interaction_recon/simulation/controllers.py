"""Physics-rate reference interpolation and rate-limited compliant controls."""
import mujoco
import numpy as np

from interaction_recon.fusion.filtering import slerp


ROOT_SPEED_M_S = 0.6
ROOT_ANGULAR_SPEED_RAD_S = 3.0
JOINT_TARGET_SPEED_RAD_S = 4.0


def limited_pose(previous, desired, dt, *, speed=ROOT_SPEED_M_S,
                 angular_speed=ROOT_ANGULAR_SPEED_RAD_S):
    result = np.asarray(previous, float).copy()
    delta = np.asarray(desired[:3]) - result[:3]
    distance = float(np.linalg.norm(delta))
    result[:3] += delta * min(1.0, max(0.0, dt) * speed / max(distance, 1e-12))
    a = previous[3:] / np.linalg.norm(previous[3:])
    b = desired[3:] / np.linalg.norm(desired[3:])
    angle = 2 * np.arccos(np.clip(abs(float(a @ b)), 0, 1))
    fraction = min(1.0, max(0.0, dt) * angular_speed / max(angle, 1e-12))
    result[3:] = slerp(a, b, fraction)
    return result


class TargetInterpolator:
    def __init__(self, model: mujoco.MjModel, targets: dict):
        self.model = model
        self.times = np.asarray(targets["timestamps"], float)
        if (
            not len(self.times) or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Controller timeline must be finite and increasing")
        self.values = np.tile(model.qpos0, (len(self.times), 1))
        used = []
        for actor in ("guider", "builder"):
            addresses = np.asarray(targets[f"{actor}_qpos_indices"], int)
            values = targets[f"{actor}_qpos_targets"]
            if values.shape != (len(self.times), len(addresses)):
                raise ValueError("Retargeted control dimensions do not match timeline")
            if not np.isfinite(values).all():
                raise ValueError("Retargeted controls must be finite")
            self.values[:, addresses] = values
            used.extend(addresses.tolist())
        used = set(used)
        self.free = []
        for joint in range(model.njnt):
            name = model.joint(joint).name or ""
            address = int(model.jnt_qposadr[joint])
            if name.startswith("block_") and any(
                x in used for x in range(address, address + 7)
            ):
                raise ValueError("Human target artifact contains block coordinates")
            if (
                name.startswith(("guider_", "builder_"))
                and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            ):
                self.free.append(address)

    def at(self, time: float) -> np.ndarray:
        left = int(np.clip(
            np.searchsorted(self.times, time, side="right") - 1, 0, len(self.times) - 1
        ))
        right = min(left + 1, len(self.times) - 1)
        fraction = (
            float(np.clip(
                (time - self.times[left]) / (self.times[right] - self.times[left]), 0, 1
            )) if right != left else 0.0
        )
        result = (1 - fraction) * self.values[left] + fraction * self.values[right]
        for address in self.free:
            result[address + 3:address + 7] = slerp(
                self.values[left, address + 3:address + 7],
                self.values[right, address + 3:address + 7], fraction,
            )
        return result


class HumanController:
    def __init__(self, model, targets, config):
        self.model, self.config = model, config
        self.reference = TargetInterpolator(model, targets)
        self.guider_qpos = np.asarray(targets["guider_qpos_indices"], int)
        self.guider_dofs = []
        self.roots = []
        self.commanded_roots = {}
        self.last_elapsed = None
        for joint in range(model.njnt):
            name = model.joint(joint).name or ""
            if name.startswith("guider_"):
                address = int(model.jnt_dofadr[joint])
                count = 6 if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else 1
                self.guider_dofs.extend(range(address, address + count))
        for side in ("left", "right"):
            qadr = int(model.joint(f"builder_{side}_root").qposadr[0])
            mocap = int(model.body(f"target_builder_{side}").mocapid[0])
            park = model.qpos0[qadr:qadr + 7].copy()
            self.roots.append((qadr, mocap, park))
            self.commanded_roots[mocap] = park.copy()
        self.actuated_qpos = np.array([
            model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)
        ], int)
        self.commanded_controls = self.builder_target(
            self.reference.times[0]
        )[self.actuated_qpos].copy()

    def builder_target(self, time):
        # Positive offset delays the builder motion. Guider time is unchanged.
        target = self.reference.at(time - self.config.timing_offset_s)
        for address, _, _ in self.roots:
            target[address:address + 3] += self.config.hand_offset_m
        return target

    def initialize_humans(self, guard):
        target = self.reference.at(self.reference.times[0])
        guard.write_qpos(self.guider_qpos, target[self.guider_qpos])
        for address, mocap, park in self.roots:
            guard.write_qpos(np.arange(address, address + 7), park)
            guard.set_mocap(mocap, park[:3], park[3:])
        guard.write_qpos(self.actuated_qpos, self.commanded_controls)
        guard.set_ctrl(self.commanded_controls)

    def kinematic_guider(self, guard, time):
        target = self.reference.at(time)
        guard.write_qpos(self.guider_qpos, target[self.guider_qpos])
        guard.write_qvel(np.asarray(self.guider_dofs, int), 0)

    def apply(self, guard, time, elapsed, *, parked=False):
        self.kinematic_guider(guard, time)
        if parked:
            self.last_elapsed = None
            for _, mocap, park in self.roots:
                self.commanded_roots[mocap] = park.copy()
                guard.set_mocap(mocap, park[:3], park[3:])
            guard.set_ctrl(self.commanded_controls)
            return
        target = self.builder_target(time)
        dt = (
            0.0 if self.last_elapsed is None
            else max(0.0, float(elapsed - self.last_elapsed))
        )
        self.last_elapsed = float(elapsed)
        fraction = float(np.clip(elapsed / self.config.startup_s, 0, 1))
        fraction = fraction * fraction * (3 - 2 * fraction)
        for address, mocap, park in self.roots:
            pose = target[address:address + 7]
            desired = np.r_[
                (1 - fraction) * park[:3] + fraction * pose[:3],
                slerp(park[3:], pose[3:], fraction),
            ]
            command = limited_pose(self.commanded_roots[mocap], desired, dt)
            self.commanded_roots[mocap] = command
            guard.set_mocap(mocap, command[:3], command[3:])
        controls = target[self.actuated_qpos]
        if self.model.nu:
            controls = np.clip(
                controls, self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        maximum = JOINT_TARGET_SPEED_RAD_S * dt
        self.commanded_controls += np.clip(
            controls - self.commanded_controls, -maximum, maximum
        )
        guard.set_ctrl(self.commanded_controls)
