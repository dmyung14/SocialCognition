from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def forward_kinematics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Only the position computations needed by mj_jac; do not compute contacts."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def joint_addresses(model, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([model.joint(name).id for name in names], dtype=int)
    if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("DLS coordinates must be scalar hinge joints")
    return model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()


@dataclass
class IKResult:
    position_residuals_m: np.ndarray
    orientation_residual_rad: float
    iterations: int


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    position_targets: dict[str, np.ndarray],
    orientation_target: tuple[str, np.ndarray] | None = None,
    *,
    iterations: int = 48,
    damping: float = 0.012,
    regularizer: float = 1e-5,
    orientation_weight: float = 0.08,
) -> IKResult:
    """Bounded damped least squares with warm-start regularization and line search."""
    ids = np.array([model.joint(name).id for name in joint_names], int)
    qa, da = joint_addresses(model, joint_names)
    lo, hi = model.jnt_range[ids].T
    warm = np.clip(data.qpos[qa].copy(), lo, hi)
    data.qpos[qa] = warm
    targets = [
        (model.site(name).id, np.asarray(target, float))
        for name, target in position_targets.items()
        if np.isfinite(target).all()
    ]
    orientation = None
    if orientation_target is not None:
        name, matrix = orientation_target
        if np.isfinite(matrix).all():
            orientation = model.site(name).id, np.asarray(matrix, float)

    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def evaluate(with_jacobian):
        forward_kinematics(model, data)
        errors, jacobians = [], []
        distances = []
        angle = np.nan
        for site, target in targets:
            delta = target - data.site_xpos[site]
            distances.append(np.linalg.norm(delta))
            errors.append(delta)
            if with_jacobian:
                mujoco.mj_jac(
                    model, data, jp, jr, data.site_xpos[site], model.site_bodyid[site]
                )
                jacobians.append(jp[:, da].copy())
        if orientation is not None:
            site, target = orientation
            current = data.site_xmat[site].reshape(3, 3)
            delta = Rotation.from_matrix(target @ current.T).as_rotvec()
            angle = float(np.linalg.norm(delta))
            errors.append(orientation_weight * delta)
            if with_jacobian:
                mujoco.mj_jac(
                    model, data, jp, jr, data.site_xpos[site], model.site_bodyid[site]
                )
                jacobians.append(orientation_weight * jr[:, da].copy())
        error = np.concatenate(errors) if errors else np.empty(0)
        jac = np.vstack(jacobians) if jacobians else np.empty((0, len(qa)))
        cost = float(error @ error + regularizer * np.sum((data.qpos[qa] - warm) ** 2))
        return error, jac, cost, np.asarray(distances), angle

    used = 0
    for used in range(1, iterations + 1):
        error, jac, cost, _, _ = evaluate(True)
        if not len(error) or np.linalg.norm(error) < 1e-6:
            break
        delta = np.linalg.solve(
            jac.T @ jac + (damping ** 2 + regularizer + 1e-12) * np.eye(len(qa)),
            jac.T @ error - regularizer * (data.qpos[qa] - warm),
        )
        delta *= min(1.0, 0.3 / max(np.linalg.norm(delta), 1e-12))
        previous = data.qpos[qa].copy()
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            data.qpos[qa] = np.clip(previous + fraction * delta, lo, hi)
            if evaluate(False)[2] < cost - 1e-14:
                accepted = True
                break
        if not accepted:
            data.qpos[qa] = previous
            break
    _, _, _, distances, angle = evaluate(False)
    return IKResult(distances, angle, used)
