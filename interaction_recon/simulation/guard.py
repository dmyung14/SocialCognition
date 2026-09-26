"""Application-write guard and machine-readable evidence from the engine path."""
import mujoco
import numpy as np


class BlockWriteGuard:
    def __init__(self, model, data):
        self.model, self._data = model, data
        qpos, qvel = [], []
        for joint in range(model.njnt):
            if (model.joint(joint).name or "").startswith("block_"):
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError("Block guard requires free joints")
                qa, va = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
                qpos.extend(range(qa, qa + 7))
                qvel.extend(range(va, va + 6))
        self.block_qpos_addresses = np.asarray(qpos, int)
        self.block_qvel_addresses = np.asarray(qvel, int)
        self.locked = False
        self.checks = self.engine_steps = self.locked_application_writes = 0
        self.violations = 0
        self._snapshot()

    def _snapshot(self):
        self._qpos = self._data.qpos[self.block_qpos_addresses].copy()
        self._qvel = self._data.qvel[self.block_qvel_addresses].copy()

    def lock(self):
        self._snapshot()
        self.locked = True

    def check(self):
        if not self.locked:
            return
        self.checks += 1
        if (
            not np.array_equal(
                self._qpos, self._data.qpos[self.block_qpos_addresses], equal_nan=True
            )
            or not np.array_equal(
                self._qvel, self._data.qvel[self.block_qvel_addresses], equal_nan=True
            )
        ):
            self.violations += 1
            raise RuntimeError("Unauthorized block qpos/qvel write after initialization")

    def _write(self, name, indices, values):
        self.check()
        indices = np.arange(getattr(self._data, name).size)[indices]
        protected = (
            self.block_qpos_addresses if name == "qpos" else self.block_qvel_addresses
        )
        if self.locked and np.isin(indices, protected).any():
            self.violations += 1
            raise RuntimeError(f"Forbidden block {name} write after initialization")
        if self.locked:
            self.locked_application_writes += 1
        getattr(self._data, name)[indices] = values

    def write_qpos(self, indices, values):
        self._write("qpos", indices, values)

    def write_qvel(self, indices, values):
        self._write("qvel", indices, values)

    def set_ctrl(self, values):
        self.check()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite control")
        self._data.ctrl[:] = values

    def set_mocap(self, index, position, quaternion):
        self.check()
        if not np.isfinite(np.r_[position, quaternion]).all():
            raise ValueError("Nonfinite mocap target")
        self._data.mocap_pos[index] = position
        self._data.mocap_quat[index] = quaternion

    def step(self):
        self.check()
        mujoco.mj_step(self.model, self._data)
        if self.locked:
            self.engine_steps += 1
        self._snapshot()

    def forward(self):
        self.check()
        mujoco.mj_forward(self.model, self._data)
        self.check()

    def state(self):
        self.check()
        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "ctrl": self._data.ctrl.copy(),
            "mocap_pos": self._data.mocap_pos.copy(),
            "mocap_quat": self._data.mocap_quat.copy(),
        }

    def report(self) -> dict:
        self.check()
        return {
            "schema": "block_write_guard_v1",
            "locked": self.locked,
            "checks": self.checks,
            "engine_steps": self.engine_steps,
            "locked_application_writes": self.locked_application_writes,
            "violations": self.violations,
            "block_qpos_addresses": self.block_qpos_addresses.tolist(),
            "block_qvel_addresses": self.block_qvel_addresses.tolist(),
            "scope": "initialization excluded; settling and rollout included",
            "limitation": (
                "Runtime application-write checks, not cryptographic proof against "
                "malicious modification of the guard or its output."
            ),
        }
