import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from interaction_recon.retarget.retarget import retarget_observations
from interaction_recon.simulation.build_scene import build_scene
from interaction_recon.simulation.controllers import TargetInterpolator
from interaction_recon.simulation.physics_scene import (
    PhysicsConfig, audit_physics_scene, physics_variant,
)
from interaction_recon.simulation.rollout import BlockWriteGuard, run_rollout
from test_scene import empty_observations


def observed_block(times=(0, 0.1, 0.2)):
    observations = empty_observations(times=times, blocks=1)
    observations["block_poses"][:] = [0, 0, 0.0125, 1, 0, 0, 0]
    observations["block_dimensions"][:] = [0.08, 0.04, 0.025]
    for name in ("block_poses", "block_dimensions"):
        observations[f"{name}_confidence"][:] = 0.8
        observations[f"{name}_observed_mask"][:] = True
        observations[f"{name}_source"][:] = 2
    return observations


def test_scene_has_no_block_actuation_constraints_tendons_or_mocap():
    observations = observed_block()
    xml, _ = build_scene(observations)
    xml = physics_variant(xml)
    model = audit_physics_scene(xml)
    assert model.nu == 46
    assert model.neq == 2
    assert model.ntendon == 0
    assert model.body("block_0").mocapid[0] == -1
    assert np.all(model.actuator_forcelimited)
    tree = ET.fromstring(xml)
    for container in ("actuator", "equality", "tendon"):
        for element in tree.findall(f"./{container}//*"):
            assert all(
                not value.startswith("block_")
                for name, value in element.attrib.items() if name != "name"
            )
    bad = ET.fromstring(xml)
    ET.SubElement(
        bad.find("equality"), "weld",
        body1="block_0", body2="builder_right_palm",
    )
    with pytest.raises(ValueError, match="block"):
        audit_physics_scene(ET.tostring(bad, encoding="unicode"))


def test_guard_rejects_block_pose_and_velocity_writes_and_alias_writes():
    xml, _ = build_scene(observed_block())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    guard = BlockWriteGuard(model, data)
    guard.lock()
    qa = int(model.joint("block_0_free").qposadr[0])
    va = int(model.joint("block_0_free").dofadr[0])
    with pytest.raises(RuntimeError, match="block qpos"):
        guard.write_qpos(qa, data.qpos[qa])
    with pytest.raises(RuntimeError, match="block qvel"):
        guard.write_qvel(va, 0)
    data.qpos[qa] += 0.01
    with pytest.raises(RuntimeError, match="Unauthorized"):
        guard.step()


def contact_fixture():
    xml = """
    <mujoco>
      <option timestep="0.002" integrator="implicitfast"/>
      <default>
        <geom friction="0.7 0.01 0.001" condim="4" solref="0.01 1"/>
      </default>
      <worldbody>
        <geom name="table" type="box" size="1 1 0.05" pos="0 0 -0.05"/>
        <body name="block_0" pos="0 0 0.025">
          <freejoint name="block_0_free"/>
          <geom name="block_0_geom" type="box" size="0.04 0.025 0.025" density="600"/>
        </body>
        <body name="builder_right_forearm" pos="-0.18 0 0.04">
          <joint name="builder_right_push" type="slide" axis="1 0 0"
                 range="0 0.35" damping="5"/>
          <geom name="builder_fingertip" type="sphere" size="0.018" mass="0.15"/>
        </body>
      </worldbody>
      <actuator>
        <position name="push" joint="builder_right_push"
                  kp="200" kv="10" ctrlrange="0 0.35" forcerange="-15 15"/>
      </actuator>
    </mujoco>
    """
    model = audit_physics_scene(xml)
    data = mujoco.MjData(model)
    guard = BlockWriteGuard(model, data)
    guard.lock()
    return model, data, guard


def test_hand_moves_block_only_after_contact():
    model, data, guard = contact_fixture()
    qa = int(model.joint("block_0_free").qposadr[0])
    block = model.geom("block_0_geom").id
    finger = model.geom("builder_fingertip").id
    for _ in range(200):
        guard.set_ctrl([0])
        guard.step()
    origin = data.qpos[qa:qa + 2].copy()
    first_contact = None
    first_movement = None
    for step in range(750):
        guard.set_ctrl([min(0.28, step * 0.0005)])
        guard.forward()
        for contact in data.contact:
            if {int(contact.geom1), int(contact.geom2)} == {block, finger}:
                first_contact = step if first_contact is None else first_contact
        guard.step()
        displacement = np.linalg.norm(data.qpos[qa:qa + 2] - origin)
        if displacement > 1e-5 and first_movement is None:
            first_movement = step
    assert first_contact is not None
    assert first_movement is not None and first_movement >= first_contact
    assert np.linalg.norm(data.qpos[qa:qa + 2] - origin) > 0.01
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_resting_block_without_hand_contact_drifts_under_one_mm_over_two_seconds():
    model, data, guard = contact_fixture()
    qa = int(model.joint("block_0_free").qposadr[0])
    for _ in range(200):
        guard.set_ctrl([0])
        guard.step()
    initial = data.qpos[qa:qa + 3].copy()
    maximum = 0.0
    for _ in range(1000):
        guard.set_ctrl([0])
        guard.step()
        maximum = max(maximum, np.linalg.norm(data.qpos[qa:qa + 3] - initial))
    assert maximum < 0.001
    assert np.isfinite(data.qpos).all()


def test_rollout_contains_substep_states_contacts_and_no_nans():
    observations = observed_block()
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    config = PhysicsConfig()
    result = run_rollout(physics_variant(xml, config), targets, observations, config)
    for name in ("qpos", "qvel", "ctrl", "block_poses", "contact_forces"):
        assert np.isfinite(result[name]).all()
    assert not result["nan_flags"].any()
    assert len(result["timestamps"]) > len(observations["timestamps"])
    assert result["contact_offsets"][-1] == len(result["contacts"])
    assert len(result["contact_positions"]) == len(result["contact_forces"])
    np.testing.assert_allclose(np.diff(result["timestamps"]), config.timestep, atol=1e-10)


def test_reference_interpolates_free_joint_rotations_on_short_arc():
    observations = empty_observations(times=[0, 1], blocks=0)
    xml, metadata = build_scene(observations)
    model = mujoco.MjModel.from_xml_string(xml)
    targets = retarget_observations(observations, xml, metadata)
    address = int(model.joint("builder_right_root").qposadr[0])
    indices = targets["builder_qpos_indices"]
    columns = [int(np.flatnonzero(indices == address + i)[0]) for i in range(7)]
    targets["builder_qpos_targets"][0, columns] = [0, 0, 1, 1, 0, 0, 0]
    targets["builder_qpos_targets"][1, columns] = [1, 0, 1, 0, 0, 0, 1]
    midpoint = TargetInterpolator(model, targets).at(0.5)
    np.testing.assert_allclose(midpoint[address:address + 3], [0.5, 0, 1])
    np.testing.assert_allclose(
        midpoint[address + 3:address + 7], [2 ** -0.5, 0, 0, 2 ** -0.5], atol=1e-8
    )
