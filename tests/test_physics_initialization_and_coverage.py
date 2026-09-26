import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from interaction_recon.fusion.physics_prerequisites import consolidate_blocks
from interaction_recon.retarget.retarget import retarget_observations
from interaction_recon.simulation.build_scene import build_scene
from interaction_recon.simulation.controllers import (
    HumanController, ROOT_SPEED_M_S, limited_pose,
)
from interaction_recon.simulation.physics_scene import PhysicsConfig, physics_variant
from interaction_recon.simulation.rollout import (
    BlockWriteGuard, fatal_instability, initialize_blocks, run_rollout,
)
from test_scene import empty_observations


def observed_blocks(centers, count=8, fps=15):
    arrays = empty_observations(times=np.arange(count) / fps, blocks=len(centers))
    arrays["block_dimensions"][:] = [0.10, 0.05, 0.025]
    for j, center in enumerate(centers):
        arrays["block_poses"][:, j] = [*center, 0.0125, 1, 0, 0, 0]
    for name in ("block_poses", "block_dimensions"):
        arrays[f"{name}_confidence"][:] = 0.8
        arrays[f"{name}_observed_mask"][:] = True
        arrays[f"{name}_source"][:] = 2
    return arrays


def test_residual_overlap_minimally_separated_before_lock_and_settled():
    observations = observed_blocks([(0, 0), (0.08, 0)])
    report = consolidate_blocks(observations)
    assert report["block_count"] == 2
    original = observations["block_poses"].copy()
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    config = PhysicsConfig()
    xml = physics_variant(xml, config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    guard = BlockWriteGuard(model, data)
    addresses, _, records = initialize_blocks(model, guard, observations, config)
    guard.forward()
    assert max((-c.dist for c in data.contact), default=0) <= 0.001
    assert abs(data.qpos[addresses[1, 0]] - data.qpos[addresses[0, 0]]) >= 0.10
    assert any(np.linalg.norm(r["separation_shift_xy_m"]) > 0 for r in records)
    np.testing.assert_allclose(observations["block_poses"], original)

    result = run_rollout(xml, targets, observations, config)
    assert float(result["initial_max_block_penetration_m"]) <= 0.001
    assert float(result["settled_max_block_penetration_m"]) <= 0.001
    assert not result["instability_flags"].any()
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["rollout_complete"]


def test_off_table_track_dropped_before_scene_construction():
    observations = observed_blocks([(0, 0), (0, 0.49)])
    report = consolidate_blocks(observations)
    assert report["block_count"] == 1
    assert report["block_dropped_tracks"][0]["reason"] == "initial_center_outside_table"
    xml, metadata = build_scene(observations)
    assert [record["id"] for record in metadata["blocks"]] == [0]
    model = mujoco.MjModel.from_xml_string(xml)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_1") == -1


def test_falling_block_does_not_terminate_full_rollout():
    observations = observed_blocks([(0, 0)], count=6, fps=2)
    observations["builder_clip_start_s"] = np.array(0.0)
    observations["builder_clip_end_s"] = np.array(3.0)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    root = ET.fromstring(physics_variant(xml))
    # Controlled loss-of-support fixture. No block pose/velocity overwrite:
    # gravity alone causes the fall throughout preroll and the source clip.
    table = root.find("./worldbody/geom[@name='table']")
    table.set("pos", "5 5 -0.035")
    result = run_rollout(ET.tostring(root, encoding="unicode"), targets, observations)
    assert result["block_poses"][-1, 0, 2] < -10
    assert result["block_fallen_flags"][-1, 0]
    assert not result["instability_flags"].any()
    assert result["timestamps"][-1] >= 3.0 - 1e-9
    metadata = json.loads(result["metadata_json"].item())
    assert not metadata["terminated_early"]
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["stable"] and metrics["rollout_complete"]
    assert metrics["builder_clip_coverage_fraction"] == pytest.approx(1)
    assert len(metrics["block_fall_events"]) == 1
    assert not metrics["m5_pass"]


def test_rollout_extends_to_builder_end_even_when_target_tail_is_shorter():
    observations = observed_blocks([(0, 0)], count=3, fps=15)
    observations["builder_clip_start_s"] = np.array(0)
    observations["builder_clip_end_s"] = np.array(0.4)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    result = run_rollout(physics_variant(xml), targets, observations)
    assert result["timestamps"][-1] >= 0.4 - 1e-9
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["builder_clip_coverage_fraction"] == pytest.approx(1)


def test_sparse_trajectory_error_is_diagnostic_not_accepted_metric():
    observations = observed_blocks([(0, 0)], count=6)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    result = run_rollout(physics_variant(xml), targets, observations)
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["block_reference_evaluated_samples"] == 6
    assert metrics["block_reference_eligible_samples"] == 6
    assert metrics["block_reference_coverage_fraction"] == pytest.approx(1)
    assert metrics["block_reference_diagnostic_position_error_m"] is not None
    assert metrics["block_trajectory_position_error_m"] is None
    assert metrics["trajectory_target_met"] is None
    assert not metrics["block_reference_sufficient"]
    assert not metrics["m5_pass"]


def test_duplicate_analysis_samples_do_not_satisfy_trajectory_minimum():
    observations = observed_blocks([(0, 0)], count=24)
    observations["builder_source_frame_indices"] = np.repeat(np.arange(6), 4)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    result = run_rollout(physics_variant(xml), targets, observations)
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["block_reference_evaluated_samples"] == 6
    assert not metrics["block_reference_sufficient"]


def test_numerical_failure_remains_fatal():
    observations = observed_blocks([(0, 0)])
    xml, _ = build_scene(observations)
    model = mujoco.MjModel.from_xml_string(physics_variant(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert not fatal_instability(model, data, data.time)[0]
    data.qvel[0] = np.nan
    fatal, reason = fatal_instability(model, data, data.time)
    assert fatal and reason == "nonfinite_state"


def test_pose_limiter_bounds_acquisition_and_return_to_rest():
    previous = np.array([0, -0.58, 0.13, 1, 0, 0, 0], float)
    observed = np.array([0.2, 0.1, 0.03, 0, 0, 0, 1], float)
    dt = 0.002
    command = limited_pose(previous, observed, dt)
    assert np.linalg.norm(command[:3] - previous[:3]) <= ROOT_SPEED_M_S * dt + 1e-12
    angle = 2 * np.arccos(np.clip(abs(command[3:] @ previous[3:]), 0, 1))
    assert angle <= 3 * dt + 1e-9
    returned = limited_pose(command, previous, dt)
    assert np.linalg.norm(returned[:3] - command[:3]) <= ROOT_SPEED_M_S * dt + 1e-12


def test_controller_never_teleports_when_observed_targets_arrive():
    observations = empty_observations(times=[0, 0.1, 0.2, 0.3], blocks=0)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    model = mujoco.MjModel.from_xml_string(physics_variant(xml))
    data = mujoco.MjData(model)
    root_address = int(model.joint("builder_right_root").qposadr[0])
    columns = [
        int(np.flatnonzero(targets["builder_qpos_indices"] == root_address + i)[0])
        for i in range(3)
    ]
    targets["builder_qpos_targets"][2:, columns] = [0.2, 0.1, 0.04]
    controller = HumanController(model, targets, PhysicsConfig())
    guard = BlockWriteGuard(model, data)
    controller.initialize_humans(guard)
    guard.lock()
    mocap = int(model.body("target_builder_right").mocapid[0])
    park = model.qpos0[root_address:root_address + 3]
    np.testing.assert_allclose(data.mocap_pos[mocap], park)
    assert park[2] == pytest.approx(0.13)
    previous = data.mocap_pos[mocap].copy()
    for step in range(301):
        time = step * 0.002
        controller.apply(guard, time, time)
        current = data.mocap_pos[mocap].copy()
        assert np.linalg.norm(current - previous) <= ROOT_SPEED_M_S * 0.002 + 1e-10
        previous = current
