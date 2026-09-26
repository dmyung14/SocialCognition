import json

import mujoco
import numpy as np

from interaction_recon.retarget.arm_ik import forward_kinematics, joint_addresses, solve_ik
from interaction_recon.retarget.hand_ik import hand_angles
from interaction_recon.retarget.models import (
    FINGERS, HAND_PREFIXES, finger_joint_names,
)
from interaction_recon.retarget.retarget import (
    actor_qpos_indices, neutral_qpos, retarget_observations,
)
from interaction_recon.simulation.build_scene import build_scene
from test_scene import empty_observations


def test_arm_dls_recovers_random_reachable_poses_below_one_cm():
    xml, _ = build_scene(empty_observations(blocks=0))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(730)
    for side in ("left", "right"):
        prefix = f"guider_{side}"
        names = [*(f"{prefix}_shoulder_{a}" for a in "xyz"), f"{prefix}_elbow"]
        qa, _ = joint_addresses(model, names)
        for _ in range(8):
            data.qpos[:] = neutral_qpos(model)
            data.qpos[qa] = [
                rng.uniform(-1.8, -0.7), rng.uniform(-0.45, 0.45),
                rng.uniform(-0.6, 0.6), rng.uniform(0.35, 1.7),
            ]
            forward_kinematics(model, data)
            targets = {
                f"{prefix}_{name}": data.site_xpos[model.site(f"{prefix}_{name}").id].copy()
                for name in ("shoulder", "elbow", "wrist")
            }
            data.qpos[:] = neutral_qpos(model)
            result = solve_ik(
                model, data, names, targets, iterations=120,
                damping=0.006, regularizer=1e-7,
            )
            assert np.max(result.position_residuals_m) < 0.01


def test_hand_angles_recover_independent_mujoco_fk_within_ten_degrees():
    xml, metadata = build_scene(empty_observations(blocks=0))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(411)
    for prefix in HAND_PREFIXES:
        side = prefix.split("_")[1]
        expected = np.column_stack((
            rng.uniform(-0.2, 0.2, 5),
            rng.uniform(0.15, 0.8, 5),
            rng.uniform(0.15, 1.2, 5),
            rng.uniform(0.10, 0.9, 5),
        ))
        data.qpos[:] = neutral_qpos(model)
        for f, finger in enumerate(FINGERS):
            qa, _ = joint_addresses(model, finger_joint_names(prefix, finger))
            data.qpos[qa] = expected[f]
        forward_kinematics(model, data)
        landmarks = np.array([
            data.site_xpos[model.site(f"{prefix}_landmark_{k}").id].copy()
            for k in range(21)
        ])
        recovered, valid = hand_angles(
            landmarks, side,
            metadata["anthropometry"]["hands"][prefix]["thumb_yaw"],
            regularizer=0,
        )
        assert valid.all()
        assert np.max(np.abs(recovered - expected)) < np.deg2rad(10)


def test_missing_chains_hold_then_rest_with_decaying_confidence():
    observations = empty_observations(times=[0, 0.2, 0.5, 0.6], blocks=0)
    name = "guider_torso_pose"
    observations[name][0] = [0.1, 0.65, 0.18, 1, 0, 0, 0]
    observations[f"{name}_confidence"][0] = 0.6
    observations[f"{name}_observed_mask"][0] = True
    observations[f"{name}_source"][0] = 1
    xml, metadata = build_scene(observations)
    result = retarget_observations(observations, xml, metadata)
    column = list(result["chain_names"]).index("guider_root")
    np.testing.assert_array_equal(result["validity_mask"][:, column], [1, 0, 0, 0])
    np.testing.assert_array_equal(result["held_mask"][:, column], [0, 1, 1, 0])
    np.testing.assert_array_equal(result["rest_mask"][:, column], [0, 0, 0, 1])
    confidence = result["confidence"][:, column]
    assert confidence[0] > confidence[1] > confidence[2] > confidence[3] == 0
    assert np.isnan(result["ik_residuals_m"][1:, column]).all()
    metrics = json.loads(result["metrics_json"].item())
    assert metrics["joint_limit_violations"] == 0


def test_no_block_coordinates_in_human_target_artifact():
    observations = empty_observations()
    xml, metadata = build_scene(observations)
    result = retarget_observations(observations, xml, metadata)
    model = mujoco.MjModel.from_xml_string(xml)
    human = np.r_[
        result["guider_qpos_indices"], result["builder_qpos_indices"]
    ]
    for identity in observations["block_ids"]:
        address = model.joint(f"block_{identity}_free").qposadr[0]
        assert not np.isin(np.arange(address, address + 7), human).any()
    assert all(not name.startswith("block_") for name in result["joint_names"])
    for actor in ("guider", "builder"):
        np.testing.assert_array_equal(
            result[f"{actor}_qpos_indices"], actor_qpos_indices(model, actor)
        )
        assert np.isfinite(result[f"{actor}_qpos_targets"]).all()


def test_builder_hand_only_root_keeps_wrist_position_and_flags_prior():
    observations = empty_observations(times=[0, 0.1], blocks=0)
    xml, metadata = build_scene(observations)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = neutral_qpos(model)
    forward_kinematics(model, data)
    prefix = "builder_right"
    points = np.array([
        data.site_xpos[model.site(f"{prefix}_landmark_{k}").id].copy()
        for k in range(21)
    ])
    name = f"{prefix}_hand_21"
    observations[name][:] = points
    observations[f"{name}_confidence"][:] = 0.3
    observations[f"{name}_observed_mask"][:] = True
    observations[f"{name}_source"][:] = 2
    result = retarget_observations(observations, xml, metadata)
    channel = list(result["chain_names"]).index("builder_right_root")
    assert result["validity_mask"][:, channel].all()
    assert not result["observed_mask"][:, channel].any()
    assert np.all(result["method"][:, channel] == "hand_axis_forearm_prior")
    wrist_index = list(result["wrist_names"]).index("builder_right_wrist")
    np.testing.assert_allclose(
        result["wrist_poses"][:, wrist_index, :3],
        np.repeat(points[None, 0], 2, axis=0), atol=1e-7,
    )


def test_retarget_cache_reuses_without_recomputing_or_rendering(tmp_path, monkeypatch):
    from interaction_recon import stages_retarget

    observations = empty_observations(times=[0, 0.1], blocks=0)
    np.savez_compressed(tmp_path / "observations.npz", **observations)
    first = stages_retarget.process_retarget(tmp_path)
    assert first["artifacts"]["retargeted.npz"]["status"] == "written"

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached M4 must not recompute IK or rendering")

    monkeypatch.setattr(stages_retarget.retarget, "retarget_observations", forbidden)
    monkeypatch.setattr(stages_retarget.render, "render_kinematic_reference", forbidden)
    second = stages_retarget.process_retarget(tmp_path)
    for artifact in ("retargeted.npz", "scene.xml", "kinematic_reference.mp4"):
        assert second["artifacts"][artifact]["status"] == "reused"
