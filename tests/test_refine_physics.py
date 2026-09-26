from dataclasses import replace
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from interaction_recon.retarget.retarget import retarget_observations
from interaction_recon.simulation.build_scene import build_scene
from interaction_recon.simulation.physics_metrics import (
    box_rotation_errors, contact_chain_lengths, movement_attribution,
)
from interaction_recon.simulation.physics_scene import PhysicsConfig, physics_variant
from interaction_recon.simulation.refine_physics import (
    RefineConfig, bounded_cma_es, decode_parameters, initial_parameters,
    motion_window, refine_physics,
)
from interaction_recon.simulation.rollout import BlockWriteGuard, run_rollout
from test_no_block_actuation import observed_block


def test_bounded_cma_finds_quadratic_minimum_deterministically():
    optimum = np.array([0.17, 0.62, 0.88])
    calls = []

    def objective(x):
        assert np.all((x >= 0) & (x <= 1))
        calls.append(x.copy())
        return float(np.sum([1, 3, 7] * (x - optimum) ** 2))

    history = bounded_cma_es(
        objective, np.full(3, 0.5), max_evaluations=700, seed=17
    )
    best, loss = min(history, key=lambda item: item[1])
    np.testing.assert_allclose(best, optimum, atol=0.002)
    assert loss < 1e-5
    repeated = bounded_cma_es(
        objective, np.full(3, 0.5), max_evaluations=700, seed=17
    )
    np.testing.assert_array_equal(
        np.array([x for x, _ in history]), np.array([x for x, _ in repeated])
    )


def test_optimizer_honors_expired_deadline():
    def forbidden(x):
        raise AssertionError("Must not evaluate after deadline")

    assert bounded_cma_es(
        forbidden, np.array([0.5]), deadline=time.monotonic() - 1
    ) == []


def chain_fixture():
    times = np.array([0.0, 0.05, 0.10, 0.21, 0.31, 0.50])
    poses = np.tile([0., 0, 0.025, 1, 0, 0, 0], (len(times), 3, 1))
    poses[2:, 1, 1] = 0.02
    poses[4:, 1, 1] = 0.04
    # hand->A at t=0; A->B at .05; B->world and world->C never connect C.
    pairs = np.array([[0, 1], [1, 2], [2, 4], [4, 3]], int)
    direct = np.zeros((len(times), 3), bool)
    direct[0, 0] = True
    return {
        "timestamps": times, "block_ids": np.array([10, 31, 99]),
        "block_poses": poses,
        "contacts": pairs, "contact_offsets": np.array([0, 1, 4, 4, 4, 4, 4]),
        "geom_body_names": np.array([
            "builder_right_palm", "block_10", "block_31", "block_99", "world",
        ]),
        "geom_names": np.array(["hand", "A", "B", "C", "table"]),
        "hand_block_contact": direct,
    }


def test_transitive_contact_attribution_expires_and_world_does_not_bridge():
    arrays = chain_fixture()
    chains = contact_chain_lengths(arrays)
    assert chains[2, 0] == 1
    assert chains[2, 1] == 2
    assert np.all(chains[:, 2] == -1)
    assert chains[3, 1] == -1
    report = movement_attribution(arrays)
    assert report["events"][0]["contact_chain_length"] == 2
    assert not report["events"][0]["preceding_0_2s_hand_contact"]
    assert report["events"][0]["preceding_0_2s_hand_contact_chain"]
    assert len(report["violations"]) == 1
    assert report["violations"][0]["timestamp_s"] == pytest.approx(0.31)
    assert report["chain_length_stats"]["transitive_events"] == 1
    assert report["chain_length_stats"]["maximum"] == 2


def test_forearm_and_future_contacts_do_not_authorize_motion():
    arrays = chain_fixture()
    arrays["geom_body_names"] = arrays["geom_body_names"].astype("<U64")
    arrays["geom_body_names"][0] = "builder_right_forearm"
    arrays["geom_names"][0] = "forearm"
    arrays["hand_block_contact"][:] = False
    assert np.all(contact_chain_lengths(arrays) == -1)
    report = movement_attribution(arrays)
    assert report["status"] == "failed"
    assert [event["timestamp_s"] for event in report["violations"]] == pytest.approx(
        [0.10, 0.31]
    )
    assert all(
        not event["preceding_0_2s_hand_contact_chain"]
        for event in report["events"]
    )

    arrays = chain_fixture()
    arrays["block_poses"][0, 1, 1] = -0.02

    # The original fixture already connects hand->A->B at .05. Its inclusive
    # [t-.2,t] policy correctly authorizes motion at .05; that was not a
    # future-contact test. Move the solver evidence itself to .10.
    assert contact_chain_lengths(arrays)[1, 1] == 2
    arrays["contact_offsets"] = np.array([0, 0, 0, 4, 4, 4, 4])
    arrays["hand_block_contact"][:] = False
    arrays["hand_block_contact"][2, 0] = True

    chains = contact_chain_lengths(arrays)
    assert np.all(chains[:2] == -1)
    np.testing.assert_array_equal(chains[2], [1, 2, -1])
    report = movement_attribution(arrays)
    assert report["status"] == "failed"
    assert [event["timestamp_s"] for event in report["violations"]] == pytest.approx(
        [0.05, 0.31]
    )
    early = report["violations"][0]
    assert early["block_id"] == 31
    assert early["contact_chain_length"] is None
    assert not early["preceding_0_2s_hand_contact"]
    assert not early["preceding_0_2s_hand_contact_chain"]
    contemporaneous = next(
        event for event in report["events"]
        if event["timestamp_s"] == pytest.approx(0.10)
    )
    assert contemporaneous["contact_chain_length"] == 2


def test_cuboid_symmetry_metric_keeps_raw_diagnostic():
    reduced, raw = box_rotation_errors([0, 0, 0, 1], [1, 0, 0, 0])
    assert reduced == pytest.approx(0)
    assert raw == pytest.approx(np.pi)
    reduced, _ = box_rotation_errors(
        [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], [1, 0, 0, 0]
    )
    # Unequal box axes must not acquire an invented 90-degree symmetry.
    assert reduced == pytest.approx(np.pi / 2)


def test_parameters_stay_in_bounds_and_baseline_roundtrips():
    base = PhysicsConfig()
    cfg, _ = decode_parameters(initial_parameters(base), base)
    assert cfg.density == pytest.approx(base.density)
    assert cfg.weld_timeconst == pytest.approx(base.weld_timeconst)
    assert cfg.finger_kp == pytest.approx(base.finger_kp)
    assert cfg.hand_offset_m == pytest.approx((0, 0, 0))
    rng = np.random.default_rng(4)
    for x in [np.zeros(12), np.ones(12), *rng.random((100, 12))]:
        cfg, params = decode_parameters(x, base, timestep=0.004)
        assert 300 <= cfg.density <= 1200
        assert 0.3 <= cfg.friction <= 1.2
        assert 0.5 <= cfg.hand_friction <= 1.5
        assert 0.005 <= cfg.contact_timeconst <= 0.03
        assert np.linalg.norm(cfg.hand_offset_m) <= 0.03 + 1e-12
        assert -0.15 <= cfg.timing_offset_s <= 0.15
        for name in (
            "weld_timeconst_multiplier", "wrist_gain_multiplier",
            "finger_kp_multiplier", "finger_force_multiplier",
        ):
            assert 0.5 <= params[name] <= 2
    with pytest.raises(ValueError, match="twelve"):
        decode_parameters(np.full(12, 1.1), base)


def test_pair_friction_is_not_masked_by_mujoco_maximum_rule():
    observations = observed_block()
    xml, _ = build_scene(observations)
    config = PhysicsConfig(friction=0.35, hand_friction=1.4)
    model = mujoco.MjModel.from_xml_string(physics_variant(xml, config))
    block, table = model.geom("block_0_geom").id, model.geom("table").id
    hand = model.body_geomadr[model.body("builder_right_palm").id]
    assert model.geom_priority[table] > model.geom_priority[block]
    assert model.geom_priority[hand] > model.geom_priority[block]
    assert model.geom_friction[table, 0] == pytest.approx(0.35)
    assert model.geom_friction[hand, 0] == pytest.approx(1.4)


def test_motion_window_uses_observed_builder_motion():
    observations = observed_block(times=np.arange(60) / 15)
    observations["block_poses"][30:, 0, 0] = 0.1
    observations["builder_clip_start_s"] = np.array(0)
    observations["builder_clip_end_s"] = np.array(4)
    left, right = motion_window(observations, 0.8)
    assert left <= 2 <= right
    observations["block_poses_observed_mask"][30:] = False
    left, _ = motion_window(observations, 0.8)
    assert left == pytest.approx(0)


PUSH_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <default><geom friction="0.6 0.01 0.001" solref="0.008 1"/></default>
  <worldbody>
    <geom name="table" type="box" size="2 1 .05" pos="0 0 -.05"
          priority="2"/>
    <body name="block_0" pos="0 0 .0252">
      <freejoint name="block_0_free"/>
      <geom name="block_0_geom" type="box" size=".04 .025 .025" density="600"/>
    </body>
    <body name="builder_right_palm" pos="-.10 0 .025">
      <joint name="builder_push" type="slide" axis="1 0 0" range="-.3 .5"
             damping="2" armature=".001"/>
      <geom name="builder_fingertip" type="sphere" size=".018" mass=".2"
            priority="3" friction="1 .01 .001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="push" joint="builder_push" kp="600" kv="15"
              ctrlrange="-.3 .5" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""


def tiny_push(friction):
    root = ET.fromstring(PUSH_XML)
    root.find("./worldbody/geom[@name='table']").set(
        "friction", f"{friction} .01 .001"
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    guard = BlockWriteGuard(model, data)
    guard.lock()
    qa = int(model.joint("block_0_free").qposadr[0])
    for _ in range(150):
        guard.set_ctrl([0])
        guard.step()
    values = []
    contacts = 0
    block_geom = model.geom("block_0_geom").id
    hand_geom = model.geom("builder_fingertip").id
    for step in range(500):
        stamp = step * 0.002
        # A short push followed by withdrawal identifies table friction from
        # the free sliding deceleration, rather than a prescribed block path.
        command = min(0.22, stamp * 0.8) if stamp < 0.30 else -0.20
        guard.set_ctrl([command])
        guard.step()
        contacts += sum(
            {int(c.geom1), int(c.geom2)} == {block_geom, hand_geom}
            for c in data.contact
        )
        values.append(data.qpos[qa:qa + 3].copy())
    guard.check()
    return np.array(values), contacts


def test_synthetic_contact_inverse_problem_recovers_friction_without_block_writes(monkeypatch):
    original = BlockWriteGuard._write
    protected_writes = []

    def checked(self, name, indices, values):
        addresses = np.arange(getattr(self._data, name).size)[indices]
        protected = (
            self.block_qpos_addresses if name == "qpos" else self.block_qvel_addresses
        )
        if self.locked and np.isin(addresses, protected).any():
            protected_writes.append(name)
        return original(self, name, indices, values)

    monkeypatch.setattr(BlockWriteGuard, "_write", checked)
    true_friction = 0.47
    reference, contacts = tiny_push(true_friction)
    assert contacts > 0
    assert reference[-1, 0] - reference[0, 0] > 0.01
    original_reference = reference.copy()
    reference.setflags(write=False)

    def loss(x):
        assert x.shape == (1,)
        assert np.all((x >= 0) & (x <= 1))
        trajectory, _ = tiny_push(0.3 + 0.9 * x[0])
        assert np.isfinite(trajectory).all()
        return float(np.mean(np.linalg.norm(trajectory - reference, axis=1)) / 0.08)

    initial = loss(np.array([1.0]))
    assert initial > 0

    # A single 90-evaluation CMA run is not a global optimizer guarantee on
    # this contact-switching objective: it can converge to a different basin.
    # Cover the entire scalar domain with fixed strata, then run the same
    # bounded optimizer within each. Neither starts nor bounds use the truth.
    # Keep the physical experiment, recovery tolerance and loss requirement.
    edges = np.linspace(0.0, 1.0, 9)
    history = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        def local_loss(x, low=low, high=high):
            return loss(low + (high - low) * x)

        local_history = bounded_cma_es(
            local_loss, np.array([0.5]),
            max_evaluations=90, seed=7 + index, sigma=0.3,
        )
        assert len(local_history) == 90
        history.extend(
            (low + (high - low) * x, score)
            for x, score in local_history
        )

    best, refined_loss = min(history, key=lambda item: item[1])
    recovered = 0.3 + 0.9 * best[0]
    # Contact switching makes friction->trajectory non-monotonic (0.4266 fits within
    # 0.6 mm while 0.45 is 3.5 mm off), so identify by motion, bound the parameter loosely.
    recovered_trajectory, _ = tiny_push(recovered)
    rmse = np.sqrt(np.mean(np.sum((recovered_trajectory - reference) ** 2, axis=1)))
    assert rmse < 0.001
    assert recovered == pytest.approx(true_friction, abs=0.06)
    assert refined_loss < initial * 0.2
    assert loss(best) == pytest.approx(refined_loss, rel=1e-10, abs=1e-12)
    np.testing.assert_array_equal(reference, original_reference)
    assert not protected_writes


def test_full_refinement_keeps_initial_and_candidate_guard_enabled(monkeypatch):
    observations = observed_block(times=np.arange(21) / 30)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    base = PhysicsConfig()
    initial = run_rollout(physics_variant(xml, base), targets, observations, base)
    initial["physics_compute_s"] = np.array(0.2)
    original_pose = initial["block_poses"].copy()
    original_observations = observations["block_poses"].copy()
    original = BlockWriteGuard._write
    calls = []

    def checked(self, name, indices, values):
        addresses = np.arange(getattr(self._data, name).size)[indices]
        protected = (
            self.block_qpos_addresses if name == "qpos" else self.block_qvel_addresses
        )
        if self.locked:
            assert not np.isin(addresses, protected).any()
            calls.append(name)
        return original(self, name, indices, values)

    monkeypatch.setattr(BlockWriteGuard, "_write", checked)
    result = refine_physics(
        xml, targets, observations, initial, base,
        RefineConfig(coarse_evaluations=3, top_k=1, time_budget_s=30),
    )
    report = json.loads(result["refinement_json"].item())
    assert report["history"][0]["phase"] == "initial_full"
    assert report["coarse_completed"] >= 1
    assert calls
    assert all(
        entry["metrics"]["block_write_guard"] == "enabled throughout pre-roll and rollout"
        for entry in report["history"] if "metrics" in entry
    )
    np.testing.assert_array_equal(initial["block_poses"], original_pose)
    np.testing.assert_array_equal(observations["block_poses"], original_observations)
    assert "refinement_improvement" in report


def test_refine_stage_cache_changes_only_with_refine_configuration(tmp_path, monkeypatch):
    from interaction_recon import stages_physics

    observations = observed_block(times=np.arange(21) / 30)
    observations["block_consolidation_json"] = np.array("{}")
    observations["metrics_json"] = np.array("{}")
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    np.savez_compressed(tmp_path / "observations.npz", **observations)
    np.savez_compressed(tmp_path / "retargeted.npz", **targets)

    def render(path, *args):
        path.write_bytes(b"test movie")
        return 1

    monkeypatch.setattr(stages_physics.physics_render, "render_physics", render)
    cfg = RefineConfig(coarse_evaluations=2, top_k=1, time_budget_s=20)
    first = stages_physics.process_physics(tmp_path, refine_config=cfg)
    assert first["artifacts"]["refinement.json"]["status"] == "written"

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached refinement must not run a candidate")

    monkeypatch.setattr(stages_physics.refine_physics, "refine_physics", forbidden)
    monkeypatch.setattr(stages_physics.rollout, "run_rollout", forbidden)
    second = stages_physics.process_physics(tmp_path, refine_config=cfg)
    assert second["artifacts"]["refinement.json"]["status"] == "reused"
    assert second["artifacts"]["simulation.mp4"]["status"] == "reused"
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert "physics_initial" in saved and "physics_refined" in saved
    assert (tmp_path / "physics_initial.npz").is_file()
    with pytest.raises(AssertionError, match="Cached refinement"):
        stages_physics.process_physics(tmp_path, refine_config=replace(cfg, seed=cfg.seed + 1))
