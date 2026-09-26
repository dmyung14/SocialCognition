import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from interaction_recon.retarget.models import HAND_PREFIXES, RetargetConfig
from interaction_recon.retarget.retarget import retarget_observations
from interaction_recon.simulation.build_scene import build_scene


def empty_observations(times=None, blocks=2):
    times = np.arange(4, dtype=float) / 15 if times is None else np.asarray(times, float)
    count = len(times)
    result = {
        "timestamps": times,
        "table_extent_xy": np.array([1.2, 0.8]),
        "block_ids": np.arange(blocks),
    }
    shapes = {
        "guider_torso_pose": (7,), "guider_head_pose": (7,),
        "guider_arm_keypoints": (2, 3, 3),
        "builder_forearm_keypoints": (2, 2, 3),
        "block_poses": (blocks, 7), "block_dimensions": (blocks, 3),
        **{f"{prefix}_hand_21": (21, 3) for prefix in HAND_PREFIXES},
    }
    for name, shape in shapes.items():
        result[name] = np.full((count, *shape), np.nan)
        auxiliary = (count, *shape[:-1])
        result[f"{name}_confidence"] = np.zeros(auxiliary)
        result[f"{name}_observed_mask"] = np.zeros(auxiliary, bool)
        result[f"{name}_interpolated_mask"] = np.zeros(auxiliary, bool)
        result[f"{name}_source"] = np.zeros(auxiliary, np.uint8)
    return result


def test_scene_compiles_blocks_are_free_and_fingers_are_articulated(tmp_path):
    observations = empty_observations()
    observations["block_dimensions"][:, 0] = [0.08, 0.04, 0.025]
    observations["block_dimensions_confidence"][:, 0] = 0.3
    xml, metadata = build_scene(observations, tmp_path / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert model.nu == model.neq == 0
    tree = ET.fromstring(xml)
    for block in metadata["blocks"]:
        body = tree.find(f".//body[@name='{block['body']}']")
        assert len(body.findall("freejoint")) == 1
        assert not body.findall("joint")
        assert not body.findall("body")
        assert model.body(block["body"]).mocapid[0] == -1
    assert not tree.findall("actuator")
    assert not tree.findall("equality")
    assert not metadata["blocks"][0]["dimension_fallback_prior"]
    assert metadata["blocks"][1]["dimension_fallback_prior"]
    np.testing.assert_allclose(metadata["blocks"][0]["dimensions_m"], [0.08, 0.04, 0.025])
    for prefix in HAND_PREFIXES:
        finger_joints = [
            model.joint(j).name for j in range(model.njnt)
            if model.joint(j).name.startswith(prefix)
            and ("flexion" in model.joint(j).name or "abduction" in model.joint(j).name)
        ]
        assert len(finger_joints) >= 20
        assert sum(
            body.get("name", "").startswith(prefix)
            for body in tree.findall(".//body")
        ) >= 17
    assert tree.find(".//body[@name='builder_torso']") is None
    assert tree.find(".//body[@name='builder_head']") is None
    table = model.geom("table").id
    assert model.geom_pos[table, 2] + model.geom_size[table, 2] == pytest.approx(0)
    np.testing.assert_allclose(model.geom_size[table, :2] * 2, observations["table_extent_xy"])


def test_collision_masks_enable_builder_blocks_and_table_not_guider():
    xml, _ = build_scene(empty_observations())
    model = mujoco.MjModel.from_xml_string(xml)

    def can_collide(a, b):
        return bool(
            model.geom_contype[a] & model.geom_conaffinity[b]
            or model.geom_contype[b] & model.geom_conaffinity[a]
        )

    block = model.geom("block_0_geom").id
    table = model.geom("table").id
    builder_body = model.body("builder_right_palm").id
    guider_body = model.body("guider_right_palm").id
    builder = model.body_geomadr[builder_body]
    guider = model.body_geomadr[guider_body]
    assert can_collide(block, builder)
    assert can_collide(block, table)
    assert can_collide(builder, table)
    assert not can_collide(block, guider)


def test_kinematic_render_frame_and_mp4_dimensions(tmp_path):
    from interaction_recon.io.media import probe_media
    from interaction_recon.render.render import kinematic_frame, render_kinematic_reference

    observations = empty_observations(times=[0, 1 / 15], blocks=0)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    config = RetargetConfig(width=320, height=240)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, width=320, height=240) as renderer:
        frame = kinematic_frame(model, data, renderer, targets, observations, 0)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 5
    path = tmp_path / "kinematic.mp4"
    count = render_kinematic_reference(path, xml, targets, observations, config)
    media = probe_media(path)
    assert (media.width, media.height) == (320, 240)
    assert media.fps == pytest.approx(30)
    assert media.frame_count == count == 4
