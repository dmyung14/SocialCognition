import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from interaction_recon.eval.audit import scene_check
from interaction_recon.eval.metrics import consolidate_metrics, target_table
from interaction_recon.io.media import MediaSequence, probe_media
from interaction_recon.render.comparison import compose_frame, source_frame, physics_status
from interaction_recon.render.video import convert_gif
from interaction_recon.simulation.build_scene import build_scene
from interaction_recon.simulation.physics_scene import physics_variant
from interaction_recon.simulation.rollout import run_rollout
from interaction_recon.retarget.retarget import retarget_observations
from test_no_block_actuation import observed_block


def sequence(start, color):
    times = np.array([start, start + 0.1], float)
    return MediaSequence(
        rgb_frames=np.full((2, 24, 32, 3), color, np.uint8),
        timestamps_s=times, fps=10, width=32, height=24,
        duration_s=0.2, source_timestamps_s=times, sample_times_s=times,
        source_frame_indices=np.arange(2),
    )


def test_comparison_layout_rgb_letterbox_and_missing_ranges():
    guider = sequence(2, [240, 0, 0])
    builder = sequence(4, [0, 240, 0])
    offset = 2.1
    stamp = 2.0
    frame = compose_frame(
        source_frame(guider, stamp), source_frame(builder, stamp + offset),
        np.full((24, 32, 3), [0, 0, 240], np.uint8),
        status=("sync visual_motion confidence=0.01", "PHYSICS (refined)"),
        panel_width=160, panel_height=120,
    )
    assert frame.shape == (212, 480, 3)
    np.testing.assert_array_equal(frame[80, 80], [240, 0, 0])
    np.testing.assert_array_equal(frame[80, 240], [0, 240, 0])
    np.testing.assert_array_equal(frame[80, 400], [0, 0, 240])
    assert source_frame(guider, 1.99) is None
    assert source_frame(builder, 4.2) is None
    missing = compose_frame(
        None, None, None, status=("", ""), panel_width=160, panel_height=120
    )
    assert np.count_nonzero(missing[28:148, :160] != 18) > 0
    assert "initial retained" in physics_status({})
    assert "PHYSICS (refined)" in physics_status({
        "refinement_selection_status": "refined_valid",
        "block_trajectory_position_error_m": 0.158,
        "block_trajectory_error_normalized_by_length": 0.79,
    })


def test_palette_gif_conversion_uses_bundled_ffmpeg(tiny_mp4, tmp_path):
    path = convert_gif(tiny_mp4, tmp_path / "tiny.gif")
    assert path.stat().st_size > 0
    metadata = probe_media(path)
    assert metadata.width <= 640
    assert metadata.average_fps <= 15.5
    assert metadata.duration_s == pytest.approx(0.6, abs=0.1)
    with pytest.raises(ValueError):
        convert_gif(tiny_mp4, fps=30)


def test_audit_rejects_injected_block_actuator():
    xml, _ = build_scene(observed_block())
    valid = physics_variant(xml)
    assert scene_check(valid)["pass"]
    root = ET.fromstring(valid)
    ET.SubElement(
        root.find("actuator"), "motor", name="illegal_block_drive",
        joint="block_0_free", gear="1 0 0 0 0 0",
    )
    result = scene_check(ET.tostring(root, encoding="unicode"))
    assert not result["pass"]
    assert "block" in result["detail"].lower()


def test_metrics_target_thresholds_are_closed_and_missing_never_passes():
    values = {
        "visible_hand_tracking_coverage": 0.8,
        "mean_visible_landmark_reprojection": 15,
        "block_centroid_reprojection": 20,
        "physics_block_trajectory": 0.25,
        "persistent_penetration": 0.1,
    }
    table = target_table(values)
    assert all(item["pass"] for item in table.values())
    values["physics_block_trajectory"] = 0.79
    values["visible_hand_tracking_coverage"] = None
    table = target_table(values)
    assert not table["physics_block_trajectory"]["pass"]
    assert table["physics_block_trajectory"]["threshold"] == 0.25
    assert table["visible_hand_tracking_coverage"]["status"] == "not_evaluable"
    assert not table["visible_hand_tracking_coverage"]["pass"]
    assert target_table({"persistent_penetration": np.nan})[
        "persistent_penetration"
    ]["value"] is None
    json.dumps(table, allow_nan=False)


def test_rollout_guard_evidence_and_metric_sections():
    observations = observed_block(times=np.arange(21) / 30)
    xml, metadata = build_scene(observations)
    targets = retarget_observations(observations, xml, metadata)
    arrays = run_rollout(physics_variant(xml), targets, observations)
    guard = json.loads(arrays["guard_log_json"].item())
    assert guard["locked"]
    assert guard["checks"] > guard["engine_steps"] > 0
    assert guard["violations"] == 0
    assert guard["engine_steps"] >= len(arrays["timestamps"]) - 1
    metrics = consolidate_metrics({
        "physics": json.loads(arrays["metrics_json"].item()),
        "retarget": json.loads(targets["metrics_json"].item()),
        "streams": {},
    }, observations, arrays)
    assert {"perception", "reconstruction", "physics", "targets"} <= metrics.keys()
    assert metrics["normalization"]["median_longest_block_side_m"] == pytest.approx(0.08)
    assert metrics["physics"]["max_persistent_penetration_m"] is not None
    assert not metrics["targets"]["visible_hand_tracking_coverage"]["pass"]
    json.dumps(metrics, allow_nan=False)
