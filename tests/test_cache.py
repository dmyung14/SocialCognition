import json

import numpy as np
import pytest

from interaction_recon.config import RunOptions
from interaction_recon.io.cache import cache_key, cached_arrays
from interaction_recon.pipeline import reconstruct_inventory
from interaction_recon.vision.detectors import MODEL_FILES, require_models
from interaction_recon.vision.head import HeadObservation


def test_cache_reuse_configuration_invalidation_and_force(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"values": np.arange(4)}

    key = cache_key({"input_sha256": "abc", "fps": 15})
    first, reused, path = cached_arrays(tmp_path, "stage", key, compute)
    assert not reused
    second, reused, same_path = cached_arrays(tmp_path, "stage", key, compute)
    assert reused and same_path == path
    np.testing.assert_array_equal(first["values"], second["values"])
    assert len(calls) == 1
    cached_arrays(tmp_path, "stage", key, compute, force=True)
    assert len(calls) == 2
    different = cache_key({"input_sha256": "abc", "fps": 20})
    cached_arrays(tmp_path, "stage", different, compute)
    assert len(calls) == 3


def test_missing_models_are_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="Missing MediaPipe Tasks model"):
        require_models(tmp_path)


class EmptyBody:
    def detect(self, rgb, timestamp_ms):
        result = np.full((33, 4), np.nan, np.float32)
        result[:, 3] = 0
        return result

    def close(self):
        pass


class EmptyHands:
    def detect(self, rgb, timestamp_ms):
        return []

    def close(self):
        pass


class EmptyHead:
    def detect(self, rgb, timestamp_ms):
        return HeadObservation(
            np.array([np.nan, np.nan, np.nan, 0], np.float32),
            np.full((4, 4), np.nan, np.float32), 0,
        )

    def close(self):
        pass


def test_pair_pipeline_cache_reuse_without_loading_media_again(
    tmp_path, tiny_mp4, monkeypatch
):
    import interaction_recon.stages as stages
    import interaction_recon.vision.tracking as tracking

    models = tmp_path / "models"
    models.mkdir()
    for filename in MODEL_FILES.values():
        (models / filename).write_bytes(b"fake models for injected detectors")
    calls = []

    def factory(paths):
        calls.append(paths)
        return EmptyBody(), EmptyHands(), EmptyHead()

    monkeypatch.setattr(tracking, "create_detectors", factory)
    builder = tmp_path / "builder.mp4"
    builder.write_bytes(tiny_mp4.read_bytes())
    output = tmp_path / "output"
    options = RunOptions(model_dir=str(models))
    manifest, code = reconstruct_inventory(tiny_mp4, builder, output, options)
    assert code == 0
    assert manifest["stage_status"] == "tracking_complete"
    assert manifest["retarget_stage_status"] == "complete"
    assert manifest["physics_stage_status"] == "complete"
    assert manifest["simulation_performed"]
    assert not manifest["m5_pass"]
    assert manifest["kinematic_reference_performed"]
    assert len(calls) == 2
    assert (output / "tracking.mp4").stat().st_size > 0
    assert (output / "sync.json").exists()
    assert len(list((output / "cache").glob("*-2d-*.npz"))) == 2
    for name in (
        "scene.xml", "retargeted.npz", "kinematic_reference.mp4",
        "physics_rollout.npz", "simulation.mp4",
    ):
        assert (output / name).stat().st_size > 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached run must not decode media or rerun tracking")

    monkeypatch.setattr(stages, "load_media", forbidden)
    monkeypatch.setattr(tracking, "create_detectors", forbidden)
    manifest, code = reconstruct_inventory(tiny_mp4, builder, output, options)
    assert code == 0
    for name in (
        "tracking.mp4", "sync.json", "scene.xml",
        "retargeted.npz", "kinematic_reference.mp4",
        "physics_rollout.npz", "simulation.mp4",
    ):
        assert manifest["artifacts"][name]["status"] == "reused"
    cached_observations = [
        value for value in manifest["artifacts"].values()
        if value.get("kind") == "per_stream_2d_observations"
    ]
    assert len(cached_observations) == 2
    assert all(value["status"] == "reused" for value in cached_observations)
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["cross_view_consistency"]["status"] == "not_estimated"


def test_pair_missing_models_writes_failure_manifest(tmp_path, tiny_mp4):
    builder = tmp_path / "builder.mp4"
    builder.write_bytes(tiny_mp4.read_bytes())
    manifest, code = reconstruct_inventory(
        tiny_mp4, builder, tmp_path / "output",
        RunOptions(model_dir=str(tmp_path / "missing")),
    )
    assert code == 1
    assert manifest["stage_status"] == "tracking_failed"
    assert "Missing MediaPipe Tasks model" in manifest["error"]
    assert manifest["artifacts"]["tracking.mp4"]["status"] == "failed"
    assert manifest["artifacts"]["physics_rollout.npz"]["status"] == "failed"
