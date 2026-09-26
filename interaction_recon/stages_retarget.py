from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

import mujoco
import numpy as np

from interaction_recon.io.cache import cache_key, cached_arrays, file_hash, write_json
from interaction_recon.render import render
from interaction_recon.retarget import arm_ik, hand_ik, models, retarget
from interaction_recon.simulation import build_scene


def _array_digest(arrays: dict) -> str:
    """Content hash, independent of ZIP timestamps and NPZ export metadata."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _export(path: Path, arrays: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_retarget(
    output: Path, *, force: bool = False,
    config: models.RetargetConfig = models.RetargetConfig(),
) -> dict:
    started = time.perf_counter()
    with np.load(output / "observations.npz", allow_pickle=False) as archive:
        observations = {name: archive[name].copy() for name in archive.files}
    code = {
        module.__name__: file_hash(module.__file__)
        for module in (models, arm_ik, hand_ik, retarget, build_scene)
    }
    key = cache_key({
        "stage": "retarget_m4_v1", "observations": _array_digest(observations),
        "config": asdict(config), "code": code,
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "stage_code": file_hash(__file__),
    })
    cache = output / "cache"

    def compute():
        start = time.perf_counter()
        xml, metadata = build_scene.build_scene(observations)
        targets = retarget.retarget_observations(observations, xml, metadata, config)
        targets["scene_xml"] = np.array(xml)
        targets["scene_metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
        targets["retarget_compute_s"] = np.array(time.perf_counter() - start)
        return targets

    arrays, reused, cache_path = cached_arrays(
        cache, "retarget", key, compute, force=force
    )
    xml = arrays["scene_xml"].item()
    scene_path = output / "scene.xml"
    temporary = scene_path.with_name(f".scene-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(xml + "\n", encoding="utf-8")
        temporary.replace(scene_path)
    finally:
        temporary.unlink(missing_ok=True)
    _export(output / "retargeted.npz", {
        name: value for name, value in arrays.items() if name != "scene_xml"
    })
    artifacts = {
        name: {"status": "reused" if reused else "written", "cache_key": key}
        for name in ("scene.xml", "retargeted.npz", cache_path.relative_to(output).as_posix())
    }
    render_key = cache_key({
        "retarget": key, "render_code": file_hash(render.__file__),
        "config": asdict(config), "mujoco": mujoco.__version__,
    })
    movie = output / "kinematic_reference.mp4"
    receipt = cache / "kinematic_reference.json"
    render_reused = False
    if not force and movie.is_file() and receipt.is_file():
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            render_reused = (
                saved["cache_key"] == render_key and saved["sha256"] == file_hash(movie)
            )
        except (OSError, ValueError, KeyError):
            pass
    if not render_reused:
        render_started = time.perf_counter()
        frames = render.render_kinematic_reference(movie, xml, arrays, observations, config)
        write_json(receipt, {
            "cache_key": render_key, "sha256": file_hash(movie),
            "frames": frames, "fps": config.render_fps,
            "render_s": time.perf_counter() - render_started,
        })
    artifacts["kinematic_reference.mp4"] = {
        "status": "reused" if render_reused else "written", "cache_key": render_key,
        "scope": "kinematic_only_not_physics",
    }
    metrics = json.loads(arrays["metrics_json"].item())
    metrics.update({
        "scene": json.loads(arrays["scene_metadata_json"].item()),
        "retarget_compute_s": float(arrays["retarget_compute_s"]),
        "stage_wall_s_this_run": time.perf_counter() - started,
        "cache_reused": reused, "render_cache_reused": render_reused,
        "runtime_target_s": 120,
        "runtime_target_met_this_run": time.perf_counter() - started < 120,
    })
    return {
        "artifacts": artifacts, "metrics": metrics,
        "warnings": [
            "M4 is kinematic playback, not a contact-driven physics reconstruction.",
            "Missing human chains use flagged hold/rest poses; missing blocks are hidden.",
            "IK residuals include anthropometric mismatch; visual M4 acceptance needs inspection.",
        ],
    }
