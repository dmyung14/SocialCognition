import json
from pathlib import Path
import uuid

import numpy as np

from interaction_recon.fusion import (
    block_geometry, filtering, physics_prerequisites, table_geometry,
    triangulation, world,
)
from interaction_recon.io.cache import cache_key, cached_arrays, file_hash, write_json
from interaction_recon.render import world_preview
from interaction_recon.vision import camera


def _code_key(*modules):
    return {module.__name__: file_hash(module.__file__) for module in modules}


def _export(path: Path, arrays: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_world(
    output: Path, observations: dict, observation_keys: dict,
    sequence, sync: dict, options, versions: dict,
) -> dict:
    cameras, camera_keys, artifacts = {}, {}, {}
    cache = output / "cache"
    for role in ("guider", "builder"):
        key = cache_key({
            "stage": "camera_world_v2",
            "observations": observation_keys[role],
            "role": role,
            "code": _code_key(camera, triangulation, world, table_geometry),
            "versions": {name: versions[name] for name in ("numpy", "scipy", "opencv")},
        })

        def compute(role=role):
            media = sequence(role)
            model = camera.estimate_intrinsics(media.rgb_frames, role)
            prior = world.metric_depth_prior(observations[role], model)
            result = camera.estimate_camera(
                media, observations[role], role, prior["depth_m"]
            )
            result["metric_scale_prior_json"] = np.array(json.dumps(prior, allow_nan=False))
            result["metric_depth_prior_m"] = np.array(prior["depth_m"])
            result["metric_depth_sigma_m"] = np.array(prior["depth_sigma_m"])
            result["scale_confidence"] = np.array(prior["confidence"])
            return result

        arrays, reused, path = cached_arrays(
            cache, f"{role}-camera", key, compute, force=options.force
        )
        cameras[role], camera_keys[role] = arrays, key
        artifacts[path.relative_to(output).as_posix()] = {
            "status": "reused" if reused else "written", "cache_key": key,
            "kind": "moving_camera_prior",
        }

    key = cache_key({
        "stage": "common_world_m5_prerequisites_v1",
        "observations": observation_keys, "cameras": camera_keys,
        "sync": sync, "fps": options.analysis_fps,
        "minimum_block_supported_frames": 5,
        "code": _code_key(
            world, triangulation, filtering, table_geometry, block_geometry,
            physics_prerequisites,
        ),
    })

    def compute_world():
        result = world.reconstruct_world(
            observations, cameras, sync, options.analysis_fps
        )
        return physics_prerequisites.prepare_physics_observations(
            result, observations["builder"], cameras["builder"], sync
        )

    arrays, reused, path = cached_arrays(
        cache, "world", key, compute_world, force=options.force
    )
    world.validate_observations(arrays)
    artifacts[path.relative_to(output).as_posix()] = {
        "status": "reused" if reused else "written", "cache_key": key,
    }
    _export(output / "observations.npz", arrays)
    camera_output = {}
    for role, values in cameras.items():
        camera_output.update({f"{role}_{name}": value for name, value in values.items()})
        camera_output[f"{role}_T_world_from_camera"] = arrays[
            f"{role}_T_world_from_camera_source"
        ]
        for suffix in (
            "table_from_initial", "layout_transform", "table_plane_initial",
            "table_extent_xy", "table_geometry_json", "table_candidate_normals_camera",
            "table_candidate_frames", "table_candidate_weights",
            "table_candidate_methods", "table_candidate_inliers",
        ):
            camera_output[f"{role}_{suffix}"] = arrays[f"{role}_{suffix}"]
        camera_output[f"{role}_table_homography_normal_candidates_initial"] = values[
            "table_normal_initial"
        ]
        camera_output[f"{role}_table_homography_normal_candidate_confidence"] = values[
            "table_normal_confidence"
        ]
        plane = arrays[f"{role}_table_plane_initial"]
        report = json.loads(arrays[f"{role}_table_geometry_json"].item())
        camera_output[f"{role}_table_normal_initial"] = plane[:3].copy()
        camera_output[f"{role}_table_normal_confidence"] = np.array(report["confidence"])
        camera_output[f"{role}_table_height_m"] = np.array(report["height_m"])
        camera_output[f"{role}_table_height_sigma_m"] = np.array(report["height_sigma_m"])

    camera_output["fusion_method"] = arrays["fusion_method"]
    camera_output["verified_fusion"] = np.array(False)
    camera_output["table_geometry_schema"] = np.array("one_initial_frame_plane_per_stream_v2")
    _export(output / "cameras.npz", camera_output)
    for name in ("cameras.npz", "observations.npz"):
        artifacts[name] = {
            "status": "reused" if reused else "written", "cache_key": key,
            "note": "Export of keyed stage cache; metric estimates retain uncertainty.",
        }

    render_key = cache_key({
        "world": key, "code": _code_key(world_preview), "fps": options.analysis_fps,
    })
    preview, receipt = output / "world_preview.mp4", cache / "world_preview.json"
    render_reused = False
    if not options.force and preview.is_file() and receipt.is_file():
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            render_reused = (
                saved["cache_key"] == render_key and saved["sha256"] == file_hash(preview)
            )
        except (OSError, ValueError, KeyError):
            pass
    if not render_reused:
        world_preview.render_world_preview(preview, arrays, options.analysis_fps)
        write_json(receipt, {"cache_key": render_key, "sha256": file_hash(preview)})
    artifacts["world_preview.mp4"] = {
        "status": "reused" if render_reused else "written", "cache_key": render_key,
    }
    return {
        "artifacts": artifacts,
        "metrics": json.loads(arrays["metrics_json"].item()),
        "warnings": [
            "Table normals are sequence-wide; raw candidates and rejected evidence are retained.",
            "Intrinsics and learned metric depth remain uncertain; height fusion reports sigma.",
            "Clipped/curved table boundaries abstain from rectangular vanishing-line estimation.",
            "Hand/block co-location uses an explicitly flagged overlap depth prior, not measured contact.",
            "Block identity consolidation is geometric and may retain ambiguous fragments.",
            "Shared interaction is not verified; guider blocks remain separate and nonphysical.",
            "Camera translation can remain unobservable or drift under the OpenCV fallback.",
        ],
    }
