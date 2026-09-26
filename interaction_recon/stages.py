import importlib.metadata
import json
from pathlib import Path

import numpy as np

from interaction_recon.config import RunOptions
from interaction_recon.io.cache import cache_key, cached_arrays, file_hash, write_json
from interaction_recon.io.media import load_media
from interaction_recon.io.sync import synchronize
from interaction_recon.render.tracking import render_tracking
from interaction_recon.stages_world import process_world
from interaction_recon.vision.detectors import require_models
from interaction_recon.vision.metric_tracking import track_metric_stream as track_stream
from interaction_recon.vision.scene_tracking import refine_scene

OBSERVATION_VERSION = "m3-human-egocentric-wrist-association-9"
LEGACY_OBSERVATION_VERSION = "m2-observations-4"
SCENE_VERSION = "m2-scene-local-table-blocks-7"
SYNC_VERSION = "m2-sync-3"
RENDER_VERSION = "m2-render-4"

HUMAN_KEYS = (
    "metadata_json", "timestamps_s", "frame_pts_s", "source_timestamps_s",
    "source_frame_indices", "duration_s", "image_size", "pose", "pose_observed",
    "hands", "hands_observed", "handedness", "handedness_confidence",
    "hand_owner", "hand_owner_confidence", "hand_ids", "head_center",
    "head_transform", "head_confidence", "head_method", "pose_world", "hands_world",
    "handedness_raw", "hand_detection_method", "hand_assignment_method",
    "wearer_forearms_2d", "wearer_forearm_method",
)


def _versions():
    result = {}
    for name in ("mediapipe", "numpy", "scipy", "imageio-ffmpeg"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    import cv2
    result["opencv"] = cv2.__version__
    return result


def _human_arrays(arrays):
    missing = set(HUMAN_KEYS) - arrays.keys()
    if missing:
        raise ValueError(f"Human observation cache lacks keys: {sorted(missing)}")
    return {name: arrays[name] for name in HUMAN_KEYS}


def _read_legacy_humans(path, expected_key):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["_cache_key"].item() != expected_key:
                return None
            return _human_arrays({name: archive[name].copy() for name in HUMAN_KEYS})
    except (OSError, ValueError, EOFError, KeyError):
        return None


def process_pair(
    guider_path: Path, builder_path: Path, output: Path, options: RunOptions,
):
    models = require_models(options.model_dir)
    model_hashes = {name: file_hash(path) for name, path in models.items()}
    paths = {"guider": guider_path, "builder": builder_path}
    hashes = {role: file_hash(path) for role, path in paths.items()}
    versions, cache = _versions(), output / "cache"
    sequences, observations, keys, artifacts = {}, {}, {}, {}

    def sequence(role):
        if role not in sequences:
            sequences[role] = load_media(paths[role], options.analysis_fps)
        return sequences[role]

    from interaction_recon.vision import blocks, table, scene_tracking
    scene_code = {
        module.__name__: file_hash(module.__file__)
        for module in (blocks, table, scene_tracking)
    }
    for role in ("guider", "builder"):
        human_key = cache_key({
            "stage": OBSERVATION_VERSION, "input_sha256": hashes[role],
            "role": role, "analysis_fps": options.analysis_fps,
            "models": model_hashes, "versions": versions, "delegate": "cpu",
        })
        humans, reused, human_path = cached_arrays(
            cache, f"{role}-2d", human_key,
            lambda role=role: _human_arrays(track_stream(
                sequence(role), role, options.model_dir, include_scene=False
            )), force=options.force,
        )
        artifacts[human_path.relative_to(output).as_posix()] = {
            "status": "reused" if reused else "written", "cache_key": human_key,
            "kind": "per_stream_human_observations",
        }
        scene_key = cache_key({
            "stage": SCENE_VERSION, "input_sha256": hashes[role], "humans": human_key,
            "analysis_fps": options.analysis_fps, "code": scene_code,
            "versions": {name: versions[name] for name in ("numpy", "scipy", "opencv")},
        })
        scene, reused, scene_path = cached_arrays(
            cache, f"{role}-scene", scene_key,
            lambda role=role, humans=humans: refine_scene(sequence(role), humans),
            force=options.force,
        )
        observations[role], keys[role] = {**humans, **scene}, scene_key
        artifacts[scene_path.relative_to(output).as_posix()] = {
            "status": "reused" if reused else "written", "cache_key": scene_key,
            "kind": "per_stream_2d_observations",
            "human_cache": human_path.relative_to(output).as_posix(),
        }

    sync_key = cache_key({
        "stage": SYNC_VERSION, "sources": hashes, "observations": keys,
        "threshold": options.sync_threshold, "versions": versions,
    })

    def compute_sync():
        result = synchronize(
            sequence("guider"), sequence("builder"), threshold=options.sync_threshold,
            guider_motion=observations["guider"]["motion_energy"],
            builder_motion=observations["builder"]["motion_energy"],
        )
        result["cache_key"] = sync_key
        return {"json": np.array(json.dumps(result, allow_nan=False))}

    sync_arrays, reused, sync_path = cached_arrays(
        cache, "sync", sync_key, compute_sync, force=options.force
    )
    sync = json.loads(sync_arrays["json"].item())
    write_json(output / "sync.json", sync)
    artifacts["sync.json"] = {
        "status": "reused" if reused else "written", "cache_key": sync_key,
    }
    artifacts[sync_path.relative_to(output).as_posix()] = {
        "status": "reused" if reused else "written",
    }
    render_key = cache_key({
        "stage": RENDER_VERSION, "observations": keys, "sync": sync_key,
        "fps": options.analysis_fps, "panel_size": 480, "versions": versions,
    })
    tracking_path, receipt_path = output / "tracking.mp4", cache / "tracking.json"
    reused = False
    if not options.force and tracking_path.is_file() and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            reused = receipt["cache_key"] == render_key and receipt["sha256"] == file_hash(tracking_path)
        except (OSError, ValueError, KeyError):
            pass
    if not reused:
        render_tracking(
            tracking_path, sequence("guider"), sequence("builder"),
            observations["guider"], observations["builder"], sync, fps=options.analysis_fps,
        )
        write_json(receipt_path, {"cache_key": render_key, "sha256": file_hash(tracking_path)})
    artifacts["tracking.mp4"] = {
        "status": "reused" if reused else "written", "cache_key": render_key,
    }

    world = process_world(output, observations, keys, sequence, sync, options, versions)
    artifacts.update(world["artifacts"])
    metrics = world["metrics"]
    metrics["synchronization_confidence"] = sync["confidence"]
    metrics["block_observation_policy"] = (
        "builder-only physical set; >=3 hits and >=30% eligible nonoccluded SOURCE frames; "
        "occluded_hold is interpolated, never observed"
    )
    for role, arrays in observations.items():
        methods = arrays["head_method"]
        metrics["streams"][role].update({
            "body_keypoint_coverage": float(np.mean(arrays["pose_observed"])),
            "frames_with_any_hand": float(np.mean(np.any(arrays["hands_observed"], axis=(1, 2)))),
            "frames_with_wearer_hand": float(np.mean(np.any(
                (arrays["hand_owner"] == 1) & np.any(arrays["hands_observed"], axis=2), axis=1
            ))),
            "frames_with_arm_blob_fallback": float(np.mean(np.any(
                arrays["wearer_forearm_method"] != "none", axis=1
            ))),
            "hand_detection_methods": {
                str(m): int(np.count_nonzero(arrays["hand_detection_method"] == m))
                for m in np.unique(arrays["hand_detection_method"])
            },
            "hand_assignment_methods": {
                str(m): int(np.count_nonzero(arrays["hand_assignment_method"] == m))
                for m in np.unique(arrays["hand_assignment_method"])
            },
            "frames_with_any_block": float(np.mean(np.any(arrays["block_observed"], axis=1))),
            "frames_with_any_block_hold": float(np.mean(np.any(arrays["block_interpolated"], axis=1))),
            "frames_with_any_block_candidate": float(np.mean(arrays["block_candidate_count"] > 0)),
            "mean_provisional_block_count": float(np.mean(arrays["block_provisional_count"])),
            "head_detection_coverage": float(np.mean(arrays["head_confidence"] > 0)),
            "head_face_detection_coverage": float(np.mean(np.isin(
                methods, ["face_pose_crop", "face_full_frame"]
            ))),
            "head_pose_fallback_coverage": float(np.mean(methods == "pose_landmarks_fallback")),
            "mean_table_confidence": float(np.mean(arrays["table_confidence"])),
        })
    consistency = metrics["cross_view_consistency"]
    sync["cross_view_consistency"] = consistency
    write_json(output / "sync.json", sync)
    write_json(output / "metrics.json", metrics)

    from interaction_recon.stages_retarget import process_retarget
    reference = process_retarget(output, force=options.force)
    artifacts.update(reference["artifacts"])
    metrics["retarget"] = reference["metrics"]
    metrics["scope"] = "Milestone 4: uncertain reconstruction and kinematic reference; no physics"
    write_json(output / "metrics.json", metrics)
    artifacts["metrics.json"] = {"status": "written", "scope": "milestone_4"}
    warnings = sync["warnings"] + world["warnings"] + reference["warnings"]
    if metrics["seated_orientation_invariant"]["status"] == "failed":
        warnings.append("ERROR: seated orientation invariant FAILED; inspect metrics.json.")
    return {
        "artifacts": artifacts,
        "synchronization": {
            name: sync[name] for name in ("offset_s", "method", "confidence", "reliable")
        },
        "cross_view_consistency": consistency, "warnings": warnings,
        "world_stage_status": "complete", "retarget_stage_status": "complete",
    }
