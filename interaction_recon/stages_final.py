"""Final deliverables, independently cached from perception and physics search."""
import json
from pathlib import Path

import numpy as np

from interaction_recon.eval.audit import audit_segment
from interaction_recon.eval.metrics import consolidate_metrics
from interaction_recon.io.cache import cache_key, file_hash, write_json
from interaction_recon.io.media import load_media
from interaction_recon.render import comparison, video, framing
from interaction_recon.render.physics import render_physics
from interaction_recon.render.render import render_kinematic_reference


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _cached_movie(output, name, configuration, compute, force):
    key = cache_key(configuration)
    movie = output / name
    receipt = output / "cache" / f"final-{name}.json"
    reused = False
    if not force and movie.is_file() and receipt.is_file():
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            reused = saved["key"] == key and saved["sha256"] == file_hash(movie)
        except (ValueError, OSError, KeyError):
            pass
    if not reused:
        compute()
        write_json(receipt, {"key": key, "sha256": file_hash(movie)})
    return {"status": "reused" if reused else "written", "cache_key": key}


def process_final(
    output: Path, guider_path: Path, builder_path: Path, options,
    segment_table: list[dict] | None = None,
) -> dict:
    observations = _load(output / "observations.npz")
    rollout = _load(output / "physics_rollout.npz")
    targets = _load(output / "retargeted.npz")
    sync = json.loads((output / "sync.json").read_text(encoding="utf-8"))
    existing = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    metrics = consolidate_metrics(existing, observations, rollout)
    write_json(output / "metrics.json", metrics)
    guard = (
        json.loads(rollout["guard_log_json"].item())
        if "guard_log_json" in rollout else {"missing": True}
    )
    write_json(output / "logs/block_write_guard.json", {
        "physics_rollout_sha256": file_hash(output / "physics_rollout.npz"),
        "guard": guard,
    })
    code = {
        module.__name__: file_hash(module.__file__)
        for module in (comparison, video, framing)
    }
    artifacts = {
        "metrics.json": {"status": "written", "scope": "milestone_7"},
        "logs/block_write_guard.json": {"status": "written"},
    }
    # These receipts also include shared framing/encoder dependencies, which
    # legacy M4/M6 render receipts did not know about.
    for name, inputs, compute in (
        (
            "simulation.mp4",
            ("scene.xml", "physics_rollout.npz"),
            lambda: render_physics(
                output / "simulation.mp4",
                (output / "scene.xml").read_text(encoding="utf-8"), rollout,
            ),
        ),
        (
            "kinematic_reference.mp4",
            ("scene_kinematic.xml", "retargeted.npz", "observations.npz"),
            lambda: render_kinematic_reference(
                output / "kinematic_reference.mp4",
                (output / "scene_kinematic.xml").read_text(encoding="utf-8"),
                targets, observations,
            ),
        ),
    ):
        artifacts[name] = _cached_movie(
            output, name,
            {"stage": "m7-final-render-1", "code": code,
             "inputs": {item: file_hash(output / item) for item in inputs}},
            compute, options.force,
        )
        # Keep the preceding stage's receipt consistent with the final render.
        legacy = output / "cache" / (
            "physics_render.json" if name == "simulation.mp4" else "kinematic_reference.json"
        )
        if legacy.exists():
            receipt = json.loads(legacy.read_text(encoding="utf-8"))
            receipt["sha256"] = file_hash(output / name)
            write_json(legacy, receipt)

    physics = metrics["physics"]
    fusion = str(observations["fusion_method"].item())
    configuration = {
        "stage": "comparison-m7-1", "code": code,
        "sources": [file_hash(guider_path), file_hash(builder_path)],
        "simulation": file_hash(output / "simulation.mp4"),
        "simulation_start_s": float(rollout["timestamps"][0]),
        "sync": sync, "fusion": fusion,
        "physics_status": comparison.physics_status(physics),
    }

    def make_comparison():
        guider = load_media(guider_path, analysis_fps=None)
        builder = load_media(builder_path, analysis_fps=None)
        comparison.render_comparison(
            output / "comparison.mp4", guider, builder, output / "simulation.mp4",
            float(rollout["timestamps"][0]), sync, fusion, physics,
        )

    artifacts["comparison.mp4"] = _cached_movie(
        output, "comparison.mp4", configuration, make_comparison, options.force
    )
    if options.render_format in ("gif", "both"):
        for stem in ("simulation", "comparison"):
            artifacts[f"{stem}.gif"] = _cached_movie(
                output, f"{stem}.gif",
                {"source": file_hash(output / f"{stem}.mp4"),
                 "code": file_hash(video.__file__), "fps": 15, "width": 640},
                lambda stem=stem: video.convert_gif(output / f"{stem}.mp4"),
                options.force,
            )
    audit = audit_segment(
        output, render_format=options.render_format, segment_table=segment_table
    )
    artifacts["audit.json"] = {"status": "written"}
    return {"artifacts": artifacts, "metrics": metrics, "audit": audit}


def summary(segment_id: str, metrics: dict, audit: dict) -> str:
    physics = metrics.get("physics", {})
    def value(name, suffix):
        number = physics.get(name)
        return "unavailable" if number is None else f"{number:.3f}{suffix}"
    return (
        f"Segment {segment_id}: audit={'PASS' if audit['pass'] else 'FAIL'}; "
        f"block={value('block_trajectory_position_error_m', ' m')} / "
        f"{value('block_trajectory_error_normalized_by_length', ' BL')}; "
        f"rotation={value('block_rotation_error_rad', ' rad')}; "
        f"penetration={value('max_penetration_m', ' m')}; "
        f"trajectory target={'PASS' if metrics['targets']['physics_block_trajectory']['pass'] else 'NOT MET'}; "
        f"M6={physics.get('m6_pass', False)}"
    )
