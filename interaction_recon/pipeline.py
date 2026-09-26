from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import uuid

from interaction_recon import SCHEMA_VERSION, __version__
from interaction_recon.config import RunOptions
from interaction_recon.io.cache import write_json as _write_json
from interaction_recon.io.discover import DiscoveredMedia, discover_input, explicit_media
from interaction_recon.io.media import MetadataError, probe_media
from interaction_recon.io.pairing import pair_media
from interaction_recon.stages import process_pair

FUTURE_ARTIFACTS = (
    "sync.json", "cameras.npz", "observations.npz", "retargeted.npz",
    "scene.xml", "physics_rollout.npz", "metrics.json", "simulation.mp4",
    "comparison.mp4", "tracking.mp4", "world_preview.mp4", "kinematic_reference.mp4",
    "scene_kinematic.xml", "physics_initial.npz", "scene_initial.xml", "refinement.json",
    "audit.json", "logs/block_write_guard.json",
)
ACTIVE_ARTIFACTS = FUTURE_ARTIFACTS


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    return output


def _logger(output: Path, debug: bool) -> logging.Logger:
    logger = logging.getLogger(f"interaction_recon.{uuid.uuid4().hex}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output / "logs" / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _base_manifest(mode: str, options: RunOptions) -> dict:
    deferred = []
    if options.inventory_only:
        deferred = [
            "media_normalization", "synchronization", "tracking",
            "camera_reconstruction", "world_fusion", "retargeting", "physics",
            "inverse_physics_refinement", "comparison_rendering", "final_audit",
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "application_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": 2 if options.inventory_only else 7,
        "mode": mode,
        "reconstruction_performed": False,
        "simulation_performed": False,
        "kinematic_reference_performed": False,
        "refinement_performed": False,
        "requested_options": asdict(options),
        "deferred_stages": deferred,
    }


def _inventory(media: list[DiscoveredMedia], logger: logging.Logger) -> list[dict]:
    result = []
    for item in media:
        logger.info("Inspecting %s", item.id)
        entry = {
            "id": item.id, "source": asdict(item.source),
            "segment_id": item.segment_id, "role": item.role,
            "metadata_status": "ok", "metadata": None, "error": None,
        }
        try:
            entry["metadata"] = asdict(probe_media(item.local_path))
        except MetadataError as exc:
            diagnostic = str(exc).replace(str(item.local_path), item.id)
            entry.update(metadata_status="error", error=diagnostic)
            logger.error("%s: %s", item.id, diagnostic)
        result.append(entry)
    return result


def _pair_manifest(segment: dict, records: dict, options: RunOptions, mode: str) -> dict:
    manifest = _base_manifest(mode, options)
    selected = [records[segment["guider"]], records[segment["builder"]]]
    manifest.update({
        "segment_id": segment["segment_id"], "status": "paired",
        "guider": segment["guider"], "builder": segment["builder"],
        "stage_status": (
            "metadata_complete" if all(item["metadata_status"] == "ok" for item in selected)
            else "metadata_failed"
        ),
        "world_stage_status": "not_run", "retarget_stage_status": "not_run",
        "physics_stage_status": "not_run", "refine_stage_status": "not_run",
        "final_stage_status": "not_run", "audit_pass": None,
        "media": selected, "warnings": [],
        "artifacts": {
            "manifest.json": {"status": "written"},
            "logs/pipeline.log": {"status": "written"},
            **{name: {"status": "not_implemented"} for name in FUTURE_ARTIFACTS},
        },
    })
    return manifest


def _execute_pair(
    directory: Path, manifest: dict, guider: DiscoveredMedia,
    builder: DiscoveredMedia, options: RunOptions,
    segment_table: list[dict] | None = None,
) -> bool:
    directory = _prepare_output(directory)
    logger = _logger(directory, options.debug)
    logger.info("Starting segment %s", manifest["segment_id"])
    try:
        if manifest["stage_status"] == "metadata_failed" or options.inventory_only:
            _write_json(directory / "manifest.json", manifest)
            return manifest["stage_status"] != "metadata_failed"
        for artifact in ACTIVE_ARTIFACTS:
            manifest["artifacts"][artifact] = {"status": "pending"}
        manifest.update(
            stage_status="tracking_running", world_stage_status="pending",
            retarget_stage_status="pending", physics_stage_status="pending",
            refine_stage_status="pending", final_stage_status="pending",
        )
        _write_json(directory / "manifest.json", manifest)
        try:
            result = process_pair(guider.local_path, builder.local_path, directory, options)
            manifest["artifacts"].update(result["artifacts"])
            manifest["warnings"].extend(result["warnings"])
            manifest["synchronization"] = result["synchronization"]
            manifest["cross_view_consistency"] = result["cross_view_consistency"]
            manifest["stage_status"] = "tracking_complete"
            manifest["world_stage_status"] = result["world_stage_status"]
            manifest["retarget_stage_status"] = result["retarget_stage_status"]
            manifest["reconstruction_performed"] = True
            manifest["kinematic_reference_performed"] = True
            manifest["verified_fusion"] = False
            from interaction_recon.stages_physics import process_physics
            manifest["physics_stage_status"] = "running"
            manifest["refine_stage_status"] = "running"
            _write_json(directory / "manifest.json", manifest)
            physics = process_physics(directory, force=options.force)
            manifest["artifacts"].update(physics["artifacts"])
            manifest["warnings"].extend(physics["warnings"])
            manifest["simulation_performed"] = True
            manifest["refinement_performed"] = True
            manifest["physics_stage_status"] = "complete"
            manifest["refine_stage_status"] = "complete"
            manifest["physics_stable"] = physics["metrics"]["stable"]
            manifest["m5_pass"] = physics["metrics"]["m5_pass"]
            manifest["m6_pass"] = physics["metrics"]["m6_pass"]
            manifest["refinement_selection_status"] = physics["metrics"]["refinement_selection_status"]
            manifest["final_stage_status"] = "running"
            logger.info("Physics complete; starting final render and audit")
            _write_json(directory / "manifest.json", manifest)
            from interaction_recon.stages_final import process_final, summary
            final = process_final(
                directory, guider.local_path, builder.local_path, options, segment_table
            )
            manifest["artifacts"].update(final["artifacts"])
            manifest["audit_pass"] = final["audit"]["pass"]
            manifest["targets"] = final["metrics"]["targets"]
            manifest["final_stage_status"] = "complete"
            manifest["summary"] = summary(
                manifest["segment_id"], final["metrics"], final["audit"]
            )
            print(manifest["summary"])
            logger.info(manifest["summary"])
            if not manifest["audit_pass"]:
                manifest["warnings"].append("Final integrity audit failed; inspect audit.json.")
            if not manifest["targets"]["physics_block_trajectory"]["pass"]:
                manifest["warnings"].append(
                    "Absolute block trajectory target <=0.25 BL is NOT met or not evaluable."
                )
        except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
            diagnostic = str(exc)
            for item in (guider, builder):
                diagnostic = diagnostic.replace(str(item.local_path), item.id)
            if manifest["final_stage_status"] == "running":
                manifest["final_stage_status"] = "failed"
                failed_artifacts = ("comparison.mp4", "audit.json")
            elif manifest["physics_stage_status"] == "running":
                manifest["physics_stage_status"] = "failed"
                manifest["refine_stage_status"] = "failed"
                failed_artifacts = (
                    "scene.xml", "physics_rollout.npz", "simulation.mp4", "refinement.json",
                )
            else:
                manifest["stage_status"] = "tracking_failed"
                manifest["world_stage_status"] = "failed"
                manifest["retarget_stage_status"] = "failed"
                manifest["physics_stage_status"] = "not_run"
                manifest["refine_stage_status"] = "not_run"
                failed_artifacts = ACTIVE_ARTIFACTS
            manifest["error"] = diagnostic
            for artifact in failed_artifacts:
                manifest["artifacts"][artifact] = {
                    "status": "failed",
                    "note": "Existing files, if any, are not certified for this run",
                }
            logger.error("Reconstruction failed: %s", diagnostic)
            print(f"error: segment {manifest['segment_id']}: {diagnostic}", file=sys.stderr)
        _write_json(directory / "manifest.json", manifest)
        # Integrity/accuracy failures remain inspectable diagnostics, not an
        # exception that discards a completed run. Execution failures are nonzero.
        return (
            manifest["stage_status"] == "tracking_complete"
            and manifest["physics_stage_status"] == "complete"
            and manifest["final_stage_status"] == "complete"
            and manifest.get("physics_stable", False)
        )
    finally:
        _close_logger(logger)


def run_inventory(
    input_path: str | Path, output_path: str | Path, options: RunOptions,
) -> tuple[dict, int]:
    with discover_input(input_path) as discovery:
        if not discovery.media:
            raise ValueError("Input contains no usable MP4/GIF file paths")
        segments, unclassified = pair_media(discovery.media)
        output = _prepare_output(output_path)
        logger = _logger(output, options.debug)
        try:
            inventory = _inventory(discovery.media, logger)
            records = {entry["id"]: entry for entry in inventory}
            sources = {item.id: item for item in discovery.media}
            for segment in segments:
                segment["output_directory"] = (
                    segment["segment_id"] if segment["status"] == "paired" else None
                )
            tracking_failed = completed = retarget_completed = physics_completed = 0
            refinement_completed = execution_failed = 0
            for segment in segments:
                if segment["status"] == "paired":
                    pair = _pair_manifest(segment, records, options, "run")
                    ok = _execute_pair(
                        output / segment["segment_id"], pair,
                        sources[segment["guider"]], sources[segment["builder"]],
                        options, segments,
                    )
                    for name in (
                        "stage_status", "world_stage_status", "retarget_stage_status",
                        "physics_stage_status", "refine_stage_status", "final_stage_status",
                        "audit_pass",
                    ):
                        segment[name] = pair[name]
                    segment["m6_pass"] = pair.get("m6_pass", False)
                    segment["summary"] = pair.get("summary")
                    tracking_failed += int(pair["stage_status"] == "tracking_failed")
                    execution_failed += int(not ok)
                    completed += int(pair["world_stage_status"] == "complete")
                    retarget_completed += int(pair["retarget_stage_status"] == "complete")
                    physics_completed += int(pair["physics_stage_status"] == "complete")
                    refinement_completed += int(pair["refine_stage_status"] == "complete")
                else:
                    segment["stage_status"] = f"skipped_{segment['status']}"
                    segment["audit_pass"] = None
                    logger.warning("Segment %s is %s", segment["segment_id"], segment["status"])
                    print(f"Segment {segment['segment_id']}: {segment['stage_status']}; no fused output")
            failed = sum(entry["metadata_status"] == "error" for entry in inventory)
            paired_count = sum(s["status"] == "paired" for s in segments)
            stage_status = (
                "metadata_failed" if failed else
                "tracking_failed" if tracking_failed else
                "physics_failed" if execution_failed else
                "tracking_complete" if paired_count and not options.inventory_only else
                "metadata_complete"
            )
            manifest = _base_manifest("run", options)
            manifest.update({
                "input": {"kind": discovery.kind, "path": discovery.input_path},
                "stage_status": stage_status,
                "reconstruction_performed": completed > 0,
                "kinematic_reference_performed": retarget_completed > 0,
                "simulation_performed": physics_completed > 0,
                "refinement_performed": refinement_completed > 0,
                "world_completed_segments": completed,
                "retarget_completed_segments": retarget_completed,
                "physics_completed_segments": physics_completed,
                "refinement_completed_segments": refinement_completed,
                "verified_fusion": False,
                "counts": {
                    "media": len(inventory), "metadata_ok": len(inventory) - failed,
                    "metadata_failed": failed, "paired_segments": paired_count,
                    "unpaired_segments": sum(s["status"] == "unpaired" for s in segments),
                    "ambiguous_segments": sum(s["status"] == "ambiguous" for s in segments),
                    "unclassified_files": len(unclassified),
                },
                "tracking_failed_segments": tracking_failed,
                "media": inventory, "segments": segments, "unclassified_files": unclassified,
                "segment_status_table": [
                    {
                        "segment_id": s["segment_id"], "status": s["status"],
                        "stage_status": s["stage_status"],
                        "final_stage_status": s.get("final_stage_status", "not_run"),
                        "audit_pass": s.get("audit_pass"),
                        "output_directory": s["output_directory"],
                    }
                    for s in segments
                ],
            })
            _write_json(output / "manifest.json", manifest)
            logger.info("Finished: %s", stage_status)
            return manifest, int(bool(failed or execution_failed))
        finally:
            _close_logger(logger)


def reconstruct_inventory(
    guider_path: str | Path, builder_path: str | Path,
    output_path: str | Path, options: RunOptions,
) -> tuple[dict, int]:
    guider = explicit_media(guider_path, "guider")
    builder = explicit_media(builder_path, "builder")
    if guider.local_path.samefile(builder.local_path):
        raise ValueError("Guider and builder must be distinct source files")
    common_id = (
        guider.segment_id
        if guider.segment_id is not None and guider.segment_id == builder.segment_id
        else "explicit_pair"
    )
    output = _prepare_output(output_path)
    logger = _logger(output, options.debug)
    try:
        records = {entry["id"]: entry for entry in _inventory([guider, builder], logger)}
    finally:
        _close_logger(logger)
    manifest = _pair_manifest(
        {"segment_id": common_id, "guider": guider.id, "builder": builder.id},
        records, options, "reconstruct",
    )
    manifest["pairing_method"] = "explicit"
    if (
        guider.segment_id is not None and builder.segment_id is not None
        and guider.segment_id != builder.segment_id
    ):
        manifest["warnings"].append(
            "Explicit sources have different inferred segment IDs; accepted as "
            "user-assigned partners, not verified as the same interaction."
        )
    ok = _execute_pair(output, manifest, guider, builder, options)
    return manifest, 0 if ok else 1
