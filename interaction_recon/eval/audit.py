"""Fail-closed artifact and physics integrity audit."""
import json
from pathlib import Path

import numpy as np

from interaction_recon.io.cache import file_hash, write_json
from interaction_recon.simulation.physics_metrics import movement_attribution
from interaction_recon.simulation.physics_scene import audit_physics_scene

REQUIRED_ARTIFACTS = (
    "manifest.json", "sync.json", "cameras.npz", "observations.npz",
    "retargeted.npz", "scene.xml", "physics_rollout.npz", "metrics.json",
    "simulation.mp4", "comparison.mp4", "tracking.mp4",
    "kinematic_reference.mp4", "scene_kinematic.xml",
    "physics_initial.npz", "scene_initial.xml", "refinement.json",
    "logs/block_write_guard.json", "logs/pipeline.log",
)


def check_result(passed: bool, detail) -> dict:
    return {"pass": bool(passed), "status": "passed" if passed else "failed", "detail": detail}


def scene_check(xml: str) -> dict:
    try:
        model = audit_physics_scene(xml)
        # The existing scene validator also forbids unauthorized transmissions,
        # tendons, block parents, disabled block contacts and block constraints.
        blocks = [
            model.body(b).name for b in range(model.nbody)
            if (model.body(b).name or "").startswith("block_")
        ]
        return check_result(True, {
            "block_bodies": blocks,
            "policy": (
                "No block actuation, weld/equality/tendon reference, mocap body or "
                "non-world parent. Builder-only compliant root welds are allowed."
            ),
        })
    except (ValueError, RuntimeError) as exc:
        return check_result(False, str(exc))


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def audit_segment(
    directory: Path, *, render_format: str = "mp4",
    segment_table: list[dict] | None = None,
) -> dict:
    directory = Path(directory)
    checks = {}
    required = list(REQUIRED_ARTIFACTS)
    if render_format in ("gif", "both"):
        required += ["simulation.gif", "comparison.gif"]
    missing = [
        name for name in required
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0
    ]
    checks["artifacts_exist_nonempty"] = check_result(not missing, {"missing": missing})
    try:
        checks["no_block_actuation_or_constraints"] = scene_check(
            (directory / "scene.xml").read_text(encoding="utf-8")
        )
    except OSError as exc:
        checks["no_block_actuation_or_constraints"] = check_result(False, str(exc))

    try:
        rollout = _load(directory / "physics_rollout.npz")
        log = json.loads((directory / "logs/block_write_guard.json").read_text(encoding="utf-8"))
        evidence = json.loads(rollout["guard_log_json"].item())
        protected = np.asarray(rollout["block_qpos_addresses"]).ravel().tolist()
        guard_ok = (
            log["physics_rollout_sha256"] == file_hash(directory / "physics_rollout.npz")
            and log["guard"] == evidence
            and evidence["locked"] and evidence["checks"] > 0
            and evidence["violations"] == 0
            and evidence["engine_steps"] >= len(rollout["timestamps"]) - 1
            and evidence["block_qpos_addresses"] == protected
        )
        checks["no_block_state_writes"] = check_result(guard_ok, evidence)
        numeric = [
            name for name, value in rollout.items()
            if np.issubdtype(value.dtype, np.number)
        ]
        bad = [name for name in numeric if not np.isfinite(rollout[name]).all()]
        checks["physics_no_nans"] = check_result(
            not bad and not rollout["nan_flags"].any(), {"nonfinite_arrays": bad}
        )
        checks["physics_stability"] = check_result(
            not rollout["instability_flags"].any(), {
                "instability_samples": int(rollout["instability_flags"].sum()),
            }
        )
        attribution = movement_attribution(rollout)
        checks["blocks_moved_only_by_transitive_contact"] = check_result(
            attribution["status"] == "passed", attribution
        )
        targets = _load(directory / "retargeted.npz")
        human = np.r_[targets["guider_qpos_indices"], targets["builder_qpos_indices"]]
        metadata = json.loads(targets["metadata_json"].item())
        physics_metadata = json.loads(rollout["metadata_json"].item())
        separate = (
            not np.isin(human, rollout["block_qpos_addresses"]).any()
            and metadata.get("block_qpos_included") is False
            and physics_metadata.get("schema", "").startswith("physics_rollout_")
            and (directory / "scene_kinematic.xml").is_file()
            and (directory / "kinematic_reference.mp4").is_file()
            and file_hash(directory / "simulation.mp4")
            != file_hash(directory / "kinematic_reference.mp4")
        )
        checks["kinematic_reference_separate"] = check_result(separate, {
            "human_targets_contain_block_addresses": bool(
                np.isin(human, rollout["block_qpos_addresses"]).any()
            ),
            "physics_schema": physics_metadata.get("schema"),
            "reference_movie": "kinematic_reference.mp4",
            "physics_movie": "simulation.mp4",
        })
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        for name in (
            "no_block_state_writes", "physics_no_nans", "physics_stability",
            "blocks_moved_only_by_transitive_contact", "kinematic_reference_separate",
        ):
            checks.setdefault(name, check_result(False, f"Evidence unavailable: {exc}"))

    names = ("observations.npz", "retargeted.npz", "physics_rollout.npz")
    try:
        paths = [directory / name for name in names]
        distinct = all(
            not paths[a].samefile(paths[b])
            and file_hash(paths[a]) != file_hash(paths[b])
            for a in range(3) for b in range(a + 1, 3)
        )
        checks["observation_target_physics_files_separate"] = check_result(distinct, list(names))
    except OSError as exc:
        checks["observation_target_physics_files_separate"] = check_result(False, str(exc))

    unpaired = [
        row["segment_id"] for row in (segment_table or [])
        if row.get("status") == "unpaired"
    ]
    checks["unpaired_segments_reported"] = check_result(
        segment_table is None or all(
            row.get("output_directory") is None
            for row in segment_table if row.get("status") in ("unpaired", "ambiguous")
        ),
        {
            "unpaired_segments": unpaired,
            "scope": "explicit pair: not applicable" if segment_table is None else "root status table",
        },
    )
    result = {
        "schema": "audit_m7_v1",
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
        "scope": (
            "Artifact and runtime-evidence integrity, not a claim of accurate fusion "
            "or trajectory recovery. Missing evidence fails closed."
        ),
        "observation_nan_policy": (
            "Missing observations may be NaN with zero confidence; the no-NaN "
            "check applies to numeric physics rollout arrays."
        ),
    }
    write_json(directory / "audit.json", result)
    return result
