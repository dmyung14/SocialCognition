"""Cached contact rollout plus M6 refinement.

Perception-free entry point:
    python -m interaction_recon.stages_physics outputs/test_1/003 --budget 210
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from interaction_recon.io.cache import cache_key, cached_arrays, file_hash, write_json
from interaction_recon.render import physics as physics_render
from interaction_recon.simulation import (
    build_scene, controllers, physics_scene, physics_metrics, rollout, refine_physics,
)
from interaction_recon.simulation.physics_scene import PhysicsConfig
from interaction_recon.simulation.refine_physics import RefineConfig
from interaction_recon.stages_retarget import _array_digest, _export


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def process_physics(
    output: Path, *, force: bool = False,
    config: PhysicsConfig = PhysicsConfig(),
    refine_config: RefineConfig = RefineConfig(),
) -> dict:
    started = time.perf_counter()
    output = Path(output)
    observations = _load(output / "observations.npz")
    targets = _load(output / "retargeted.npz")
    if "block_consolidation_json" not in observations:
        raise ValueError(
            "M5 prerequisites are absent. Run the normal reconstruction once to "
            "consolidate cached geometry and apply hand-depth priors before IK."
        )
    if config.timestep != 0.002:
        raise ValueError("M6 initial and final rollouts require timestep=0.002")
    kinematic_xml, _ = build_scene.build_scene(observations)
    initial_xml = physics_scene.physics_variant(kinematic_xml, config)
    model = mujoco.MjModel.from_xml_string(initial_xml)
    for name, address in zip(targets["joint_names"], targets["joint_qpos_addresses"]):
        if int(model.joint(str(name)).qposadr[0]) != int(address):
            raise ValueError("Physics scene joint layout differs from retargeted.npz")
    code = {
        module.__name__: file_hash(module.__file__)
        for module in (build_scene, controllers, physics_scene, physics_metrics, rollout)
    }
    observation_digest = _array_digest(observations)
    target_digest = _array_digest(targets)
    initial_key = cache_key({
        "stage": "physics_initial_v3",
        "observations": observation_digest, "retargeted": target_digest,
        "scene": initial_xml, "config": asdict(config), "code": code,
        "mujoco": mujoco.__version__, "numpy": np.__version__,
    })
    cache = output / "cache"

    def compute_initial():
        before = time.perf_counter()
        result = rollout.run_rollout(initial_xml, targets, observations, config)
        result["physics_compute_s"] = np.array(time.perf_counter() - before)
        return result

    initial, initial_reused, initial_path = cached_arrays(
        cache, "physics", initial_key, compute_initial, force=force
    )
    # Initial metrics and the actual initial trajectory always remain available.
    _export(output / "physics_initial.npz", initial)
    (output / "scene_initial.xml").write_text(initial_xml + "\n", encoding="utf-8")
    key = cache_key({
        "stage": "refine_v1", "scene": kinematic_xml,
        "retargeted": target_digest, "observations": observation_digest,
        "physics_initial": initial_key,
        "config": asdict(refine_config), "physics_config": asdict(config),
        "code": {**code, "refine": file_hash(refine_physics.__file__)},
        "stage_code": file_hash(__file__),
        "mujoco": mujoco.__version__, "numpy": np.__version__,
    })
    arrays, reused, path = cached_arrays(
        cache, "refine", key,
        lambda: refine_physics.refine_physics(
            kinematic_xml, targets, observations, initial, config, refine_config
        ),
        force=force,
    )
    report = json.loads(arrays["refinement_json"].item())
    xml = arrays["refined_scene_xml"].item()
    selected_config = PhysicsConfig(**report["selected_config"])
    _export(output / "physics_rollout.npz", {
        name: value for name, value in arrays.items()
        if name not in ("refinement_json", "refined_scene_xml")
    })
    write_json(output / "refinement.json", report)
    (output / "scene_kinematic.xml").write_text(kinematic_xml + "\n", encoding="utf-8")
    (output / "scene.xml").write_text(xml + "\n", encoding="utf-8")
    artifacts = {
        name: {"status": "reused" if reused else "written", "cache_key": key}
        for name in (
            "physics_rollout.npz", "scene.xml", "scene_kinematic.xml",
            "refinement.json", path.relative_to(output).as_posix(),
        )
    }
    for name in (
        "physics_initial.npz", "scene_initial.xml",
        initial_path.relative_to(output).as_posix(),
    ):
        artifacts[name] = {
            "status": "reused" if initial_reused else "written", "cache_key": initial_key,
        }
    render_key = cache_key({
        "refine": key, "code": file_hash(physics_render.__file__),
        "config": asdict(selected_config), "fps": 30,
    })
    movie, receipt = output / "simulation.mp4", cache / "physics_render.json"
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
        frames = physics_render.render_physics(movie, xml, arrays, selected_config)
        write_json(receipt, {
            "cache_key": render_key, "sha256": file_hash(movie),
            "frames": frames, "fps": 30,
        })
    artifacts["simulation.mp4"] = {
        "status": "reused" if render_reused else "written", "cache_key": render_key,
        "scope": "selected_contact_physics_not_kinematic_reference",
    }
    metrics = json.loads(arrays["metrics_json"].item())
    metrics.update({
        "stage_wall_s_this_run": time.perf_counter() - started,
        "initial_physics_compute_s": float(initial["physics_compute_s"]),
        "refinement_compute_s": report["wall_s"],
        "cache_reused": reused, "initial_cache_reused": initial_reused,
        "render_cache_reused": render_reused,
        "runtime_target_s": 240,
        "runtime_target_met_this_run": time.perf_counter() - started <= 240,
        "config": asdict(selected_config),
        "refinement_selection_status": report["selection_status"],
        "refinement_selected_valid": report["selected_valid"],
        "m6_pass": report["m6_pass"],
    })
    metrics_path = output / "metrics.json"
    existing = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists() else json.loads(observations["metrics_json"].item())
    )
    existing.update({
        "physics": metrics,
        "physics_initial": {
            **report["physics_initial"], "total_loss": report["initial_total_loss"],
        },
        "physics_refined": {
            **report["physics_refined"], "total_loss": report["refined_total_loss"],
            "selected_valid": report["selected_valid"], "m6_pass": report["m6_pass"],
        },
        "refinement_improvement": report["refinement_improvement"],
        "m6_pass": report["m6_pass"],
        "scope": "Milestone 6: bounded inverse physics; estimated references and fusion uncertainty retained",
    })
    write_json(metrics_path, existing)
    artifacts["metrics.json"] = {"status": "written", "scope": "milestone_6"}
    warnings = [
        "Geometric block identities, hand-depth priors, and shared interaction remain uncertain."
    ]
    if not metrics["m5_pass"]:
        warnings.append("M5 has NOT passed; inspect coverage, contact attribution, and stability.")
    if not metrics["m6_pass"]:
        warnings.append(
            "M6 has NOT passed: no selected valid rollout measurably improves the "
            "accepted full-clip block position error. Initial metrics are retained."
        )
    if not report["selected_valid"]:
        warnings.append(
            "No valid refinement candidate was available. The initial rollout is "
            "retained for diagnostics, NOT certified as a valid refined result."
        )
    return {"artifacts": artifacts, "metrics": metrics, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rerun cached M5/M6 without perception or IK")
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--budget", type=float, default=210.0)
    parser.add_argument("--evaluations", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=6023)
    args = parser.parse_args(argv)
    result = process_physics(
        args.output, force=args.force,
        refine_config=RefineConfig(
            time_budget_s=args.budget, coarse_evaluations=args.evaluations,
            top_k=args.top_k, seed=args.seed,
        ),
    )
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    return 0 if result["metrics"]["stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
