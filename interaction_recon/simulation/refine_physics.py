"""Bounded, deterministic, perception-free inverse physics.

All candidate state evolution goes through run_rollout and BlockWriteGuard.
Only model material/controller parameters and builder references are adjusted.
"""
from dataclasses import asdict, dataclass, replace
import json
import time
from typing import Callable

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

from interaction_recon.fusion.physics_prerequisites import surface_distance
from interaction_recon.simulation.controllers import TargetInterpolator
from interaction_recon.simulation.physics_metrics import physics_metrics
from interaction_recon.simulation.physics_scene import PhysicsConfig, physics_variant
from interaction_recon.simulation.rollout import run_rollout


PARAMETER_NAMES = (
    "density_kg_m3", "table_friction", "hand_friction", "contact_timeconst_s",
    "weld_timeconst_multiplier", "wrist_gain_multiplier",
    "finger_kp_multiplier", "finger_force_multiplier",
    "offset_x_m", "offset_y_m", "offset_z_m", "timing_offset_s",
)
TERM_NAMES = (
    "block_position", "block_rotation", "contact_timing", "hand_reference",
    "penetration", "joint_limit", "instability",
)
LARGE_LOSS = 1e6


@dataclass(frozen=True)
class RefineConfig:
    time_budget_s: float = 210.0
    coarse_evaluations: int = 32
    top_k: int = 3
    window_s: float = 1.2
    seed: int = 6023
    weights: tuple[float, ...] = (1.0, 0.15, 0.10, 0.05, 2.0, 0.10, 1000.0)
    max_penetration_m: float = 0.01
    max_penetration_dimension_fraction: float = 0.20
    proximity_length_fraction: float = 0.15
    timing_scale_s: float = 0.2
    minimum_improvement_m: float = 0.0001
    minimum_relative_improvement: float = 0.001

    def __post_init__(self):
        for name in (
            "time_budget_s", "window_s", "max_penetration_m",
            "max_penetration_dimension_fraction", "proximity_length_fraction",
            "timing_scale_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.coarse_evaluations < 2 or self.top_k < 1:
            raise ValueError("Need at least two coarse evaluations and one finalist")
        if (
            len(self.weights) != 7 or not np.isfinite(self.weights).all()
            or min(self.weights) < 0 or self.weights[0] <= 0
        ):
            raise ValueError("Seven nonnegative weights, with positive position weight, required")
        if (
            not np.isfinite(self.minimum_improvement_m)
            or self.minimum_improvement_m < 0
            or not np.isfinite(self.minimum_relative_improvement)
            or not 0 <= self.minimum_relative_improvement < 1
        ):
            raise ValueError("Invalid measurable-improvement thresholds")


def _reflect(x):
    y = np.mod(x, 2.0)
    return np.where(y <= 1, y, 2 - y)


def bounded_cma_es(
    objective: Callable[[np.ndarray], float], x0: np.ndarray, *,
    max_evaluations: int = 32, seed: int = 6023, sigma: float = 0.22,
    deadline: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Small rank-mu CMA-ES with reflection at [0,1] boundaries.

    Returns completed evaluations in evaluation order. TimeoutError ends the
    search without manufacturing a score for an incomplete evaluation.
    """
    mean = np.asarray(x0, float).copy()
    if (
        mean.ndim != 1 or not len(mean) or not np.isfinite(mean).all()
        or np.any((mean < 0) | (mean > 1))
        or max_evaluations < 1 or not np.isfinite(sigma) or sigma <= 0
    ):
        raise ValueError("Invalid normalized CMA-ES configuration")
    rng = np.random.default_rng(seed)
    n = len(mean)
    population = max(4, 4 + int(3 * np.log(n)))
    mu = population // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    effective_mu = 1 / np.sum(weights ** 2)
    cc = (4 + effective_mu / n) / (n + 4 + 2 * effective_mu / n)
    cs = (effective_mu + 2) / (n + effective_mu + 5)
    c1 = 2 / ((n + 1.3) ** 2 + effective_mu)
    cmu = min(1 - c1, 2 * (effective_mu - 2 + 1 / effective_mu) / ((n + 2) ** 2 + effective_mu))
    damping = 1 + 2 * max(0.0, np.sqrt((effective_mu - 1) / (n + 1)) - 1) + cs
    expected_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    covariance = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    history = []

    def evaluate(x):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("CMA-ES deadline")
        score = float(objective(x.copy()))
        if not np.isfinite(score):
            score = LARGE_LOSS
        history.append((x.copy(), score))
        return score

    try:
        evaluate(mean)
        generation = 0
        while len(history) < max_evaluations:
            generation += 1
            eigenvalues, basis = np.linalg.eigh((covariance + covariance.T) / 2)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            transform = basis @ np.diag(np.sqrt(eigenvalues))
            inverse_sqrt = (basis / np.sqrt(eigenvalues)) @ basis.T
            batch = []
            for _ in range(min(population, max_evaluations - len(history))):
                x = _reflect(mean + sigma * (transform @ rng.normal(size=n)))
                batch.append((x, evaluate(x)))
            if len(batch) < mu:
                break
            batch.sort(key=lambda item: item[1])
            selected = np.array([item[0] for item in batch[:mu]])
            old = mean.copy()
            mean = weights @ selected
            y = (mean - old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * effective_mu) * (inverse_sqrt @ y)
            correction = np.sqrt(1 - (1 - cs) ** (2 * generation))
            hsig = float(
                np.linalg.norm(ps) / correction < (1.4 + 2 / (n + 1)) * expected_norm
            )
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * effective_mu) * y
            steps = (selected - old) / sigma
            covariance = (
                (1 - c1 - cmu) * covariance
                + c1 * (
                    np.outer(pc, pc)
                    + (1 - hsig) * cc * (2 - cc) * covariance
                )
                + cmu * np.einsum("i,ij,ik->jk", weights, steps, steps)
            )
            sigma *= np.exp((cs / damping) * (np.linalg.norm(ps) / expected_norm - 1))
            sigma = float(np.clip(sigma, 1e-10, 0.7))
    except TimeoutError:
        pass
    return history


def initial_parameters(base: PhysicsConfig) -> np.ndarray:
    if np.linalg.norm(base.hand_offset_m) > 1e-12 or base.timing_offset_s != 0:
        raise ValueError("Refinement baseline must have zero reference offsets")
    return np.array([
        (base.density - 300) / 900,
        (base.friction - 0.3) / 0.9,
        (base.hand_friction - 0.5),
        (base.contact_timeconst - 0.005) / 0.025,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])


def decode_parameters(x, base: PhysicsConfig, timestep=0.002):
    x = np.asarray(x, float)
    if x.shape != (12,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("Physics search parameters must be twelve values in [0,1]")
    weld, wrist, finger, force = 2.0 ** (2 * x[4:8] - 1)
    offset = (2 * x[8:11] - 1) * 0.03
    offset *= min(1.0, 0.03 / max(float(np.linalg.norm(offset)), 1e-12))
    config = replace(
        base, timestep=timestep,
        # Clip: float rounding at x=1 can exceed the validator's closed bounds.
        density=float(np.clip(300 + 900 * x[0], 300, 1200)),
        friction=float(np.clip(0.3 + 0.9 * x[1], 0.3, 1.2)),
        hand_friction=float(np.clip(0.5 + x[2], 0.5, 1.5)),
        contact_timeconst=float(np.clip(0.005 + 0.025 * x[3], 0.005, 0.03)),
        weld_timeconst=float(base.weld_timeconst * weld),
        wrist_kp=float(base.wrist_kp * wrist),
        wrist_kv=float(base.wrist_kv * np.sqrt(wrist)),
        wrist_force=float(base.wrist_force * wrist),
        finger_kp=float(base.finger_kp * finger),
        finger_kv=float(base.finger_kv * np.sqrt(finger)),
        finger_force=float(base.finger_force * force),
        hand_offset_m=tuple(map(float, offset)),
        timing_offset_s=float(-0.15 + 0.30 * x[11]),
    )
    values = (
        config.density, config.friction, config.hand_friction,
        config.contact_timeconst, float(weld), float(wrist), float(finger),
        float(force), *config.hand_offset_m, config.timing_offset_s,
    )
    return config, dict(zip(PARAMETER_NAMES, values))


def motion_window(observations: dict, duration_s: float) -> tuple[float, float]:
    times = np.asarray(observations["timestamps"], float)
    start = max(float(times[0]), float(observations.get("builder_clip_start_s", times[0])))
    end = min(float(times[-1]), float(observations.get("builder_clip_end_s", times[-1])))
    if end <= start:
        raise ValueError("No builder/reference temporal overlap")
    pose = observations["block_poses"]
    good = (
        observations["block_poses_observed_mask"]
        & (observations["block_poses_confidence"] > 0)
        & np.isfinite(pose).all(axis=-1)
    )
    source = observations.get("builder_source_frame_indices", np.arange(len(times)))
    energy = np.zeros(len(times))
    for j in range(pose.shape[1]):
        seen, previous = set(), None
        for i in np.flatnonzero(good[:, j] & (times >= start) & (times <= end)):
            if int(source[i]) < 0 or int(source[i]) in seen:
                continue
            seen.add(int(source[i]))
            if previous is not None and times[i] - times[previous] <= 0.3:
                energy[i] += np.linalg.norm(pose[i, j, :3] - pose[previous, j, :3])
            previous = i
    width = min(duration_s, end - start)
    candidates = np.unique(np.r_[start, np.clip(times, start, end - width)])
    scores = [
        energy[(times >= left) & (times <= left + width)].sum()
        for left in candidates
    ]
    left = float(candidates[int(np.argmax(scores))])
    return left, left + width


def reference_scope(observations, window):
    if window is None:
        return observations
    result = dict(observations)
    mask = observations["block_poses_observed_mask"].copy()
    times = observations["timestamps"]
    mask[(times < window[0]) | (times > window[1])] = False
    result["block_poses_observed_mask"] = mask
    return result


def observed_contact_events(observations, config: RefineConfig, window=None):
    """Unique observed 3D proximity onsets; held blocks/hands are excluded.

    These remain uncertain observation-derived events, not contact ground truth.
    Inferred M5 contact-window flags are deliberately not used as evidence.
    """
    times = observations["timestamps"]
    source = observations.get("builder_source_frame_indices", np.arange(len(times)))
    events = {int(identity): [] for identity in observations["block_ids"]}
    samples = 0
    for j, identity in enumerate(observations["block_ids"]):
        seen, previous_near, previous_time = set(), False, -np.inf
        for i, stamp in enumerate(times):
            if window is not None and not window[0] <= stamp <= window[1]:
                continue
            sid = int(source[i])
            if sid < 0 or sid in seen:
                continue
            seen.add(sid)
            pose = observations["block_poses"][i, j]
            dims = observations["block_dimensions"][i, j]
            if (
                not observations["block_poses_observed_mask"][i, j]
                or observations["block_poses_confidence"][i, j] <= 0
                or not np.isfinite(np.r_[pose, dims]).all()
            ):
                previous_near = False
                continue
            distances = []
            for side in ("left", "right"):
                name = f"builder_{side}_hand_21"
                if name not in observations:
                    continue
                points = observations[name][i]
                valid = (
                    observations[f"{name}_observed_mask"][i]
                    & (observations[f"{name}_confidence"][i] > 0)
                    & np.isfinite(points).all(axis=-1)
                )
                if valid.any():
                    distances.append(surface_distance(points[valid], pose, dims))
            if not distances:
                previous_near = False
                continue
            samples += 1
            near = min(distances) <= config.proximity_length_fraction * max(dims)
            if near and (not previous_near or stamp - previous_time > 0.3):
                events[int(identity)].append(float(stamp))
            previous_near, previous_time = near, stamp
    return events, samples


def contact_timing_loss(rollout, events, config, window=None):
    errors = []
    times = rollout["timestamps"]
    for j, identity in enumerate(rollout["block_ids"]):
        flags = rollout["hand_block_contact"][:, j]
        onset = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
        simulated = times[onset]
        if window is not None:
            simulated = simulated[(simulated >= window[0]) & (simulated <= window[1])]
        observed = np.asarray(events.get(int(identity), []))
        if not len(observed) and not len(simulated):
            continue
        if len(observed) and len(simulated):
            cost = np.minimum(
                np.abs(observed[:, None] - simulated[None, :]) / config.timing_scale_s, 1
            )
            rows, cols = linear_sum_assignment(cost)
            errors.extend(cost[rows, cols].tolist())
            errors.extend([1.0] * abs(len(observed) - len(simulated)))
        else:
            errors.extend([1.0] * max(len(observed), len(simulated)))
    return float(np.mean(errors)) if errors else 0.0


def objective(
    arrays, observations, targets, model, config: RefineConfig, *,
    window=None, proximity=None,
):
    scoped = reference_scope(observations, window)
    metrics = physics_metrics(arrays, scoped, model)
    dims = observations["block_dimensions"]
    good_dims = dims[np.isfinite(dims).all(axis=-1) & (dims > 0).all(axis=-1)]
    length = float(np.median(np.max(good_dims, axis=-1))) if len(good_dims) else 1.0
    smallest = float(np.min(good_dims)) if len(good_dims) else 0.0
    bound = min(config.max_penetration_m, config.max_penetration_dimension_fraction * smallest)
    reasons = []
    for condition, reason in (
        (not metrics["stable"] or metrics["has_nans"], "unstable_or_nan"),
        (not metrics["rollout_complete"], "incomplete_rollout"),
        (bool(metrics["block_fall_events"]), "block_fell"),
        (metrics["max_penetration_m"] > bound, "penetration_bound"),
        (metrics["no_block_actuation_audit"] != "passed", "block_actuation"),
        (metrics["block_write_guard"] != "enabled throughout pre-roll and rollout", "guard_absent"),
        (metrics["block_reference_evaluated_samples"] == 0, "no_reference_samples"),
        (bool(metrics["blocks_moved_by_contact"]["violations"]), "unattributed_block_motion"),
    ):
        if condition:
            reasons.append(reason)

    events, proximity_samples = (
        observed_contact_events(observations, config, window)
        if proximity is None else proximity
    )
    timing = contact_timing_loss(arrays, events, config, window) if proximity_samples else 0.0
    reference = TargetInterpolator(model, targets)
    indices = np.flatnonzero(
        np.ones(len(arrays["timestamps"]), bool) if window is None else
        (arrays["timestamps"] >= window[0]) & (arrays["timestamps"] <= window[1])
    )
    stride = max(1, int(round(0.04 / model.opt.timestep)))
    hand_errors = []
    chains = {str(name): j for j, name in enumerate(targets.get("chain_names", []))}
    for k in indices[::stride]:
        stamp = float(arrays["timestamps"][k])
        target = reference.at(stamp)
        ti = int(np.clip(np.searchsorted(targets["timestamps"], stamp), 0, len(targets["timestamps"]) - 1))
        for side in ("left", "right"):
            root = f"builder_{side}_root"
            if root in chains and not targets["validity_mask"][ti, chains[root]]:
                continue
            address = int(model.joint(root).qposadr[0])
            hand_errors.append(float(np.linalg.norm(
                arrays["qpos"][k, address:address + 3] - target[address:address + 3]
            )) / length)

    position = metrics["block_reference_diagnostic_normalized_error"]
    rotation = metrics["block_reference_diagnostic_rotation_error_rad"]
    terms = dict(zip(TERM_NAMES, (
        float(position) if position is not None else LARGE_LOSS,
        0.5 * float(rotation) if rotation is not None else LARGE_LOSS,
        timing,
        float(np.mean(hand_errors)) if hand_errors else 0.0,
        metrics["max_penetration_m"] / length,
        metrics["max_joint_limit_overshoot_rad"],
        float(any(reason in reasons for reason in (
            "unstable_or_nan", "incomplete_rollout", "block_fell"
        ))),
    )))
    loss = float(np.dot(config.weights, list(terms.values())))
    if not np.isfinite(loss):
        loss = LARGE_LOSS
        reasons.append("nonfinite_loss")
    return {
        "loss": loss, "loss_terms": terms, "valid": not reasons,
        "invalid_reasons": reasons, "metrics": metrics,
        "normalization_block_length_m": length,
        "penetration_validity_bound_m": bound,
        "proximity_evaluable_samples": proximity_samples,
        "hand_reference_evaluable_samples": len(hand_errors),
    }


def improvement(initial, refined):
    result = {}
    for name, before, after in (
        (
            "block_position_error_m",
            initial["metrics"]["block_trajectory_position_error_m"],
            refined["metrics"]["block_trajectory_position_error_m"],
        ),
        ("total_loss", initial["loss"], refined["loss"]),
    ):
        result[name] = {
            "initial": before, "refined": after,
            "absolute_reduction": before - after if before is not None and after is not None else None,
            "relative_reduction": (
                (before - after) / abs(before)
                if before is not None and after is not None and abs(before) > 1e-12 else None
            ),
        }
    return result


def refine_physics(
    kinematic_xml: str, targets: dict, observations: dict, initial: dict,
    base: PhysicsConfig = PhysicsConfig(), config: RefineConfig = RefineConfig(),
) -> dict:
    started = time.monotonic()
    deadline = started + config.time_budget_s
    if base.timestep != 0.002:
        raise ValueError("M6 initial/full-resolution comparison requires timestep=0.002")
    if not 0.04 <= base.weld_timeconst <= 0.1:
        raise ValueError("Baseline weld timeconst must allow bounded x0.5..x2 search")
    x0 = initial_parameters(base)
    initial_xml = physics_variant(kinematic_xml, base)
    initial_model = mujoco.MjModel.from_xml_string(initial_xml)
    full_proximity = observed_contact_events(observations, config)
    initial_score = objective(
        initial, observations, targets, initial_model, config, proximity=full_proximity
    )
    _, initial_params = decode_parameters(x0, base)
    history = [{
        "phase": "initial_full", "normalized_params": x0.tolist(),
        "params": initial_params, **initial_score,
    }]
    selected_arrays, selected_xml, selected_config = initial, initial_xml, base
    selected_score, selected_index = initial_score, 0
    best_valid_loss = initial_score["loss"] if initial_score["valid"] else np.inf
    window = None
    skip_reason = None
    if not len(observations["block_ids"]):
        skip_reason = "no_blocks"
    elif initial_score["metrics"]["block_reference_evaluated_samples"] == 0:
        skip_reason = "no_evaluable_block_reference"
    else:
        try:
            window = motion_window(observations, config.window_s)
        except ValueError as exc:
            skip_reason = str(exc)

    coarse_candidates = []
    if skip_reason is None:
        proximity = observed_contact_events(observations, config, window)
        initial_seconds = float(initial.get("physics_compute_s", 10.0))
        reserve = min(config.time_budget_s * 0.5, max(5.0, initial_seconds * 1.5 * config.top_k))
        coarse_deadline = deadline - reserve

        def evaluate(x, phase, candidate_deadline):
            nonlocal selected_arrays, selected_xml, selected_config
            nonlocal selected_score, selected_index, best_valid_loss
            candidate_config, params = decode_parameters(
                x, base, 0.004 if phase == "coarse" else 0.002
            )
            entry = {
                "phase": phase, "normalized_params": x.tolist(),
                "params": params, "timestep_s": candidate_config.timestep,
            }
            before = time.monotonic()
            try:
                xml = physics_variant(kinematic_xml, candidate_config)
                arrays = run_rollout(
                    xml, targets, observations, candidate_config,
                    stop_time_s=window[1] if phase == "coarse" else None,
                    deadline=candidate_deadline,
                )
                score = objective(
                    arrays, observations, targets, mujoco.MjModel.from_xml_string(xml),
                    config, window=window if phase == "coarse" else None,
                    proximity=proximity if phase == "coarse" else full_proximity,
                )
                entry.update(score)
                if phase == "coarse":
                    coarse_candidates.append((x.copy(), score))
                elif score["valid"] and score["loss"] < best_valid_loss:
                    best_valid_loss = score["loss"]
                    selected_arrays, selected_xml, selected_config = arrays, xml, candidate_config
                    selected_score, selected_index = score, len(history)
            except TimeoutError as exc:
                entry.update(
                    loss=LARGE_LOSS, loss_terms={}, valid=False,
                    invalid_reasons=["time_budget"], error=str(exc),
                )
                raise
            except (ValueError, RuntimeError) as exc:
                # A guard violation is an implementation failure, not a bad fit.
                if "block qpos" in str(exc) or "block qvel" in str(exc) or "Unauthorized" in str(exc):
                    raise
                entry.update(
                    loss=LARGE_LOSS, loss_terms={}, valid=False,
                    invalid_reasons=["candidate_simulation_failed"], error=str(exc),
                )
            finally:
                entry["wall_s"] = time.monotonic() - before
                history.append(entry)
            return entry["loss"] if entry["valid"] else LARGE_LOSS + min(entry["loss"], LARGE_LOSS)

        bounded_cma_es(
            lambda x: evaluate(x, "coarse", coarse_deadline),
            x0, max_evaluations=config.coarse_evaluations,
            seed=config.seed, deadline=coarse_deadline,
        )
        # Fine reevaluation is authoritative. Invalid coarse candidates may
        # recover at the finer step, but rank behind valid coarse candidates.
        coarse_candidates.sort(key=lambda item: (not item[1]["valid"], item[1]["loss"]))
        unique = set()
        finalists = []
        for x, _ in coarse_candidates:
            key = tuple(np.round(x, 12))
            if key in unique or np.allclose(x, x0, atol=1e-12, rtol=0):
                continue
            unique.add(key)
            finalists.append(x)
            if len(finalists) >= config.top_k:
                break
        for x in finalists:
            if time.monotonic() >= deadline:
                break
            try:
                evaluate(x, "fine_full", deadline)
            except TimeoutError:
                break

    changes = improvement(initial_score, selected_score)
    delta = changes["block_position_error_m"]["absolute_reduction"]
    baseline_error = changes["block_position_error_m"]["initial"]
    threshold = max(
        config.minimum_improvement_m,
        config.minimum_relative_improvement * (baseline_error or 0),
    )
    passed = bool(
        selected_score["valid"] and selected_index != 0
        and selected_score["metrics"]["block_reference_sufficient"]
        and delta is not None and delta > threshold
    )
    report = {
        "schema": "inverse_physics_v1",
        "optimizer": "serial bounded rank-mu CMA-ES, reflected [0,1] coordinates",
        "seed": config.seed, "config": asdict(config),
        "parameter_names": list(PARAMETER_NAMES),
        "window_s": list(window) if window is not None else None,
        "short_rollout_policy": (
            "Simulate original initialization and entire prefix to window end; "
            "score references only in the motion window. Never reset to a later block pose."
        ),
        "objective_units": {
            "block_position": "mean position error / each block's longest side",
            "block_rotation": "0.5 * symmetry-reduced radians (length-scaled arc / length)",
            "contact_timing": "one-to-one onset error / timing_scale_s, capped at 1; unmatched=1",
            "hand_reference": "dynamic builder root deviation from ORIGINAL reference / median block length",
            "penetration": "maximum penetration / median block length",
            "joint_limit": "maximum angular overshoot, radians (dimensionless)",
            "instability": "1 for NaN/explosion/fall/incomplete rollout",
        },
        "contact_timing_evidence": (
            "Observed finite hand landmarks vs observed cuboids; unique source "
            "frames, no held samples. These uncertain 3D proximity events may "
            "already include upstream depth priors and are not contact ground truth."
        ),
        "history": history,
        "selected_history_index": selected_index,
        "selected_config": asdict(selected_config),
        "selected_valid": selected_score["valid"],
        "selection_status": (
            "refined_valid" if selected_index else
            "initial_valid_retained" if selected_score["valid"] else
            "no_valid_rollout_initial_retained_for_diagnostics"
        ),
        "skip_reason": skip_reason,
        "physics_initial": initial_score["metrics"],
        "physics_refined": selected_score["metrics"],
        "initial_total_loss": initial_score["loss"],
        "refined_total_loss": selected_score["loss"],
        "refinement_improvement": changes,
        "measurable_position_improvement_threshold_m": threshold,
        "m6_pass": passed,
        "wall_s": time.monotonic() - started,
        "time_budget_s": config.time_budget_s,
        "coarse_completed": sum(
            e["phase"] == "coarse" and "metrics" in e for e in history
        ),
        "fine_completed": sum(
            e["phase"] == "fine_full" and "metrics" in e for e in history
        ),
        "limitations": [
            "A fixed seed reproduces evaluation order; wall-time cutoff may change evaluation count.",
            "Serial search is used on Windows; no multiprocessing or new dependencies.",
            "MuJoCo refsafety clamps solref below twice the current timestep.",
            "Eight signed quaternion representatives encode four distinct proper cuboid rotations.",
            "No improvement is guaranteed in a short, contact-discontinuous, partially identifiable search.",
        ],
    }
    result = dict(selected_arrays)
    final_metrics = dict(selected_score["metrics"])
    final_metrics["m6_pass"] = passed
    result["metrics_json"] = np.array(json.dumps(final_metrics, allow_nan=False))
    result["refinement_json"] = np.array(json.dumps(report, allow_nan=False))
    result["refined_scene_xml"] = np.array(selected_xml)
    return result
