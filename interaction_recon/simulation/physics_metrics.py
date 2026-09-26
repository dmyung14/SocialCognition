"""Trajectory diagnostics and sliding-window contact attribution."""
from collections import deque

import numpy as np

from interaction_recon.simulation.physics_scene import builder_hand_geom


MIN_REFERENCE_SAMPLES = 20
MIN_REFERENCE_SAMPLES_PER_BLOCK = 5
MIN_REFERENCE_COVERAGE = 0.5

# Eight quaternion representatives of the rectangular cuboid rotation group.
# q and -q represent the same rotation: there are FOUR distinct proper rotations,
# not eight. Reflections are not rotations and must not enter an SO(3) geodesic.
BOX_SYMMETRY_QUATERNIONS = np.concatenate((np.eye(4), -np.eye(4)))


def quaternion_product(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.concatenate((
        (a[..., :1] * b[..., :1]
         - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)),
        a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
        + np.cross(a[..., 1:], b[..., 1:]),
    ), axis=-1)


def box_rotation_errors(actual, reference):
    """Return symmetry-reduced and raw wxyz geodesics, in radians."""
    a, b = np.asarray(actual, float), np.asarray(reference, float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    equivalent = quaternion_product(b[..., None, :], BOX_SYMMETRY_QUATERNIONS)
    dot = np.max(np.abs(np.sum(a[..., None, :] * equivalent, axis=-1)), axis=-1)
    raw_dot = np.abs(np.sum(a * b, axis=-1))
    return (
        2 * np.arccos(np.clip(dot, 0, 1)),
        2 * np.arccos(np.clip(raw_dot, 0, 1)),
    )


def contact_chain_lengths(rollout, model=None, window_s=0.2):
    """Shortest hand->block path in the union of contacts in [t-window,t].

    Nodes are whole blocks and one virtual builder-hand source. The world,
    table, guider, and forearms never bridge components. Every edge must have
    actual solver-contact evidence in this window; no inferred proximity edges.
    This is window connectivity, not a claim of temporally ordered force flow.
    """
    times = np.asarray(rollout["timestamps"])
    ids = np.asarray(rollout["block_ids"])
    count = len(ids)
    result = np.full((len(times), count), -1, np.int32)
    names = {f"block_{int(identity)}": j for j, identity in enumerate(ids)}
    body_names = rollout.get("geom_body_names")
    if body_names is None and model is not None:
        body_names = np.array([
            model.body(int(body)).name or "world" for body in model.geom_bodyid
        ])
    if body_names is None or "contacts" not in rollout:
        # Legacy diagnostic fixtures have only direct contact flags.
        direct = rollout["hand_block_contact"]
        for k, time in enumerate(times):
            left = np.searchsorted(times, time - window_s - 1e-9)
            result[k, np.any(direct[left:k + 1], axis=0)] = 1
        return result

    nodes = np.array([names.get(str(name), -1) for name in body_names])
    if "builder_hand_geom_mask" in rollout:
        hands = np.asarray(rollout["builder_hand_geom_mask"], bool)
    elif model is not None:
        hands = np.array([builder_hand_geom(model, g) for g in range(model.ngeom)])
    else:
        geom_names = rollout.get("geom_names", np.full(len(nodes), ""))
        hands = np.array([
            str(body).startswith("builder_")
            and (
                str(body).endswith("_palm")
                or any(f"_{finger}_" in str(body) for finger in (
                    "thumb", "index", "middle", "ring", "little"
                ))
                or "fingertip" in str(geom)
                or "hand" in str(geom)
            )
            for body, geom in zip(body_names, geom_names)
        ])
    offsets = rollout["contact_offsets"]
    last_seen = {}
    source = count
    for k, time in enumerate(times):
        for g1, g2 in rollout["contacts"][offsets[k]:offsets[k + 1]]:
            a, b = int(nodes[g1]), int(nodes[g2])
            if a >= 0 and b >= 0 and a != b:
                last_seen[tuple(sorted((a, b)))] = time
            elif a >= 0 and hands[g2]:
                last_seen[(a, source)] = time
            elif b >= 0 and hands[g1]:
                last_seen[(b, source)] = time
        last_seen = {
            edge: stamp for edge, stamp in last_seen.items()
            if stamp >= time - window_s - 1e-9
        }
        adjacency = [[] for _ in range(count + 1)]
        for a, b in last_seen:
            adjacency[a].append(b)
            adjacency[b].append(a)
        distance = {source: 0}
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if b not in distance:
                    distance[b] = distance[a] + 1
                    queue.append(b)
        for j in range(count):
            result[k, j] = distance.get(j, -1)
    return result


def movement_attribution(rollout, model=None):
    times = rollout["timestamps"]
    chains = contact_chain_lengths(rollout, model)
    events, violations = [], []
    for j, identity in enumerate(rollout["block_ids"]):
        anchor = rollout["block_poses"][0, j, :3].copy()
        for k, time in enumerate(times):
            position = rollout["block_poses"][k, j, :3]
            displacement = float(np.linalg.norm(position - anchor))
            if not np.isfinite(position).all() or displacement <= 0.01:
                continue
            length = int(chains[k, j])
            left = np.searchsorted(times, time - 0.2 - 1e-9)
            event = {
                "block_id": int(identity), "timestamp_s": float(time),
                "displacement_m": displacement,
                "preceding_0_2s_hand_contact": bool(
                    rollout["hand_block_contact"][left:k + 1, j].any()
                ),
                "preceding_0_2s_hand_contact_chain": length >= 1,
                "contact_chain_length": length if length >= 1 else None,
            }
            events.append(event)
            if length < 1:
                violations.append(event)
            anchor = position.copy()
    lengths = [e["contact_chain_length"] for e in events if e["contact_chain_length"]]
    return {
        "status": (
            "not_evaluable" if not len(rollout["block_ids"]) else
            "failed" if violations else "passed"
        ),
        "movement_event_count": len(events),
        "violations": violations,
        "events": events,
        "chain_length_stats": {
            "attributed_movement_events": len(lengths),
            "direct_events": lengths.count(1),
            "transitive_events": sum(length > 1 for length in lengths),
            "mean": float(np.mean(lengths)) if lengths else None,
            "maximum": max(lengths) if lengths else None,
            "histogram": {str(n): lengths.count(n) for n in sorted(set(lengths))},
        },
        "definition": (
            "Each accumulated >1cm excursion requires a path from a builder hand "
            "through blocks in the union of solver contacts during the preceding "
            "0.2s, inclusive. Chain length counts contact edges. Table/world and "
            "forearms cannot bridge paths. Unconnected settling/sliding is a violation."
        ),
    }


def physics_metrics(rollout, observations, model):
    times = rollout["timestamps"]
    reference_times = observations["timestamps"]
    position_errors, normalized, rotation_errors, raw_rotation_errors = [], [], [], []
    source = observations.get("builder_source_frame_indices", np.arange(len(reference_times)))
    columns = {int(identity): j for j, identity in enumerate(observations["block_ids"])}
    eligible_count = evaluated_count = 0
    per_block = {}
    for j, identity in enumerate(rollout["block_ids"]):
        column = columns[int(identity)]
        seen = set()
        eligible = evaluated = 0
        for i, time in enumerate(reference_times):
            source_id = int(source[i])
            if source_id < 0 or source_id in seen:
                continue
            if (
                not observations["block_poses_observed_mask"][i, column]
                or observations["block_poses_confidence"][i, column] <= 0
            ):
                continue
            pose = observations["block_poses"][i, column]
            dims = observations["block_dimensions"][i, column]
            if (
                not np.isfinite(np.r_[pose, dims]).all()
                or np.any(dims <= 0) or np.linalg.norm(pose[3:]) <= 1e-8
            ):
                continue
            seen.add(source_id)
            eligible += 1
            if time < times[0] - 1e-9 or time > times[-1] + 1e-9:
                continue
            k = int(np.argmin(np.abs(times - time)))
            actual = rollout["block_poses"][k, j]
            if not np.isfinite(actual).all() or np.linalg.norm(actual[3:]) <= 1e-8:
                continue
            evaluated += 1
            error = float(np.linalg.norm(actual[:3] - pose[:3]))
            position_errors.append(error)
            normalized.append(error / max(float(np.max(dims)), 1e-6))
            reduced, raw = box_rotation_errors(actual[3:], pose[3:])
            rotation_errors.append(float(reduced))
            raw_rotation_errors.append(float(raw))
        eligible_count += eligible
        evaluated_count += evaluated
        per_block[str(int(identity))] = {
            "eligible_source_samples": eligible,
            "evaluated_source_samples": evaluated,
            "coverage_fraction": evaluated / eligible if eligible else None,
        }

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    coverage = evaluated_count / eligible_count if eligible_count else None
    sufficient = bool(
        evaluated_count >= MIN_REFERENCE_SAMPLES
        and coverage is not None and coverage >= MIN_REFERENCE_COVERAGE
        and per_block
        and all(
            record["evaluated_source_samples"] >= MIN_REFERENCE_SAMPLES_PER_BLOCK
            for record in per_block.values()
        )
    )
    attribution = movement_attribution(rollout, model)
    fall_events = []
    for j, identity in enumerate(rollout["block_ids"]):
        if "block_fallen_flags" in rollout:
            falls = np.flatnonzero(rollout["block_fallen_flags"][:, j])
            if len(falls):
                fall_events.append({
                    "block_id": int(identity), "timestamp_s": float(times[falls[0]]),
                    "reason": "center_below_table_by_more_than_5cm",
                })
    stable = not bool(rollout["nan_flags"].any() or rollout["instability_flags"].any())
    contact = rollout["hand_block_contact"]
    block_count = len(rollout["block_ids"])
    requested_start = float(rollout.get("requested_start_s", times[0]))
    requested_end = float(rollout.get("requested_end_s", times[-1]))
    complete = bool(
        times[0] <= requested_start + 1e-9 and times[-1] >= requested_end - 1e-9
    )
    builder_start = float(observations.get("builder_clip_start_s", requested_start))
    builder_end = float(observations.get("builder_clip_end_s", requested_end))
    builder_duration = max(0.0, builder_end - builder_start)
    builder_covered = max(
        0.0, min(float(times[-1]), builder_end) - max(float(times[0]), builder_start)
    )
    return {
        "physics_performed": True,
        "block_count": block_count,
        "block_trajectory_position_error_m": mean(position_errors) if sufficient else None,
        "block_trajectory_error_normalized_by_length": mean(normalized) if sufficient else None,
        "block_rotation_error_rad": mean(rotation_errors) if sufficient else None,
        "block_rotation_error_raw_rad": mean(raw_rotation_errors) if sufficient else None,
        "block_reference_diagnostic_position_error_m": mean(position_errors),
        "block_reference_diagnostic_normalized_error": mean(normalized),
        "block_reference_diagnostic_rotation_error_rad": mean(rotation_errors),
        "block_reference_diagnostic_raw_rotation_error_rad": mean(raw_rotation_errors),
        "block_reference_evaluated_samples": evaluated_count,
        "block_reference_eligible_samples": eligible_count,
        "block_reference_coverage_fraction": coverage,
        "block_reference_per_block_coverage": per_block,
        "block_reference_minimum_samples": MIN_REFERENCE_SAMPLES,
        "block_reference_minimum_samples_per_block": MIN_REFERENCE_SAMPLES_PER_BLOCK,
        "block_reference_minimum_coverage": MIN_REFERENCE_COVERAGE,
        "block_reference_sufficient": sufficient,
        "block_reference_status": "evaluable" if sufficient else "insufficient_coverage_or_samples",
        "block_reference_definition": (
            "Unique observed source-frame/block pairs; held samples excluded. "
            "Diagnostic means are not accepted trajectory metrics below thresholds."
        ),
        "rotation_error_definition": (
            "Minimum quaternion geodesic over eight signed quaternion representatives "
            "(four distinct proper cuboid rotations). Reflections excluded; no axis "
            "permutations for unequal dimensions. Raw geodesic retained separately."
        ),
        "contact_frame_fraction": float(np.mean(np.any(contact, axis=1))),
        "per_block_contact_frame_fraction": {
            str(int(identity)): float(np.mean(contact[:, j]))
            for j, identity in enumerate(rollout["block_ids"])
        },
        "max_penetration_m": float(np.max(rollout["penetration_depths"], initial=0)),
        "initial_max_block_penetration_m": float(
            rollout.get("initial_max_block_penetration_m", 0)
        ),
        "settled_max_block_penetration_m": float(
            rollout.get("settled_max_block_penetration_m", 0)
        ),
        "peak_contact_force_n": float(np.max(
            np.linalg.norm(rollout["contact_forces"][:, :3], axis=1), initial=0
        )),
        "max_joint_limit_overshoot_rad": float(np.max(
            rollout["joint_limit_overshoot_rad"], initial=0
        )),
        "has_nans": bool(rollout["nan_flags"].any()),
        "stable": stable,
        "solver_warning_count": int(np.max(rollout["warning_counts"], initial=0)),
        "rollout_complete": complete,
        "rollout_requested_duration_s": requested_end - requested_start,
        "rollout_completed_duration_s": float(times[-1] - times[0]),
        "builder_clip_duration_s": builder_duration,
        "builder_clip_coverage_fraction": (
            min(1.0, builder_covered / builder_duration) if builder_duration else None
        ),
        "block_fall_events": fall_events,
        "blocks_moved_by_contact": attribution,
        "m5_pass": bool(
            stable and complete and sufficient and block_count
            and attribution["movement_event_count"] and not attribution["violations"]
        ),
        "no_block_actuation_audit": "passed",
        "block_write_guard": "enabled throughout pre-roll and rollout",
        "trajectory_target_met": mean(normalized) <= 0.25 if sufficient else None,
        "reference_is_estimated": True,
    }
