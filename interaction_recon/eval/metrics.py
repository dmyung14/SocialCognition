"""Spec 19 sections and spec 20 targets. Missing evidence never passes a target."""
import numpy as np

TARGETS = {
    "visible_hand_tracking_coverage": (">=", 0.80, "fraction"),
    "mean_visible_landmark_reprojection": ("<=", 15.0, "px"),
    "block_centroid_reprojection": ("<=", 20.0, "px"),
    "physics_block_trajectory": ("<=", 0.25, "block_lengths"),
    "persistent_penetration": ("<=", 0.10, "smallest_block_dimension"),
}


def target_table(values: dict, details: dict | None = None) -> dict:
    result = {}
    for name, (operator, threshold, units) in TARGETS.items():
        value = values.get(name)
        evaluable = value is not None and bool(np.isfinite(value))
        passed = bool(
            evaluable and (
                value >= threshold if operator == ">=" else value <= threshold
            )
        )
        result[name] = {
            "value": float(value) if evaluable else None,
            "operator": operator,
            "threshold": threshold,
            "units": units,
            "pass": passed,
            "status": "passed" if passed else "failed" if evaluable else "not_evaluable",
            "detail": (details or {}).get(name, ""),
        }
    return result


def _mean(values) -> float | None:
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def _ratio(value, scale):
    return float(value / scale) if value is not None and scale is not None and scale > 0 else None


def persistent_penetration(rollout: dict, duration_s: float = 0.1) -> float | None:
    """Maximum depth sustained by any block for at least duration_s.

    Per-block depth is its deepest solver contact in each substep. Contact
    points may switch, but the same block must remain penetrated throughout.
    """
    times = rollout["timestamps"]
    ids = rollout["block_ids"]
    if not len(ids) or len(times) < 2:
        return None
    dt = float(np.median(np.diff(times)))
    if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-7, rtol=1e-4):
        return None
    window = int(np.ceil(duration_s / dt - 1e-8)) + 1
    if len(times) < window:
        return None
    lookup = {f"block_{int(identity)}": j for j, identity in enumerate(ids)}
    bodies = rollout["geom_body_names"]
    nodes = np.array([lookup.get(str(name), -1) for name in bodies], int)
    depths = np.zeros((len(times), len(ids)))
    offsets = rollout["contact_offsets"]
    for k in range(len(times)):
        a, b = int(offsets[k]), int(offsets[k + 1])
        for pair, depth in zip(rollout["contacts"][a:b], rollout["penetration_depths"][a:b]):
            for geom in pair:
                j = nodes[geom]
                if j >= 0:
                    depths[k, j] = max(depths[k, j], float(depth))
    windows = np.lib.stride_tricks.sliding_window_view(depths, window, axis=0)
    return float(np.max(np.min(windows, axis=-1), initial=0))


def consolidate_metrics(existing: dict, observations: dict, rollout: dict) -> dict:
    # Retain legacy top-level diagnostics for existing callers and old analyses.
    result = dict(existing)
    streams = existing.get("streams", {})
    physics = dict(existing.get("physics", {}))
    retarget = existing.get("retarget", {})
    dimensions = observations["block_dimensions"]
    good = dimensions[np.isfinite(dimensions).all(axis=-1) & (dimensions > 0).all(axis=-1)]
    length = float(np.median(np.max(good, axis=-1))) if len(good) else None
    smallest = float(np.min(good)) if len(good) else None
    persistent = persistent_penetration(rollout)
    physics.update({
        "max_persistent_penetration_m": persistent,
        "persistent_penetration_duration_s": 0.1,
        "persistent_penetration_normalized_by_smallest_dimension": _ratio(persistent, smallest),
        "max_penetration_normalized_by_smallest_dimension": _ratio(
            physics.get("max_penetration_m"), smallest
        ),
        "persistent_penetration_definition": (
            "Maximum minimum per-block solver-contact depth over any continuous "
            "100ms substep window; the deepest contact point may switch."
        ),
    })
    perception_streams, reconstruction_streams = {}, {}
    for role, values in streams.items():
        perception_streams[role] = {
            "hand_detection_coverage": {
                "frames_with_any_hand": values.get("frames_with_any_hand"),
                "frames_with_wearer_hand": values.get("frames_with_wearer_hand"),
                "visible_hand_recall": None,
                "detail": (
                    "Frame availability, not recall over independently annotated visible hands. "
                    "No independent visibility denominator is available."
                ),
            },
            "body_keypoint_coverage": values.get("body_keypoint_coverage"),
            "block_tracking_coverage": {
                "frames_with_any_block": values.get("frames_with_any_block"),
                "tracks": values.get("block_tracks", []),
            },
            "camera_pose_coverage": values.get("camera_pose_coverage"),
            "reprojection_error_px": values.get("lifted_keypoint_reprojection_error_px"),
            "reprojection_scope": values.get("reprojection_scope"),
            "mean_visible_landmark_reprojection_px": None,
            "block_centroid_reprojection_px": None,
        }
        table = values.get("table_plane_consistency", {})
        variance = values.get("block_dimension_variance_m2")
        reconstruction_streams[role] = {
            "table_plane_consistency": {
                **table,
                "rest_contact_residual_normalized_by_block_length": _ratio(
                    table.get("rest_contact_residual_m"), length
                ),
                "detail": "Within-view table evidence; not verified cross-view plane agreement.",
            },
            "block_dimension_variance_m2": variance,
            "block_dimension_variance_normalized_by_length_squared": _ratio(
                variance, length * length if length is not None else None
            ),
            "scale": values.get("scale"),
        }
    residuals = [
        record["mean"] for record in retarget.get("ik_residual_m", {}).values()
        if record.get("mean") is not None
    ]
    ik = _mean(residuals)
    result.update({
        "schema": "metrics_m7_v1",
        "scope": "Estimated observations, retargeting and contact physics; not ground truth",
        "perception": {
            "streams": perception_streams,
            "synchronization_confidence": existing.get("synchronization_confidence"),
            "visible_hand_tracking_coverage": None,
            "mean_visible_landmark_reprojection_px": None,
            "block_centroid_reprojection_px": None,
            "unavailable_metric_reason": (
                "Visibility-conditioned recall and final-scene mean reprojection have "
                "not been independently measured. Raw lifting residuals and frame "
                "availability are retained as diagnostics, not substituted as success."
            ),
        },
        "reconstruction": {
            "cross_view_consistency": existing.get("cross_view_consistency"),
            "cross_view_landmark_consistency": existing.get("cross_view_landmark_consistency"),
            "streams": reconstruction_streams,
            "ik_residual_m": retarget.get("ik_residual_m", {}),
            "mean_chain_ik_residual_m": ik,
            "mean_chain_ik_residual_normalized_by_block_length": _ratio(ik, length),
            "joint_limit_violations": retarget.get("joint_limit_violations"),
            "max_joint_limit_overshoot_rad": retarget.get("max_joint_limit_overshoot_rad"),
            "retarget_coverage": retarget.get("retarget_coverage", {}),
        },
        "physics": physics,
        "normalization": {
            "median_longest_block_side_m": length,
            "smallest_block_dimension_m": smallest,
            "scale_is_estimated": True,
            "trajectory": "Each sample normalized by that block's longest side.",
            "other_lengths": "Median longest block side; penetration uses smallest dimension.",
        },
    })
    result["targets"] = target_table({
        "physics_block_trajectory": physics.get("block_trajectory_error_normalized_by_length"),
        "persistent_penetration": physics[
            "persistent_penetration_normalized_by_smallest_dimension"
        ],
    }, {
        "visible_hand_tracking_coverage": "Independent visible-hand denominator unavailable.",
        "mean_visible_landmark_reprojection": (
            "Raw pre-prior lifting mean-of-medians is not final mean visible-landmark reprojection."
        ),
        "block_centroid_reprojection": "Independent final-scene centroid reprojection unavailable.",
        "physics_block_trajectory": (
            "Accepted unique observed reference samples only; insufficient coverage is not a pass."
        ),
        "persistent_penetration": "100ms persistence; denominator is smallest estimated block dimension.",
    })
    result["all_measured_targets_pass"] = all(
        item["pass"] for item in result["targets"].values()
    )
    result["target_note"] = (
        "Spec 20 engineering targets, not metric ground-truth accuracy claims. "
        "M6 measurable improvement is separate from the <=0.25 BL absolute target."
    )
    return result
