"""Gap-limited, provenance-preserving filtering. Quaternions use wxyz."""
import numpy as np


def slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    a, b = np.asarray(first, float), np.asarray(second, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        result = a + fraction * (b - a)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (
        np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b
    ) / np.sin(angle)


def _distance(a, b, quaternion):
    if quaternion:
        dot = np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))
        return 2 * np.arccos(np.clip(abs(dot), 0, 1))
    return np.linalg.norm(a - b)


def filter_track(
    times: np.ndarray,
    values: np.ndarray,
    confidence: np.ndarray,
    source: np.ndarray,
    *,
    observed: np.ndarray | None = None,
    quaternion: bool = False,
    max_gap_s: float = 0.3,
    max_speed: float = 12.0,
    min_cutoff: float = 2.0,
    beta: float = 3.0,
) -> dict:
    times, values = np.asarray(times, float), np.asarray(values, float)
    if times.ndim != 1 or len(times) != len(values) or np.any(np.diff(times) <= 0):
        raise ValueError("Filtering requires a strictly increasing timeline")
    shape = values.shape[:-1]
    if np.shape(confidence) != shape or np.shape(source) != shape:
        raise ValueError("Confidence/source must match values without coordinate axis")
    if quaternion and values.shape[-1] != 4:
        raise ValueError("Quaternion tracks must be wxyz")
    count = len(times)
    tracks = int(np.prod(values.shape[1:-1]))
    data = values.copy().reshape(count, tracks, values.shape[-1])
    conf = np.asarray(confidence, float).copy().reshape(count, tracks)
    provenance = np.asarray(source, np.uint8).copy().reshape(count, tracks)
    obs = (
        np.isfinite(values).all(axis=-1) & (confidence > 0)
        if observed is None else np.asarray(observed, bool).copy()
    ).reshape(count, tracks)
    interpolated = np.zeros((count, tracks), bool)
    rejected = np.zeros((count, tracks), bool)

    for track in range(tracks):
        valid = np.isfinite(data[:, track]).all(axis=1) & (conf[:, track] > 0)
        if quaternion:
            valid &= np.linalg.norm(data[:, track], axis=1) > 1e-8
            data[valid, track] /= np.linalg.norm(data[valid, track], axis=1)[:, None]
        indices = np.flatnonzero(valid)
        for left, middle, right in zip(indices, indices[1:], indices[2:]):
            if times[right] - times[left] > max_gap_s:
                continue
            speed1 = _distance(data[left, track], data[middle, track], quaternion) / (
                times[middle] - times[left]
            )
            speed2 = _distance(data[middle, track], data[right, track], quaternion) / (
                times[right] - times[middle]
            )
            bridge = _distance(data[left, track], data[right, track], quaternion) / (
                times[right] - times[left]
            )
            if min(speed1, speed2) > max_speed and bridge < max_speed * 0.5:
                valid[middle] = False
                rejected[middle, track] = True
        data[~valid, track] = np.nan
        conf[~valid, track] = 0
        provenance[~valid, track] = 0
        obs[~valid, track] = False
        indices = np.flatnonzero(valid)
        for left, right in zip(indices, indices[1:]):
            if right == left + 1 or times[right] - times[left] > max_gap_s + 1e-9:
                continue
            for middle in range(left + 1, right):
                fraction = (times[middle] - times[left]) / (times[right] - times[left])
                a, b = data[left, track], data[right, track]
                data[middle, track] = (
                    slerp(a, b, fraction) if quaternion else (1 - fraction) * a + fraction * b
                )
                conf[middle, track] = min(conf[left, track], conf[right, track]) * 0.5
                provenance[middle, track] = provenance[left, track] | provenance[right, track]
                interpolated[middle, track] = True
                valid[middle] = True
        previous = None
        previous_raw = None
        for i in range(count):
            if not valid[i]:
                previous = previous_raw = None
                continue
            raw = data[i, track].copy()
            if previous is not None and times[i] - times[previous] <= max_gap_s:
                dt = times[i] - times[previous]
                speed = _distance(raw, previous_raw, quaternion) / dt
                cutoff = min_cutoff + beta * speed
                alpha = 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
                data[i, track] = (
                    slerp(data[previous, track], raw, alpha)
                    if quaternion else
                    (1 - alpha) * data[previous, track] + alpha * raw
                )
            previous, previous_raw = i, raw

    return {
        "value": data.reshape(values.shape),
        "confidence": conf.reshape(shape),
        "observed_mask": obs.reshape(shape),
        "interpolated_mask": interpolated.reshape(shape),
        "source": provenance.reshape(shape),
        "rejected_mask": rejected.reshape(shape),
    }
