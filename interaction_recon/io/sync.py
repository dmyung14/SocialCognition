from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import resample_poly

from interaction_recon.io.cache import write_json
from interaction_recon.io.media import MediaSequence


@dataclass(frozen=True)
class LagEstimate:
    lag_s: float
    confidence: float
    coarse_lag_s: float
    peak_to_sidelobe: float
    correlation: float
    overlap_samples: int


def _mono(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Audio must be finite mono or samples-by-channels")
    if len(values):
        values = values - values.mean()
    scale = np.std(values) if len(values) else 0.0
    return values / scale if scale > 1e-10 else np.zeros_like(values)


def _resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = gcd(source_rate, target_rate)
    return resample_poly(values, target_rate // divisor, source_rate // divisor)


def _overlap(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    # b index = a index + lag.
    begin = max(0, -lag)
    end = min(len(a), len(b) - lag)
    return a[begin:end], b[begin + lag:end + lag]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-10 else 0.0


def _psr(values: np.ndarray, best: int, guard: int) -> float:
    side = values[np.abs(np.arange(len(values)) - best) > guard]
    if len(side) < 3:
        return 0.0
    return max(0.0, float(
        (values[best] - np.mean(side)) / max(np.std(side), 1e-10)
    ))


def _gcc(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = next_fast_len(len(a) + len(b) - 1)
    cross = rfft(b, size) * np.conj(rfft(a, size))
    correlation = irfft(cross / np.maximum(np.abs(cross), 1e-12), size)
    negative = correlation[-(len(a) - 1):] if len(a) > 1 else np.empty(0)
    return (
        np.arange(-(len(a) - 1), len(b)),
        np.concatenate((negative, correlation[:len(b)])),
    )


def _valid_lags(lags: np.ndarray, na: int, nb: int) -> np.ndarray:
    overlap = np.minimum(na, nb - lags) - np.maximum(0, -lags)
    return overlap >= max(3, int(min(na, nb) * 0.35))


def gcc_phat(
    guider_audio: np.ndarray,
    builder_audio: np.ndarray,
    sample_rate: int = 16000,
) -> LagEstimate:
    """Positive lag means the matching event occurs later in builder audio."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    a, b = _mono(guider_audio), _mono(builder_audio)
    empty = LagEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    if min(len(a), len(b)) < max(16, sample_rate // 10):
        return empty
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return empty

    coarse_rate = min(1000, sample_rate)
    ca = _resample(a, sample_rate, coarse_rate)
    cb = _resample(b, sample_rate, coarse_rate)
    lags, values = _gcc(ca, cb)
    valid = _valid_lags(lags, len(ca), len(cb))
    coarse_index = np.argmax(np.where(valid, values, -np.inf))
    coarse = float(lags[coarse_index] / coarse_rate)

    lags, values = _gcc(a, b)
    valid = _valid_lags(lags, len(a), len(b))
    local = valid & (np.abs(lags / sample_rate - coarse) <= 0.05)
    if not np.any(local):
        return empty
    best = int(np.argmax(np.where(local, values, -np.inf)))
    lag = int(lags[best])
    aa, bb = _overlap(a, b, lag)
    correlation = max(0.0, _ncc(aa, bb))

    valid_values = values[valid]
    best_valid = int(np.searchsorted(lags[valid], lag))
    psr = _psr(valid_values, best_valid, max(1, int(sample_rate * 0.01)))
    # A large PHAT peak alone is not evidence of shared audio. Gate it with
    # actual overlapping waveform coherence to reject unrelated recordings.
    confidence = float(
        np.clip((psr - 4) / 12, 0, 1)
        * np.clip((correlation - 0.1) / 0.7, 0, 1)
    )
    return LagEstimate(
        lag / sample_rate, confidence, coarse, psr, correlation, len(aa)
    )


def motion_energy(sequence: MediaSequence) -> np.ndarray:
    """Lower-scene motion baseline; tracking supplies a hand/table ROI version."""
    previous = None
    energy = []
    for rgb in sequence.rgb_frames:
        gray = cv2.cvtColor(cv2.resize(rgb, (96, 96)), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32)
        roi = gray[32:, 8:88]
        energy.append(0.0 if previous is None else float(np.mean(np.abs(roi - previous))))
        previous = roi
    return np.asarray(energy, dtype=np.float64)


def visual_ncc(a: np.ndarray, b: np.ndarray, fps: float) -> LagEstimate:
    a, b = _mono(a), _mono(b)
    if min(len(a), len(b)) < 5 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return LagEstimate(0, 0, 0, 0, 0, 0)
    lags = np.arange(-(len(a) - 1), len(b))
    lags = lags[_valid_lags(lags, len(a), len(b))]
    values = np.array([_ncc(*_overlap(a, b, int(lag))) for lag in lags])
    best = int(np.argmax(values))
    lag = int(lags[best])
    psr = _psr(values, best, max(1, round(0.2 * fps)))
    peak = max(0.0, float(values[best]))
    # Motion alignment is less specific than shared audio.
    confidence = float(
        0.8 * np.clip((psr - 2) / 6, 0, 1) * np.clip((peak - 0.3) / 0.7, 0, 1)
    )
    overlap = len(_overlap(a, b, lag)[0])
    return LagEstimate(lag / fps, confidence, lag / fps, psr, peak, overlap)


def synchronize(
    guider: MediaSequence,
    builder: MediaSequence,
    *,
    threshold: float = 0.5,
    guider_motion: np.ndarray | None = None,
    builder_motion: np.ndarray | None = None,
    output: Path | None = None,
) -> dict:
    """Convention: corresponding source timestamps satisfy tb = tg + offset_s."""
    audio = None
    candidates = {}
    if guider.audio_waveform is not None and builder.audio_waveform is not None:
        rate = 16000
        ga = _resample(guider.audio_waveform, guider.audio_sample_rate, rate)
        ba = _resample(builder.audio_waveform, builder.audio_sample_rate, rate)
        audio = gcc_phat(ga, ba, rate)
        candidates["audio_gcc_phat"] = asdict(audio)
        candidates["audio_gcc_phat"]["offset_s"] = (
            builder.audio_start_time_s - guider.audio_start_time_s + audio.lag_s
        )

    if audio is not None and audio.confidence >= threshold:
        method = "audio_gcc_phat"
    else:
        if not np.isclose(guider.fps, builder.fps):
            raise ValueError("Visual synchronization requires a common analysis FPS")
        visual = visual_ncc(
            motion_energy(guider) if guider_motion is None else guider_motion,
            motion_energy(builder) if builder_motion is None else builder_motion,
            guider.fps,
        )
        candidates["visual_motion"] = asdict(visual)
        candidates["visual_motion"]["offset_s"] = float(
            builder.sample_times_s[0] - guider.sample_times_s[0] + visual.lag_s
        )
        method = "visual_motion"

    selected = candidates[method]
    offset = float(selected["offset_s"])
    gs = float(guider.source_timestamps_s[0])
    bs = float(builder.source_timestamps_s[0])
    overlap = max(
        0.0,
        min(gs + guider.duration_s, bs + builder.duration_s - offset)
        - max(gs, bs - offset),
    )
    result = {
        "offset_s": offset,
        "method": method,
        "confidence": float(selected["confidence"]),
        "offset_convention": "builder_source_time = guider_source_time + offset_s",
        "confidence_threshold": threshold,
        "reliable": bool(selected["confidence"] >= threshold and overlap > 0),
        "overlap_s": overlap,
        "candidates": candidates,
        "cross_view_consistency": {
            "status": "not_estimated",
            "confidence": None,
            "reason": "2D temporal alignment does not establish a shared room or interaction",
        },
        "warnings": [
            "Offset is a hypothesis, not proof that the sources show the same interaction."
        ],
    }
    if not result["reliable"]:
        result["warnings"].append("Low-confidence synchronization; inspect both source views.")
    if output is not None:
        write_json(output, result)
    return result
