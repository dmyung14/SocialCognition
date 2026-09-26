import numpy as np
import pytest

from interaction_recon.io.media import MediaSequence
from interaction_recon.io.sync import gcc_phat, synchronize, visual_ncc


@pytest.mark.parametrize("shift", [0.237, -0.181])
def test_gcc_phat_recovers_known_noise_offset(shift):
    rate = 16000
    rng = np.random.default_rng(123)
    noise = rng.normal(size=rate * 4).astype(np.float32)
    padding = round(abs(shift) * rate)
    a = noise
    b = np.concatenate((np.zeros(padding, np.float32), noise))
    if shift < 0:
        a, b = b, a
    result = gcc_phat(a, b, rate)
    assert result.lag_s == pytest.approx(shift, abs=2 / rate)
    assert result.coarse_lag_s == pytest.approx(shift, abs=0.003)
    assert 0.8 <= result.confidence <= 1
    assert result.peak_to_sidelobe > 10


def test_unrelated_audio_and_silence_are_not_confident():
    rng = np.random.default_rng(42)
    result = gcc_phat(rng.normal(size=64000), rng.normal(size=48000))
    assert result.confidence < 0.2
    assert gcc_phat(np.zeros(16000), np.zeros(16000)).confidence == 0


def _sequence(start=0.0, audio=None, audio_start=None):
    times = start + np.arange(60, dtype=np.float64) / 15
    return MediaSequence(
        rgb_frames=np.zeros((60, 8, 8, 3), np.uint8),
        timestamps_s=times,
        fps=15.0, width=8, height=8,
        audio_waveform=audio,
        audio_sample_rate=16000 if audio is not None else None,
        audio_start_time_s=audio_start,
        duration_s=4.0,
        source_timestamps_s=times,
        sample_times_s=times,
        source_frame_indices=np.arange(60),
    )


def test_visual_fallback_and_offset_convention(tmp_path):
    rng = np.random.default_rng(7)
    motion = rng.random(60)
    builder_motion = np.r_[np.zeros(5), motion[:-5]]
    estimate = visual_ncc(motion, builder_motion, 15)
    assert estimate.lag_s == pytest.approx(5 / 15)
    result = synchronize(
        _sequence(2), _sequence(4),
        guider_motion=motion, builder_motion=builder_motion,
        output=tmp_path / "sync.json",
    )
    assert result["method"] == "visual_motion"
    assert result["offset_s"] == pytest.approx(2 + 5 / 15)
    assert 0 <= result["confidence"] <= 1
    assert (tmp_path / "sync.json").is_file()
    assert result["cross_view_consistency"]["status"] == "not_estimated"


def test_audio_source_start_times_are_included():
    rng = np.random.default_rng(4)
    audio = rng.normal(size=64000).astype(np.float32)
    builder = np.r_[np.zeros(3200, np.float32), audio]
    result = synchronize(
        _sequence(2, audio, 2.1),
        _sequence(4, builder, 4.2),
    )
    assert result["method"] == "audio_gcc_phat"
    assert result["offset_s"] == pytest.approx(2.3, abs=0.001)


def test_low_audio_confidence_invokes_visual_fallback():
    silence = np.zeros(64000, np.float32)
    result = synchronize(
        _sequence(0, silence, 0), _sequence(0, silence, 0)
    )
    assert result["method"] == "visual_motion"
    assert result["confidence"] == 0
    assert not result["reliable"]
    assert "audio_gcc_phat" in result["candidates"]
