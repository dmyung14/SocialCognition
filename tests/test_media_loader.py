import numpy as np
import pytest

from interaction_recon.io.media import load_media


def test_generated_mp4_rgb_audio_and_resampling(tiny_mp4):
    sequence = load_media(tiny_mp4, analysis_fps=15)
    sequence.validate()
    assert sequence.rgb_frames.shape == (9, 48, 64, 3)
    assert sequence.rgb_frames.dtype == np.uint8
    assert sequence.rgb_frames[..., 2].mean() > 200
    assert sequence.rgb_frames[..., 0].mean() < 30
    assert len(sequence.source_timestamps_s) == 6
    assert sequence.duration_s == pytest.approx(0.6, abs=0.02)
    np.testing.assert_allclose(np.diff(sequence.sample_times_s), 1 / 15)
    np.testing.assert_array_equal(
        sequence.timestamps_s,
        sequence.source_timestamps_s[sequence.source_frame_indices],
    )
    assert sequence.audio_waveform.dtype == np.float32
    assert sequence.audio_waveform.ndim == 1
    assert sequence.audio_sample_rate == 16000
    assert len(sequence.audio_waveform) >= 9000
    assert np.std(sequence.audio_waveform) > 0.01
    assert np.isfinite(sequence.audio_start_time_s)


def test_gif_synthetic_timestamps_and_original_duration(tiny_gif):
    original = load_media(tiny_gif, analysis_fps=None)
    np.testing.assert_allclose(original.timestamps_s, [0, 0.1, 0.4], atol=0.001)
    assert original.duration_s == pytest.approx(0.6, abs=0.001)
    assert original.audio_waveform is None
    normalized = load_media(tiny_gif, analysis_fps=10)
    assert len(normalized.rgb_frames) == 6
    np.testing.assert_allclose(normalized.source_timestamps_s, original.timestamps_s)
    np.testing.assert_array_equal(normalized.source_frame_indices, [0, 1, 1, 1, 2, 2])
