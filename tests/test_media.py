import subprocess

import numpy as np
import pytest

from interaction_recon.contracts import MediaSequence
from interaction_recon.io.media import (
    MetadataError,
    _parse_metadata,
    probe_media,
)


def test_mp4_metadata_and_audio(tiny_mp4):
    metadata = probe_media(tiny_mp4)
    assert (metadata.width, metadata.height) == (64, 48)
    assert metadata.frame_count == 6
    assert metadata.fps == pytest.approx(10)
    assert metadata.average_fps == pytest.approx(10, rel=0.02)
    assert metadata.duration_s == pytest.approx(0.6, abs=0.02)
    assert metadata.has_audio is True
    assert metadata.audio_sample_rate == 16000
    assert metadata.audio_codec == "aac"
    assert metadata.variable_frame_rate is False


def test_gif_metadata_uses_frame_delays(tiny_gif):
    metadata = probe_media(tiny_gif)
    assert (metadata.width, metadata.height) == (1, 1)
    assert metadata.frame_count == 3
    assert metadata.duration_s == pytest.approx(0.6, abs=0.015)
    assert metadata.variable_frame_rate is True
    assert metadata.has_audio is False
    assert metadata.audio_sample_rate is None
    assert metadata.duration_method == "decoded_last_frame_duration"


def test_corrupt_media_raises_useful_error(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not an mp4")
    with pytest.raises(MetadataError, match="FFmpeg exited"):
        probe_media(broken)


def test_vfr_and_nonzero_source_start_are_not_rebased():
    log = """
Input #0, mov, from 'source.mp4':
  Duration: 00:00:10.00, start: 2.000000, bitrate: 500 kb/s
  Stream #0:0: Video: h264, yuv420p, 64x48, 25 fps, 25 tbr
[Parsed_showinfo_0] n: 0 pts: 2000 pts_time:2 duration:40 duration_time:0.04 fmt:yuv420p s:64x48
[Parsed_showinfo_0] n: 1 pts: 2040 pts_time:2.04 duration:80 duration_time:0.08 fmt:yuv420p s:64x48
[Parsed_showinfo_0] n: 2 pts: 2120 pts_time:2.12 duration:40 duration_time:0.04 fmt:yuv420p s:64x48
"""
    metadata = _parse_metadata(log)
    assert metadata.start_time_s == 2
    assert metadata.duration_s == pytest.approx(0.16)
    assert metadata.container_duration_s == 10
    assert metadata.frame_count == 3
    assert metadata.variable_frame_rate is True
    assert metadata.fps == 25
    assert metadata.average_fps == pytest.approx(18.75)


def test_missing_frame_duration_is_marked_estimated():
    log = """
  Stream #0:0: Video: h264, yuv420p, 64x48, 10 fps
[Parsed_showinfo_0] n: 0 pts: 0 pts_time:0 fmt:yuv420p s:64x48
[Parsed_showinfo_0] n: 1 pts: 1 pts_time:0.1 fmt:yuv420p s:64x48
"""
    metadata = _parse_metadata(log)
    assert metadata.duration_s == pytest.approx(0.2)
    assert metadata.duration_method == "estimated_from_last_positive_interval"
    assert metadata.variable_frame_rate is None


def test_probe_timeout_is_wrapped(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(MetadataError, match="exceeded"):
        probe_media(tmp_path / "source.mp4", executable="unused", timeout_s=1)


def test_media_sequence_contract():
    sequence = MediaSequence(
        rgb_frames=np.zeros((2, 3, 4, 3), dtype=np.uint8),
        timestamps_s=np.array([2.0, 2.2], dtype=np.float64),
        fps=5.0,
        width=4,
        height=3,
    )
    sequence.validate()
    sequence.timestamps_s = np.array([2.2, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="nondecreasing"):
        sequence.validate()


def test_audio_contract_requires_source_timing():
    sequence = MediaSequence(
        rgb_frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
        timestamps_s=np.array([0.0], dtype=np.float64),
        fps=10.0,
        width=1,
        height=1,
        audio_waveform=np.zeros(100, dtype=np.float32),
        audio_sample_rate=16000,
    )
    with pytest.raises(ValueError, match="source start time"):
        sequence.validate()
    sequence.audio_start_time_s = -0.01
    sequence.validate()
