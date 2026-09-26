from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SourceRef:
    """Persistent source locator. Never points into temporary ZIP extraction."""

    kind: str
    path: str
    member: str | None = None


@dataclass(frozen=True)
class MediaMetadata:
    width: int
    height: int
    frame_count: int
    fps: float | None
    average_fps: float
    duration_s: float
    container_duration_s: float | None
    start_time_s: float
    variable_frame_rate: bool | None
    video_codec: str | None
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channel_layout: str | None
    timing_source: str
    duration_method: str


@dataclass
class MediaSequence:
    """Future media-loader output contract, not a decoder implementation.

    rgb_frames:
        uint8 RGB, shape (N, height, width, 3).
    timestamps_s:
        float64 presentation times on the source timeline, shape (N,).
        Do not silently rebase to zero or replace VFR times with frame/fps.
    fps:
        Target analysis sampling rate, or nominal rate when not resampled.
    audio_waveform:
        Optional float32 waveform, shape (samples,) or (samples, channels).
        Audio is separate from RGB observations and simulation state.
    audio_start_time_s:
        First audio sample's source presentation time; required with audio.
    """

    rgb_frames: NDArray[np.uint8]
    timestamps_s: NDArray[np.float64]
    fps: float | None
    width: int
    height: int
    audio_waveform: NDArray[np.float32] | None = None
    audio_sample_rate: int | None = None
    audio_start_time_s: float | None = None

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MediaSequence dimensions must be positive")
        frames = self.rgb_frames
        if frames.dtype != np.uint8:
            raise ValueError("rgb_frames must be uint8")
        if frames.ndim != 4 or frames.shape[1:] != (
            self.height, self.width, 3
        ):
            raise ValueError("rgb_frames must have shape (N, height, width, 3)")
        times = self.timestamps_s
        if times.dtype != np.float64 or times.shape != (len(frames),):
            raise ValueError("timestamps_s must be float64 with shape (N,)")
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("timestamps_s must be finite and nondecreasing")
        if self.fps is not None and (
            not np.isfinite(self.fps) or self.fps <= 0
        ):
            raise ValueError("fps must be positive and finite, or None")
        if self.audio_waveform is None:
            if (
                self.audio_sample_rate is not None
                or self.audio_start_time_s is not None
            ):
                raise ValueError("absent audio must not have timing metadata")
            return
        audio = self.audio_waveform
        if audio.dtype != np.float32 or audio.ndim not in (1, 2):
            raise ValueError("audio must be float32 mono or samples-by-channels")
        if not np.isfinite(audio).all():
            raise ValueError("audio must be finite")
        if self.audio_sample_rate is None or self.audio_sample_rate <= 0:
            raise ValueError("audio requires a positive sample rate")
        if (
            self.audio_start_time_s is None
            or not np.isfinite(self.audio_start_time_s)
        ):
            raise ValueError("audio requires a finite source start time")
