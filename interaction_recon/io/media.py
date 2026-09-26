from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import statistics
import subprocess

import imageio_ffmpeg
import numpy as np

from interaction_recon.contracts import MediaMetadata, MediaSequence as BaseMediaSequence

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PTS_RE = re.compile(rf"\bpts_time:\s*({NUMBER})")
FRAME_DURATION_RE = re.compile(rf"\bduration_time:\s*({NUMBER})")
SIZE_RE = re.compile(r"\bs:\s*(\d+)x(\d+)\b")
FRAME_RE = re.compile(r"\bn:\s*(\d+)\b")
FPS_RE = re.compile(r"(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?\s+fps\b")
CONTAINER_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class MetadataError(RuntimeError):
    pass


@dataclass
class MediaSequence(BaseMediaSequence):
    """Decoded RGB observations.

    timestamps_s contains the actual source PTS of each selected frame.
    sample_times_s is the uniform analysis grid (source-clock seconds).
    source_timestamps_s retains every original decoded frame PTS.
    Repeated frames are intentional when upsampling. Relative landmark depth
    is not metric depth. Audio starts on its own original source clock.
    """

    duration_s: float = 0.0
    source_timestamps_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    sample_times_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    source_frame_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


@dataclass(frozen=True)
class _FrameTiming:
    pts: float
    duration: float | None
    width: int
    height: int


def ffmpeg_executable() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (RuntimeError, OSError) as exc:
        raise MetadataError(
            "FFmpeg is unavailable. Install imageio-ffmpeg with its bundled "
            "binary; system FFmpeg/FFprobe is not required."
        ) from exc


def _timings(log: str) -> list[_FrameTiming]:
    frames = []
    for line in log.splitlines():
        if "showinfo" not in line or "ashowinfo" in line:
            continue
        if FRAME_RE.search(line) is None:
            continue
        pts, size = PTS_RE.search(line), SIZE_RE.search(line)
        if pts is None or size is None:
            raise MetadataError("Decoded frame lacks presentation time or size")
        duration = FRAME_DURATION_RE.search(line)
        frames.append(_FrameTiming(
            float(pts.group(1)),
            float(duration.group(1)) if duration else None,
            int(size.group(1)),
            int(size.group(2)),
        ))
    return frames


def _parse_metadata(log: str) -> MediaMetadata:
    frames = _timings(log)
    if not frames:
        raise MetadataError("No decodable video frames were found")
    if not all(math.isfinite(frame.pts) for frame in frames):
        raise MetadataError("Video contains non-finite presentation times")
    first = frames[0]
    if first.width <= 0 or first.height <= 0:
        raise MetadataError("Video dimensions are invalid")
    if any((f.width, f.height) != (first.width, first.height) for f in frames):
        raise MetadataError("Changing frame dimensions are not supported yet")
    deltas = [b.pts - a.pts for a, b in zip(frames, frames[1:])]
    if any(delta < 0 for delta in deltas):
        raise MetadataError("Decoded presentation times are not monotonic")

    input_log = log.split("Output #", 1)[0]
    video_line = next(
        (s for s in input_log.splitlines() if "Stream #" in s and "Video:" in s), ""
    )
    audio_line = next(
        (s for s in input_log.splitlines() if "Stream #" in s and "Audio:" in s), ""
    )
    fps_match = FPS_RE.search(video_line)
    fps = None
    if fps_match:
        numerator = float(fps_match.group(1))
        denominator = float(fps_match.group(2) or 1)
        if numerator > 0 and denominator > 0:
            fps = numerator / denominator

    tail = frames[-1].duration
    duration_method = "decoded_last_frame_duration"
    if tail is None or not math.isfinite(tail) or tail <= 0:
        positive = [d for d in deltas if d > 0]
        if positive:
            tail = positive[-1]
            duration_method = "estimated_from_last_positive_interval"
        elif fps is not None:
            tail = 1 / fps
            duration_method = "estimated_from_nominal_fps"
        else:
            raise MetadataError("Cannot determine the final frame duration")
    duration = frames[-1].pts + tail - first.pts
    if not math.isfinite(duration) or duration <= 0:
        raise MetadataError("Video duration is invalid")

    match = CONTAINER_DURATION_RE.search(input_log)
    container_duration = None
    if match:
        hours, minutes, seconds = map(float, match.groups())
        container_duration = hours * 3600 + minutes * 60 + seconds
    video_codec = re.search(r"\bVideo:\s*([^,\s]+)", video_line)
    audio_codec = re.search(r"\bAudio:\s*([^,\s]+)", audio_line)
    sample_rate = re.search(r"\b(\d+)\s+Hz\b", audio_line)
    layout = re.search(r"\b\d+\s+Hz,\s*([^,\r\n]+)", audio_line)
    vfr = None
    if len(deltas) >= 2:
        median = statistics.median(deltas)
        tolerance = max(1e-5, abs(median) * 0.001)
        vfr = any(abs(d - median) > tolerance for d in deltas)
    return MediaMetadata(
        width=first.width,
        height=first.height,
        frame_count=len(frames),
        fps=fps,
        average_fps=len(frames) / duration,
        duration_s=duration,
        container_duration_s=container_duration,
        start_time_s=first.pts,
        variable_frame_rate=vfr,
        video_codec=video_codec.group(1) if video_codec else None,
        has_audio=bool(audio_line),
        audio_codec=audio_codec.group(1) if audio_codec else None,
        audio_sample_rate=int(sample_rate.group(1)) if sample_rate else None,
        audio_channel_layout=layout.group(1).strip() if layout else None,
        timing_source="ffmpeg_decoded_presentation_timestamps",
        duration_method=duration_method,
    )


def _input_command(source: Path, executable: str | None) -> list[str]:
    command = [
        executable or ffmpeg_executable(),
        "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info",
        "-xerror", "-copyts",
    ]
    if source.suffix.casefold() == ".gif":
        # GIF demuxer synthesizes PTS from the graphic-control frame delays.
        command += ["-ignore_loop", "1", "-min_delay", "0"]
    return command + ["-i", str(source)]


def _run(command: list[str], timeout_s: float, capture: bool) -> tuple[bytes, str]:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MetadataError(f"Metadata decoding exceeded {timeout_s:g} seconds") from exc
    except OSError as exc:
        raise MetadataError(f"Could not launch bundled FFmpeg: {exc}") from exc
    log = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode:
        diagnostic = "\n".join(
            s.strip() for s in log.splitlines()
            if s.strip() and "showinfo" not in s
        )[-4000:]
        raise MetadataError(
            f"FFmpeg exited with code {completed.returncode}: {diagnostic}"
        )
    return completed.stdout or b"", log


def probe_media(
    path: str | Path, *, timeout_s: float = 600, executable: str | None = None
) -> MediaMetadata:
    command = _input_command(Path(path).resolve(), executable) + [
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough",
        "-f", "null", "-",
    ]
    _, log = _run(command, timeout_s, False)
    return _parse_metadata(log)


def load_media(
    path: str | Path,
    analysis_fps: float | None = 15.0,
    *,
    timeout_s: float = 600,
    executable: str | None = None,
) -> MediaSequence:
    if analysis_fps is not None and (
        not np.isfinite(analysis_fps) or analysis_fps <= 0
    ):
        raise ValueError("analysis_fps must be positive and finite")
    source = Path(path).resolve()
    command = _input_command(source, executable) + [
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo", "pipe:1",
    ]
    raw, log = _run(command, timeout_s, True)
    metadata = _parse_metadata(log)
    source_times = np.array([f.pts for f in _timings(log)], dtype=np.float64)
    expected = metadata.frame_count * metadata.height * metadata.width * 3
    if len(raw) != expected:
        raise MetadataError(
            f"Raw RGB byte count mismatch: expected {expected}, received {len(raw)}"
        )
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(
        metadata.frame_count, metadata.height, metadata.width, 3
    )
    if analysis_fps is None:
        grid = source_times.copy()
        indices = np.arange(len(source_times), dtype=np.int64)
    else:
        count = max(1, int(np.ceil(metadata.duration_s * analysis_fps - 1e-8)))
        grid = source_times[0] + np.arange(count, dtype=np.float64) / analysis_fps
        indices = np.searchsorted(source_times, grid + 1e-8, side="right") - 1
        indices = np.clip(indices, 0, len(source_times) - 1).astype(np.int64)
    frames = decoded[indices].copy()
    del decoded, raw

    audio = None
    audio_start = None
    if metadata.has_audio:
        audio_command = _input_command(source, executable) + [
            "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-af",
            "aresample=16000:async=1,aformat=sample_fmts=s16:channel_layouts=mono,ashowinfo",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "s16le", "pipe:1",
        ]
        pcm, audio_log = _run(audio_command, timeout_s, True)
        first_pts = next(
            (
                PTS_RE.search(line)
                for line in audio_log.splitlines()
                if "ashowinfo" in line and PTS_RE.search(line)
            ),
            None,
        )
        if not pcm or first_pts is None or len(pcm) % 2:
            raise MetadataError("Audio stream exists but could not be decoded with timing")
        audio = (np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768).copy()
        audio_start = float(first_pts.group(1))

    sequence = MediaSequence(
        rgb_frames=frames,
        timestamps_s=source_times[indices],
        fps=analysis_fps or metadata.fps or metadata.average_fps,
        width=metadata.width,
        height=metadata.height,
        audio_waveform=audio,
        audio_sample_rate=16000 if audio is not None else None,
        audio_start_time_s=audio_start,
        duration_s=metadata.duration_s,
        source_timestamps_s=source_times,
        sample_times_s=grid,
        source_frame_indices=indices,
    )
    sequence.validate()
    return sequence
