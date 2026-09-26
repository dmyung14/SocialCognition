"""Bundled-FFmpeg encoding, with atomic publication and bounded GIF settings."""
from pathlib import Path
import subprocess
import tempfile
import uuid
from collections.abc import Iterable

import numpy as np

from interaction_recon.io.media import ffmpeg_executable


def write_mp4(
    output: Path, frames: Iterable[np.ndarray], width: int, height: int,
    fps: float = 30,
) -> int:
    output = Path(output)
    if width % 2 or height % 2 or min(width, height) <= 0:
        raise ValueError("H.264 output dimensions must be positive and even")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.mp4")
    process = None
    count = 0
    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(temporary),
    ]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            try:
                for frame in frames:
                    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                        raise ValueError("Encoder requires correctly sized uint8 RGB frames")
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    count += 1
                process.stdin.close()
                code = process.wait(timeout=120)
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                process.kill()
                process.wait()
                errors.seek(0)
                raise RuntimeError(
                    "FFmpeg encoding failed: " + errors.read().decode(errors="replace")
                ) from exc
            if code or not count:
                errors.seek(0)
                raise RuntimeError(
                    "FFmpeg produced no valid movie: "
                    + errors.read().decode(errors="replace")
                )
        temporary.replace(output)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        temporary.unlink(missing_ok=True)
    return count


def convert_gif(
    source: Path, output: Path | None = None, *, fps: float = 15, width: int = 640,
) -> Path:
    source = Path(source)
    output = Path(output) if output is not None else source.with_suffix(".gif")
    if not 0 < fps <= 15 or not 1 <= width <= 640:
        raise ValueError("GIF requires 0 < fps <= 15 and 1 <= width <= 640")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.gif")
    filters = (
        f"[0:v]fps={fps},scale=w='min({width},iw)':h=-1:flags=lanczos,"
        "split[a][b];[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=sierra2_4a"
    )
    try:
        result = subprocess.run(
            [
                ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-filter_complex_threads", "1",
                "-filter_complex", filters, "-an", "-loop", "0",
                "-y", str(temporary),
            ],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=300, check=False,
        )
        if result.returncode or not temporary.exists() or not temporary.stat().st_size:
            raise RuntimeError(
                "GIF conversion failed: " + result.stderr.decode(errors="replace")
            )
        temporary.replace(output)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GIF conversion exceeded 300 seconds") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output
