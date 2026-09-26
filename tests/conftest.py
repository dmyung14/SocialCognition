from pathlib import Path
import subprocess

import imageio_ffmpeg
import pytest

BENCHMARK_NAMES = (
    "segment_002_instruction_candidate.mp4",
    "segment_003_instruction_candidate.mp4",
    "segment_003_assembler_ego_with_audio.mp4",
    "segment_012_assembler_ego_with_audio.mp4",
    "segment_021_assembler_ego_with_audio.mp4",
    "segment_027_assembler_ego_with_audio.mp4",
)


def write_gif(path: Path, delays_cs: tuple[int, ...] = (10, 30, 20)) -> Path:
    """A valid 1x1 GIF with explicitly controlled frame delays."""
    data = bytearray(
        b"GIF89a"
        b"\x01\x00\x01\x00"
        b"\x80\x00\x00"
        b"\x00\x00\x00\xff\xff\xff"
    )
    for delay in delays_cs:
        data.extend(b"\x21\xf9\x04\x00")
        data.extend(delay.to_bytes(2, "little"))
        data.extend(b"\x00\x00")
        data.extend(
            b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"
            b"\x02\x02\x44\x01\x00"
        )
    data.extend(b"\x3b")
    path.write_bytes(data)
    return path


@pytest.fixture(scope="session")
def tiny_mp4(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=64x48:r=10:d=0.6",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.6",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "mpeg4", "-q:v", "4", "-threads", "1",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest", "-y", str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return path


@pytest.fixture
def benchmark_folder(tmp_path, tiny_mp4) -> Path:
    root = tmp_path / "test_1"
    nested = root / "recordings"
    nested.mkdir(parents=True)
    for name in BENCHMARK_NAMES:
        (nested / name).write_bytes(tiny_mp4.read_bytes())
    return root


@pytest.fixture
def tiny_gif(tmp_path) -> Path:
    return write_gif(tmp_path / "variable.gif")
