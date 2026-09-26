import os
from pathlib import Path

import pytest

from interaction_recon.config import RunOptions
from interaction_recon.pipeline import run_inventory


@pytest.mark.skipif(
    not os.environ.get("RECON_TEST_INPUT"),
    reason="Set RECON_TEST_INPUT to test_1 or test_1.zip to test real media",
)
def test_real_six_file_benchmark(tmp_path):
    source = Path(os.environ["RECON_TEST_INPUT"])
    manifest, code = run_inventory(source, tmp_path / "real", RunOptions())
    assert code == 0
    assert manifest["counts"]["media"] == 6
    assert manifest["counts"]["paired_segments"] == 1
    assert manifest["counts"]["unpaired_segments"] == 4
    paired = [
        segment for segment in manifest["segments"]
        if segment["status"] == "paired"
    ]
    assert [segment["segment_id"] for segment in paired] == ["003"]

    expected = {
        "segment_002_instruction_candidate.mp4": (704, 704, 20, 16.30),
        "segment_003_instruction_candidate.mp4": (704, 704, 20, 12.45),
        "segment_003_assembler_ego_with_audio.mp4": (768, 768, 15, 5.40),
        "segment_012_assembler_ego_with_audio.mp4": (768, 768, 15, 14.00),
        "segment_021_assembler_ego_with_audio.mp4": (768, 768, 15, 8.67),
        "segment_027_assembler_ego_with_audio.mp4": (768, 768, 15, 5.93),
    }
    for item in manifest["media"]:
        width, height, fps, duration = expected[Path(item["id"]).name]
        metadata = item["metadata"]
        assert metadata["width"] == width
        assert metadata["height"] == height
        assert metadata["fps"] == pytest.approx(fps, abs=0.05)
        assert metadata["duration_s"] == pytest.approx(duration, abs=0.12)
        assert metadata["has_audio"] is True
