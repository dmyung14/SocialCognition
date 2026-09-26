import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from interaction_recon.cli import main
from interaction_recon.config import RunOptions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("as_zip", [False, True])
def test_module_cli_discovers_six_and_writes_pair(tmp_path, benchmark_folder, as_zip):
    source = benchmark_folder
    if as_zip:
        source = tmp_path / "test_1.zip"
        with zipfile.ZipFile(source, "w") as handle:
            for path in benchmark_folder.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(benchmark_folder).as_posix())
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable, "-m", "interaction_recon", "run",
            "--input", str(source), "--output", str(output), "--inventory-only",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = read_manifest(output)
    assert manifest["schema_version"] == "1.0"
    assert manifest["milestone"] == 2
    assert manifest["reconstruction_performed"] is False
    assert manifest["counts"] == {
        "media": 6, "metadata_ok": 6, "metadata_failed": 0,
        "paired_segments": 1, "unpaired_segments": 4,
        "ambiguous_segments": 0, "unclassified_files": 0,
    }
    assert all(item["metadata"]["has_audio"] for item in manifest["media"])
    pair = read_manifest(output / "003")
    assert pair["stage_status"] == "metadata_complete"
    assert pair["artifacts"]["scene.xml"]["status"] == "not_implemented"
    assert not (output / "003" / "scene.xml").exists()
    assert not (output / "002").exists()
    assert (output / "003" / "logs" / "pipeline.log").is_file()
    for item in manifest["media"]:
        assert item["source"]["path"] == str(source.resolve())
        assert "\\" not in item["source"]["member"]
        assert "local_path" not in item
    assert "interaction-recon-" not in (output / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_explicit_gifs_and_requested_options(tmp_path, tiny_gif):
    from dataclasses import asdict

    builder = tmp_path / "builder.gif"
    builder.write_bytes(tiny_gif.read_bytes())
    output = tmp_path / "explicit"
    result = main([
        "reconstruct", "--guider", str(tiny_gif), "--builder", str(builder),
        "--output", str(output), "--render-format", "both", "--device", "cpu",
        "--debug", "--inventory-only",
    ])
    assert result == 0
    manifest = read_manifest(output)
    assert manifest["segment_id"] == "explicit_pair"
    assert manifest["pairing_method"] == "explicit"
    assert manifest["requested_options"] == asdict(RunOptions(
        render_format="both", device="cpu", debug=True, inventory_only=True
    ))
    assert len(manifest["media"]) == 2
    assert {entry["role"] for entry in manifest["media"]} == {"guider", "builder"}
    assert all(entry["metadata"]["frame_count"] == 3 for entry in manifest["media"])
    assert not (output / "explicit_pair").exists()
    assert not (output / "simulation.mp4").exists()


def test_corrupt_source_is_reported_without_losing_good_files(tmp_path, tiny_gif):
    source = tmp_path / "input"
    source.mkdir()
    (source / "segment_5_instruction_candidate.gif").write_bytes(tiny_gif.read_bytes())
    (source / "segment_5_assembler_ego_with_audio.mp4").write_bytes(b"broken")
    output = tmp_path / "output"
    assert main(["run", "--input", str(source), "--output", str(output)]) == 1
    manifest = read_manifest(output)
    assert manifest["counts"]["metadata_ok"] == 1
    assert manifest["counts"]["metadata_failed"] == 1
    assert manifest["counts"]["paired_segments"] == 1
    assert manifest["stage_status"] == "metadata_failed"
    assert read_manifest(output / "005")["stage_status"] == "metadata_failed"
    failed = next(
        item for item in manifest["media"] if item["metadata_status"] == "error"
    )
    assert failed["metadata"] is None
    assert failed["error"]


def test_unpaired_gif_is_successful_inventory(tmp_path, tiny_gif):
    source = tmp_path / "input"
    source.mkdir()
    (source / "segment_77_instruction_candidate.gif").write_bytes(tiny_gif.read_bytes())
    output = tmp_path / "output"
    assert main(["run", "--input", str(source), "--output", str(output)]) == 0
    manifest = read_manifest(output)
    assert manifest["segments"][0]["status"] == "unpaired"
    assert manifest["segments"][0]["output_directory"] is None
    assert not (output / "077").exists()


def test_empty_input_is_rejected(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    output = tmp_path / "output"
    assert main(["run", "--input", str(source), "--output", str(output)]) == 2
    assert not output.exists()


def test_missing_input_is_rejected(tmp_path):
    assert main([
        "run", "--input", str(tmp_path / "missing"), "--output", str(tmp_path / "output")
    ]) == 2


def test_same_explicit_file_is_rejected(tmp_path, tiny_gif):
    assert main([
        "reconstruct", "--guider", str(tiny_gif), "--builder", str(tiny_gif),
        "--output", str(tmp_path / "output"),
    ]) == 2


def test_existing_output_is_allowed_and_unrelated_files_are_preserved(tmp_path, tiny_gif):
    source = tmp_path / "input"
    source.mkdir()
    (source / "source.gif").write_bytes(tiny_gif.read_bytes())
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    (output / "manifest.json").write_text("old manifest", encoding="utf-8")
    assert main(["run", "--input", str(source), "--output", str(output)]) == 0
    assert marker.read_text(encoding="utf-8") == "existing"
    assert read_manifest(output)["counts"]["media"] == 1


def test_argument_validation():
    with pytest.raises(SystemExit) as error:
        main(["run", "--input", "test_1"])
    assert error.value.code == 2
