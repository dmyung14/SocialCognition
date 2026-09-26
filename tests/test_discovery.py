from pathlib import Path
import stat
import zipfile

import pytest

from interaction_recon.io.discover import (
    discover_input,
    infer_role,
    infer_segment,
)
from interaction_recon.io.pairing import pair_media


def test_six_files_and_only_one_pair(benchmark_folder):
    with discover_input(benchmark_folder) as discovered:
        assert len(discovered.media) == 6
        segments, unknown = pair_media(discovered.media)
    assert unknown == []
    assert [item["segment_id"] for item in segments] == [
        "002", "003", "012", "021", "027"
    ]
    assert [item["segment_id"] for item in segments if item["status"] == "paired"] == [
        "003"
    ]
    pair = next(item for item in segments if item["status"] == "paired")
    assert pair["guider"].endswith("segment_003_instruction_candidate.mp4")
    assert pair["builder"].endswith("segment_003_assembler_ego_with_audio.mp4")


def test_zip_junk_nested_discovery_and_cleanup(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/SEGMENT_41_instruction_candidate.MP4", b"test")
        handle.writestr("nested/segment_041_assembler_ego_with_audio.GIF", b"test")
        handle.writestr("__MACOSX/segment_99_instruction_candidate.mp4", b"junk")
        handle.writestr("nested/._segment_41_instruction_candidate.mp4", b"junk")
        handle.writestr(".hidden/segment_41_instruction_candidate.mp4", b"junk")
        handle.writestr("nested/.DS_Store", b"junk")
        handle.writestr("notes.txt", b"not media")
    with discover_input(archive) as discovered:
        assert len(discovered.media) == 2
        paths = [item.local_path for item in discovered.media]
        assert all(path.exists() for path in paths)
        assert all(item.source.path == str(archive.resolve()) for item in discovered.media)
        assert all(item.source.kind == "zip" for item in discovered.media)
        segments, unknown = pair_media(discovered.media)
        assert unknown == []
        assert segments[0]["segment_id"] == "041"
        assert segments[0]["status"] == "paired"
    assert all(not path.exists() for path in paths)


def test_folder_hidden_files_and_unknown_names(tmp_path):
    (tmp_path / "visible.GIF").write_bytes(b"test")
    (tmp_path / ".hidden.mp4").write_bytes(b"test")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "ignored.mp4").write_bytes(b"test")
    with discover_input(tmp_path) as discovered:
        assert [item.id for item in discovered.media] == ["visible.GIF"]
        segments, unknown = pair_media(discovered.media)
        assert segments == []
        assert unknown == ["visible.GIF"]


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.mp4",
        "/absolute.mp4",
        r"..\escaped.mp4",
        r"C:\escaped.mp4",
        "nested/../../escaped.mp4",
        "nested/CON.mp4",
        "nested/name.mp4.",
        "nested/file:stream.mp4",
    ],
)
def test_reject_unsafe_zip_paths(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(name, b"bad")
    with pytest.raises(ValueError, match="ZIP path"):
        with discover_input(archive):
            pass
    assert not (tmp_path / "escaped.mp4").exists()


def test_reject_zip_symlink(tmp_path):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link.mp4")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "../target")
    with pytest.raises(ValueError, match="symlinks"):
        with discover_input(archive):
            pass


def test_reject_case_colliding_zip_members(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/a.mp4", b"a")
        handle.writestr("NESTED/A.MP4", b"b")
    with pytest.raises(ValueError, match="Duplicate"):
        with discover_input(archive):
            pass


def test_ambiguous_pair_is_not_selected(tmp_path):
    names = [
        "segment_8_instruction_candidate.mp4",
        "segment_008_instruction_candidate.gif",
        "segment_08_assembler_ego_with_audio.mp4",
        "segment_008_other.gif",
        "unrelated.gif",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"test")
    with discover_input(tmp_path) as discovered:
        segments, unknown = pair_media(discovered.media)
    assert len(segments) == 1
    assert segments[0]["status"] == "ambiguous"
    assert segments[0]["guider"] is None
    assert len(segments[0]["candidates"]["guider"]) == 2
    assert set(unknown) == {"segment_008_other.gif", "unrelated.gif"}


def test_temporary_extraction_is_removed_after_exception(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.gif", b"test")
    extracted = None
    with pytest.raises(RuntimeError, match="abort"):
        with discover_input(archive) as discovered:
            extracted = discovered.media[0].local_path
            raise RuntimeError("abort")
    assert isinstance(extracted, Path)
    assert not extracted.exists()


def test_inference():
    assert infer_segment("SEGMENT_0007_instruction_candidate.MP4") == "007"
    assert infer_segment("segment_1200_clip.gif") == "1200"
    assert infer_segment("clip.gif") is None
    assert infer_role("INSTRUCTION_CANDIDATE.mp4") == "guider"
    assert infer_role("assembler_ego_with_audio.gif") == "builder"
    assert infer_role("instruction_candidate_assembler_ego_with_audio.mp4") is None
