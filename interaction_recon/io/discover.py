from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Iterator
import zipfile

from interaction_recon.contracts import SourceRef

MEDIA_SUFFIXES = {".mp4", ".gif"}
MAX_ZIP_MEMBERS = 100_000
MAX_ZIP_MEDIA_BYTES = 20 * 1024**3
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
SEGMENT_RE = re.compile(r"segment_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveredMedia:
    id: str
    local_path: Path
    source: SourceRef
    segment_id: str | None
    role: str | None


@dataclass(frozen=True)
class Discovery:
    kind: str
    input_path: str
    media: list[DiscoveredMedia]


def infer_segment(name: str) -> str | None:
    match = SEGMENT_RE.search(name)
    if match is None:
        return None
    # Avoid integer conversion and its digit-count limit.
    return (match.group(1).lstrip("0") or "0").zfill(3)


def infer_role(name: str) -> str | None:
    lowered = name.casefold()
    guider = "instruction_candidate" in lowered
    builder = "assembler_ego_with_audio" in lowered
    if guider == builder:
        return None
    return "guider" if guider else "builder"


def _junk(parts: tuple[str, ...]) -> bool:
    return any(
        part.startswith(".")
        or part.casefold() == "__macosx"
        for part in parts
    )


def _sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _item(
    path: Path, member: str, kind: str, original_input: Path
) -> DiscoveredMedia:
    return DiscoveredMedia(
        id=member,
        local_path=path,
        source=SourceRef(kind, str(original_input), member),
        segment_id=infer_segment(path.name),
        role=infer_role(path.name),
    )


def _folder_media(root: Path) -> list[DiscoveredMedia]:
    result = []

    def walk_error(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        current_path = Path(current)
        directories[:] = sorted(
            (
                name for name in directories
                if not _junk((name,))
                and not (current_path / name).is_symlink()
            ),
            key=_sort_key,
        )
        for name in sorted(files, key=_sort_key):
            path = current_path / name
            if (
                _junk((name,))
                or path.is_symlink()
                or path.suffix.casefold() not in MEDIA_SUFFIXES
            ):
                continue
            member = path.relative_to(root).as_posix()
            result.append(_item(path, member, "folder", root))
    return sorted(result, key=lambda item: _sort_key(item.id))


def _zip_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    name = info.filename.replace("\\", "/")
    if not name or name.startswith("/") or "\x00" in name:
        raise ValueError(f"Unsafe ZIP path: {info.filename!r}")
    parts = tuple(part for part in name.split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Unsafe ZIP path: {info.filename!r}")
    for part in parts:
        if (
            any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED
        ):
            raise ValueError(f"Windows-unsafe ZIP path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlinks are not supported: {info.filename!r}")
    return parts


def _extract_media(archive: Path, root: Path) -> list[DiscoveredMedia]:
    result = []
    try:
        with zipfile.ZipFile(archive) as handle:
            entries = handle.infolist()
            if len(entries) > MAX_ZIP_MEMBERS:
                raise ValueError("ZIP exceeds the member-count safety limit")
            selected = []
            seen = set()
            total_bytes = 0
            # Validate before writing any selected file.
            for info in entries:
                parts = _zip_parts(info)
                if _junk(parts) or info.is_dir():
                    continue
                member = PurePosixPath(*parts).as_posix()
                if PurePosixPath(member).suffix.casefold() not in MEDIA_SUFFIXES:
                    continue
                key = member.casefold()
                if key in seen:
                    raise ValueError(f"Duplicate ZIP media path: {member}")
                seen.add(key)
                total_bytes += info.file_size
                if total_bytes > MAX_ZIP_MEDIA_BYTES:
                    raise ValueError("ZIP exceeds the uncompressed-media limit")
                selected.append((member, parts, info))
            for member, parts, info in sorted(
                selected, key=lambda entry: _sort_key(entry[0])
            ):
                destination = root.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                result.append(_item(destination, member, "zip", archive))
    except (
        zipfile.BadZipFile, zipfile.LargeZipFile,
        RuntimeError, NotImplementedError,
    ) as exc:
        raise ValueError(f"Cannot read ZIP {archive}: {exc}") from exc
    return result


@contextmanager
def discover_input(input_path: str | Path) -> Iterator[Discovery]:
    """Keep extracted paths valid only for the lifetime of this context."""
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        yield Discovery("folder", str(path), _folder_media(path))
        return
    if not path.is_file():
        raise ValueError(f"Input does not exist or is not readable: {path}")
    if path.suffix.casefold() != ".zip":
        raise ValueError("run --input must be a directory or a .zip file")
    with tempfile.TemporaryDirectory(prefix="interaction-recon-") as temporary:
        media = _extract_media(path, Path(temporary))
        yield Discovery("zip", str(path), media)


def explicit_media(path: str | Path, role: str) -> DiscoveredMedia:
    if role not in {"guider", "builder"}:
        raise ValueError(f"Unknown explicit role: {role}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{role} media does not exist: {resolved}")
    if resolved.suffix.casefold() not in MEDIA_SUFFIXES:
        raise ValueError(f"{role} media must be .mp4 or .gif: {resolved}")
    return DiscoveredMedia(
        id=f"{role}/{resolved.name}",
        local_path=resolved,
        source=SourceRef("file", str(resolved)),
        segment_id=infer_segment(resolved.name),
        role=role,
    )
