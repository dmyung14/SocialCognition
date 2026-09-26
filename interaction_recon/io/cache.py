import hashlib
import json
from pathlib import Path
from typing import Callable
import uuid

import numpy as np

# Human detection keys deliberately do not receive this salt.
BLOCK_CACHE_GENERATION = "local-table-blocks-1"


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(configuration: dict) -> str:
    configuration = dict(configuration)
    stage = str(configuration.get("stage", ""))
    if stage.startswith(("m2-scene", "camera_world", "common_world")):
        configuration["block_cache_generation"] = BLOCK_CACHE_GENERATION
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_arrays(
    directory: Path,
    stage: str,
    key: str,
    compute: Callable[[], dict[str, np.ndarray]],
    *,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], bool, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}-{key}.npz"
    if path.exists() and not force:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if archive["_cache_key"].item() == key:
                    return {
                        name: archive[name].copy()
                        for name in archive.files if name != "_cache_key"
                    }, True, path
        except (OSError, ValueError, EOFError, KeyError):
            pass
    arrays = compute()
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Cache arrays must not contain pickled Python objects")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, _cache_key=np.array(key), **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return arrays, False, path
