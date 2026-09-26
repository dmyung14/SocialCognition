"""Synchronized source/source/physics comparison; source PTS are not rebased."""
from pathlib import Path

import cv2
import numpy as np

from interaction_recon.io.media import MediaSequence
from interaction_recon.render.video import write_mp4


def source_frame(sequence: MediaSequence, time: float) -> np.ndarray | None:
    start = float(sequence.source_timestamps_s[0])
    if not start <= time < start + sequence.duration_s:
        return None
    index = int(np.clip(
        np.searchsorted(sequence.timestamps_s, time + 1e-9, side="right") - 1,
        0, len(sequence.rgb_frames) - 1,
    ))
    return sequence.rgb_frames[index]


def _text(image, text: str, position: tuple[int, int], scale: float = 0.48):
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, (235, 235, 235), 1, cv2.LINE_AA,
    )


def panel(
    image: np.ndarray | None, label: str, width: int, height: int,
    missing: str = "no source frame",
) -> np.ndarray:
    result = np.full((height + 28, width, 3), 18, np.uint8)
    if image is None:
        _text(result, missing, (12, 28 + height // 2))
    else:
        scale = min(width / image.shape[1], height / image.shape[0])
        w, h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        x, y = (width - w) // 2, 28 + (height - h) // 2
        result[y:y + h, x:x + w] = resized
    _text(result, label, (8, 19), min(0.48, width / 850))
    return result


def compose_frame(
    guider: np.ndarray | None, builder: np.ndarray | None,
    reconstruction: np.ndarray | None, *, status: tuple[str, str],
    panel_width: int = 480, panel_height: int = 360,
) -> np.ndarray:
    if panel_width % 2 or panel_height % 2:
        raise ValueError("Comparison panel dimensions must be even")
    result = np.full((panel_height + 92, panel_width * 3, 3), 12, np.uint8)
    for index, (image, label) in enumerate((
        (guider, "source guider"),
        (builder, "source builder"),
        (reconstruction, "fused MuJoCo reconstruction"),
    )):
        result[:panel_height + 28, index * panel_width:(index + 1) * panel_width] = panel(
            image, label, panel_width, panel_height,
            "no physics frame" if index == 2 else "no source frame",
        )
    for i, line in enumerate(status):
        # Fit long method names without silently truncating confidence/error text.
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        scale = min(0.48, 0.48 * (result.shape[1] - 16) / max(text_width, 1))
        _text(result, line, (8, panel_height + 51 + i * 26), scale)
    return result


def physics_status(metrics: dict) -> str:
    selected = metrics.get("refinement_selection_status", "")
    label = "PHYSICS (refined)" if selected == "refined_valid" else "PHYSICS (initial retained)"
    position = metrics.get("block_trajectory_position_error_m")
    normalized = metrics.get("block_trajectory_error_normalized_by_length")
    error = (
        f"block error {position:.3f} m / {normalized:.3f} BL; target <=0.25 BL"
        if position is not None and normalized is not None
        else "block error unavailable: insufficient observed reference"
    )
    return f"{label} | {error}"


def render_comparison(
    output: Path, guider: MediaSequence, builder: MediaSequence,
    simulation: Path, simulation_start_s: float, sync: dict, fusion_method: str,
    physics: dict, *, panel_width: int = 480, panel_height: int = 360,
) -> int:
    offset = float(sync["offset_s"])
    start = min(guider.source_timestamps_s[0], builder.source_timestamps_s[0] - offset)
    end = max(
        guider.source_timestamps_s[0] + guider.duration_s,
        builder.source_timestamps_s[0] + builder.duration_s - offset,
    )
    capture = cv2.VideoCapture(str(simulation))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode physics movie: {simulation}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("Physics movie has invalid FPS")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = max(1, int(np.ceil((end - start) * 30 - 1e-8)))

    def frames():
        current_index, current = -1, None
        for index in range(count):
            time = float(start + index / 30)
            requested = int(np.floor((time - simulation_start_s) * fps + 1e-7))
            reconstruction = None
            if 0 <= requested < total:
                while current_index < requested:
                    ok, bgr = capture.read()
                    if not ok:
                        raise RuntimeError("Unexpected end of physics movie")
                    current = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    current_index += 1
                reconstruction = current
            yield compose_frame(
                source_frame(guider, time), source_frame(builder, time + offset),
                reconstruction,
                status=(
                    f"t={time:.3f}s | sync {sync['method']} "
                    f"confidence={sync['confidence']:.3f} offset={offset:+.3f}s | "
                    f"fusion {fusion_method} (shared interaction unverified)",
                    physics_status(physics),
                ),
                panel_width=panel_width, panel_height=panel_height,
            )

    try:
        return write_mp4(
            output, frames(), panel_width * 3, panel_height + 92, 30
        )
    finally:
        capture.release()
