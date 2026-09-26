from pathlib import Path
import subprocess
import tempfile
import uuid

import cv2
import numpy as np

from interaction_recon.io.media import MediaSequence, ffmpeg_executable
from interaction_recon.vision.body import POSE_EDGES
from interaction_recon.vision.hands import HAND_EDGES


def _skeleton(
    image: np.ndarray,
    points: np.ndarray,
    edges: tuple,
    color: tuple[int, int, int],
    *,
    minimum_confidence: float = 0.2,
) -> None:
    valid = (
        np.isfinite(points[:, :2]).all(axis=1)
        & (points[:, 3] >= minimum_confidence)
    )
    for a, b in edges:
        if valid[a] and valid[b]:
            pa, pb = np.round(points[[a, b], :2]).astype(int)
            cv2.line(image, tuple(pa), tuple(pb), color, 2, cv2.LINE_AA)
    for point in points[valid, :2]:
        cv2.circle(image, tuple(np.round(point).astype(int)), 2, color, -1)


def overlay(rgb: np.ndarray, observations: dict, index: int) -> np.ndarray:
    image = rgb.copy()
    height, width = image.shape[:2]
    mask = cv2.resize(
        observations["table_masks"][index], (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (30, 210, 255), 2)
    _skeleton(
        image, observations["pose"][index], POSE_EDGES, (50, 230, 80),
        minimum_confidence=0.5,
    )
    for slot, hand in enumerate(observations["hands"][index]):
        owner = observations["hand_owner"][index, slot]
        color = (255, 180, 40) if owner == 1 else (220, 70, 230)
        _skeleton(image, hand, HAND_EDGES, color)
        if hand[0, 3] > 0 and np.isfinite(hand[0, :2]).all():
            handedness = observations["handedness"][index, slot]
            label = {-1: "?", 0: "L", 1: "R"}[int(handedness)]
            who = {-1: "unknown", 0: "other", 1: "wearer"}[int(owner)]
            point = tuple(np.round(hand[0, :2]).astype(int))
            cv2.putText(
                image, f"{label} {who}", point, cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA,
            )
    for slot, track_id in enumerate(observations["block_ids"][index]):
        if track_id < 0:
            continue
        corners = observations["block_corners"][index, slot].astype(np.int32)
        cv2.polylines(image, [corners], True, (255, 70, 40), 2)
        x, y, _, _ = observations["block_boxes"][index, slot].astype(int)
        cv2.putText(
            image, f"B{track_id}", (x, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 70, 40), 1, cv2.LINE_AA,
        )
    center = observations["head_center"][index]
    transform = observations["head_transform"][index]
    if center[3] > 0 and np.isfinite(transform).all():
        origin = center[:2]
        rotation = transform[:3, :3].astype(float)
        rotation /= np.maximum(np.linalg.norm(rotation, axis=0), 1e-8)
        for axis, color in enumerate(((255, 50, 50), (50, 255, 50), (50, 100, 255))):
            endpoint = origin + rotation[:2, axis] * [35, -35]
            cv2.line(
                image, tuple(origin.astype(int)), tuple(endpoint.astype(int)),
                color, 2, cv2.LINE_AA,
            )
        methods = observations.get("head_method")
        if methods is not None and methods[index] == "pose_landmarks_fallback":
            cv2.putText(
                image, "head: pose fallback",
                (int(origin[0]), max(14, int(origin[1]) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 190, 80), 1, cv2.LINE_AA,
            )
    return image


def _panel(
    sequence: MediaSequence,
    observations: dict,
    source_time: float,
    label: str,
    size: int,
) -> np.ndarray:
    panel = np.zeros((size, size, 3), np.uint8)
    start = sequence.source_timestamps_s[0]
    valid = start <= source_time < start + sequence.duration_s
    if valid:
        index = int(np.searchsorted(sequence.sample_times_s, source_time, side="right") - 1)
        index = max(0, min(index, len(sequence.rgb_frames) - 1))
        frame = overlay(sequence.rgb_frames[index], observations, index)
        scale = min(size / sequence.width, size / sequence.height)
        width = max(1, round(sequence.width * scale))
        height = max(1, round(sequence.height * scale))
        frame = cv2.resize(frame, (width, height))
        x, y = (size - width) // 2, (size - height) // 2
        panel[y:y + height, x:x + width] = frame
    else:
        cv2.putText(
            panel, "No source frame", (20, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA,
        )
    cv2.rectangle(panel, (0, 0), (size, 28), (0, 0, 0), -1)
    cv2.putText(
        panel, f"{label} source t={source_time:.3f}s", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def render_tracking(
    output: Path,
    guider: MediaSequence,
    builder: MediaSequence,
    guider_observations: dict,
    builder_observations: dict,
    sync: dict,
    *,
    fps: float = 15.0,
    panel_size: int = 480,
) -> None:
    """Render the union of both durations; missing intervals are black, not frozen."""
    offset = sync["offset_s"]
    start = min(guider.source_timestamps_s[0], builder.source_timestamps_s[0] - offset)
    end = max(
        guider.source_timestamps_s[0] + guider.duration_s,
        builder.source_timestamps_s[0] + builder.duration_s - offset,
    )
    width, height = panel_size * 2, panel_size + 64
    temporary = output.with_name(f".tracking-{uuid.uuid4().hex}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(temporary),
    ]
    process = None
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
            )
            try:
                count = max(1, int(np.ceil((end - start) * fps - 1e-8)))
                for index in range(count):
                    time = start + index / fps
                    frame = np.zeros((height, width, 3), np.uint8)
                    frame[:panel_size, :panel_size] = _panel(
                        guider, guider_observations, time, "Guider", panel_size
                    )
                    frame[:panel_size, panel_size:] = _panel(
                        builder, builder_observations, time + offset, "Builder", panel_size
                    )
                    text = (
                        f"tb=tg{offset:+.3f}s  {sync['method']}  "
                        f"confidence={sync['confidence']:.3f}  "
                        + ("alignment hypothesis" if sync["reliable"] else "LOW CONFIDENCE")
                    )
                    cv2.putText(
                        frame, text, (8, panel_size + 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, "2D only | shared interaction NOT verified | no 3D fusion",
                        (8, panel_size + 49), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 190, 80), 1, cv2.LINE_AA,
                    )
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                code = process.wait(timeout=120)
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                process.kill()
                process.wait()
                errors.seek(0)
                raise RuntimeError(
                    "Tracking FFmpeg failed: " + errors.read().decode(errors="replace")
                ) from exc
            if code:
                errors.seek(0)
                raise RuntimeError(
                    "Tracking FFmpeg failed: " + errors.read().decode(errors="replace")
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
