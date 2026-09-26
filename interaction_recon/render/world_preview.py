from pathlib import Path
import subprocess
import tempfile
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from interaction_recon.io.media import ffmpeg_executable
from interaction_recon.vision.hands import HAND_EDGES


def _pixel(point, front: bool, extent):
    x, y, z = point
    scale = 180 / max(1.0, float(np.max(extent)))
    return (
        int(np.clip(240 + x * scale, -10000, 10000)),
        int(np.clip((360 - z * scale) if front else (240 - y * scale), -10000, 10000)),
    )


def _lines(image, points, edges, color, front, extent):
    valid = np.isfinite(points).all(axis=-1)
    for a, b in edges:
        if valid[a] and valid[b]:
            cv2.line(
                image, _pixel(points[a], front, extent), _pixel(points[b], front, extent),
                color, 2, cv2.LINE_AA,
            )
    for point in points[valid]:
        cv2.circle(image, _pixel(point, front, extent), 2, color, -1)


def preview_frame(arrays: dict, index: int) -> np.ndarray:
    canvas = np.zeros((544, 960, 3), np.uint8)
    extent = arrays["table_extent_xy"]
    table = np.array([
        [-extent[0] / 2, -extent[1] / 2, 0],
        [extent[0] / 2, -extent[1] / 2, 0],
        [extent[0] / 2, extent[1] / 2, 0],
        [-extent[0] / 2, extent[1] / 2, 0],
    ])
    for view, front in enumerate((False, True)):
        panel = canvas[:480, view * 480:(view + 1) * 480]
        panel[:] = (24, 27, 31)
        _lines(panel, table, ((0, 1), (1, 2), (2, 3), (3, 0)),
               (130, 110, 80), front, extent)
        for role, color in (("guider", (60, 230, 110)), ("builder", (255, 170, 60))):
            for side in ("left", "right"):
                name = f"{role}_{side}_hand_21"
                points = arrays[name][index].copy()
                points[arrays[f"{name}_confidence"][index] <= 0] = np.nan
                _lines(panel, points, HAND_EDGES, color, front, extent)
            name = "guider_arm_keypoints" if role == "guider" else "builder_forearm_keypoints"
            arms = arrays[name][index]
            for arm in arms:
                edges = ((0, 1), (1, 2)) if len(arm) == 3 else ((0, 1),)
                _lines(panel, arm, edges, color, front, extent)
        for name, radius in (("guider_head_pose", 9), ("guider_torso_pose", 6)):
            point = arrays[name][index, :3]
            if np.isfinite(point).all():
                cv2.circle(panel, _pixel(point, front, extent), radius, (60, 230, 110), 2)

        for pose, dimensions in zip(arrays["block_poses"][index], arrays["block_dimensions"][index]):
            if not np.isfinite(pose).all() or not np.isfinite(dimensions).all():
                continue
            corners = np.array([
                [x, y, z] for z in (-0.5, 0.5)
                for y in (-0.5, 0.5) for x in (-0.5, 0.5)
            ]) * dimensions
            rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            corners = corners @ rotation.T + pose[:3]
            edges = (
                (0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
                (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7),
            )
            _lines(panel, corners, edges, (220, 220, 230), front, extent)
        for camera, color in zip(
            arrays["camera_poses"][index], ((100, 200, 255), (240, 100, 230))
        ):
            if not np.isfinite(camera).all():
                continue
            local = np.array([
                [0, 0, 0], [-0.12, -0.09, 0.2], [0.12, -0.09, 0.2],
                [0.12, 0.09, 0.2], [-0.12, 0.09, 0.2],
            ])
            world = local @ camera[:3, :3].T + camera[:3, 3]
            _lines(panel, world, (
                (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)
            ), color, front, extent)
        cv2.putText(
            panel, "FRONT: x / z" if front else "TOP: x / y; guider +y",
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA,
        )
    text = (
        f"t={arrays['timestamps'][index]:.3f}s | "
        f"{arrays['fusion_method'].item()} | confidence={float(arrays['fusion_confidence']):.2f}"
    )
    cv2.putText(canvas, text, (10, 505), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Uncertain metric priors | interaction NOT verified | no physics",
                (10, 531), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 90), 1, cv2.LINE_AA)
    return canvas


def render_world_preview(output: Path, arrays: dict, fps: float) -> None:
    temporary = output.with_name(f".world-{uuid.uuid4().hex}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x544",
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
                for i in range(len(arrays["timestamps"])):
                    process.stdin.write(preview_frame(arrays, i).tobytes())
                process.stdin.close()
                code = process.wait(timeout=120)
                if code:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
            except (BrokenPipeError, subprocess.TimeoutExpired) as exc:
                process.kill()
                process.wait()
                errors.seek(0)
                raise RuntimeError(
                    "World-preview FFmpeg failed: " + errors.read().decode(errors="replace")
                ) from exc
        temporary.replace(output)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        temporary.unlink(missing_ok=True)
