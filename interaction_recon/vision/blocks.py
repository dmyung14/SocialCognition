"""Local table-relative segmentation and camera-compensated block tracking."""
from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from interaction_recon.vision.table import (
    extend_table_mask, filled_table_hull, valid_image_mask,
)

BLOCK_PERCEPTION_VERSION = "local-table-blocks-1"


@dataclass
class BlockDetection:
    bbox: np.ndarray
    corners: np.ndarray
    confidence: float
    track_id: int = -1
    observed: bool = False
    interpolated: bool = False
    method: str = "candidate"


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    rgbf = rgb.astype(np.float32)
    r, g, b = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    skin = (
        (r > 70) & (r > g * 1.08) & (g > b * 1.04)
        & (r - b > 25) & (hsv[..., 1] > 65) & (hsv[..., 1] < 180)
    )
    return skin.astype(np.uint8) * 255


def gradient_texture(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    bins = np.minimum(
        (np.mod(np.arctan2(gy, gx), np.pi) * (9 / np.pi)).astype(np.int32), 8
    )
    return magnitude, bins


def table_texture_matches(
    lab, interior, surrounding, magnitude, orientation_bins, center,
    contrast_threshold,
):
    inside, outside = lab[interior], lab[surrounding]
    if len(inside) < 16 or len(outside) < 24:
        return False
    if np.linalg.norm(inside.mean(axis=0) - center) >= contrast_threshold:
        return False
    a, b = inside.std(axis=0), outside.std(axis=0)
    if np.linalg.norm(a - b) / max(3.0, np.linalg.norm(a), np.linalg.norm(b)) < 0.40:
        return True

    def histogram(mask):
        values = np.bincount(
            orientation_bins[mask], weights=magnitude[mask], minlength=9
        ).astype(float)
        return values / max(np.linalg.norm(values), 1e-8)

    a, b = float(magnitude[interior].mean()), float(magnitude[surrounding].mean())
    if min(a, b) < 1:
        return max(a, b) < 2
    return np.dot(histogram(interior), histogram(surrounding)) > 0.94 and 0.4 < a / b < 2.5


def _resize_mask(mask, shape):
    if mask is None:
        return np.zeros(shape, np.uint8)
    return cv2.resize(
        np.asarray(mask, np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST
    )


def local_table_model(lab, support, center):
    """Normalized convolution of table-only samples, before skin exclusion.

    Chroma and lightness gates remove foreground from the model, not from the
    support hull. Iteration allows spatially varying illumination. The returned
    scatter is a local photometric noise estimate, not metric uncertainty.
    """
    chroma = np.linalg.norm(lab[..., 1:] - 128, axis=-1)
    center_chroma = float(np.linalg.norm(center[1:] - 128))
    selected = support & (
        np.linalg.norm(lab[..., 1:] - center[1:], axis=-1)
        <= max(6.0, center_chroma * 0.50)
    )
    samples = lab[selected]
    if len(samples) < 16:
        samples = lab[support]
    spread = np.maximum(
        1.4826 * np.median(np.abs(samples - np.median(samples, axis=0)), axis=0),
        [2.0, 1.0, 1.0],
    )
    selected &= chroma >= max(1.0, center_chroma * 0.82)
    selected &= lab[..., 0] <= center[0] + max(6.0, 2.5 * spread[0])
    sigma = max(4.0, min(support.shape) * 0.045)
    model = np.broadcast_to(center, lab.shape).copy()
    scatter = np.broadcast_to(spread, lab.shape).copy()
    for _ in range(3):
        weight = selected.astype(np.float32)
        denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
        numerator = cv2.GaussianBlur(lab * weight[..., None], (0, 0), sigma)
        enough = denominator > 0.025
        model[enough] = numerator[enough] / denominator[enough, None]
        square = cv2.GaussianBlur(
            (lab - model) ** 2 * weight[..., None], (0, 0), sigma
        )
        scatter[enough] = np.sqrt(square[enough] / denominator[enough, None])
        scatter = np.maximum(scatter, [2.0, 1.0, 1.0])
        background_chroma = np.linalg.norm(model[..., 1:] - 128, axis=-1)
        selected = (
            support
            & (chroma >= 0.82 * background_chroma)
            & (np.abs(chroma - background_chroma)
               <= np.maximum(5, background_chroma * 0.40))
            & (lab[..., 0] <= model[..., 0] + np.maximum(6, 2.5 * scatter[..., 0]))
            & (np.abs(lab[..., 1] - model[..., 1])
               <= np.maximum(4, 3 * scatter[..., 1]))
        )
    return model, scatter, selected


def _largest_rectangle(binary):
    """Largest axis-aligned all-foreground rectangle; used in rectified ROIs."""
    heights = np.zeros(binary.shape[1], int)
    best = (0, 0, 0, 0, 0)
    for y, row in enumerate(binary):
        heights = (heights + 1) * row
        stack = []
        for x in range(len(heights) + 1):
            value = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > value:
                left, h = stack.pop()
                area = (x - left) * h
                if area > best[0]:
                    best = area, left, y - h + 1, x - left, h
                start = left
            if not stack or stack[-1][1] < value:
                stack.append((start, value))
    return best


def split_component(mask, minimum_area):
    """Decompose low-rectangularity components in their minAreaRect frame.

    This does not invent a seam between two visually indistinguishable blocks
    whose union is itself a perfect rectangle.
    """
    points = cv2.findNonZero(mask)
    if points is None:
        return []
    rectangle = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rectangle).astype(np.float32)
    area = float(np.count_nonzero(mask))
    rw, rh = rectangle[1]
    if area / max(rw * rh, 1) >= 0.87:
        return [mask]
    w, h = max(2, round(rw) + 1), max(2, round(rh) + 1)
    destination = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    inverse = np.linalg.inv(transform)
    work = cv2.warpPerspective(mask, transform, (w, h), flags=cv2.INTER_NEAREST) != 0
    pieces = []
    for _ in range(8):
        size, x, y, bw, bh = _largest_rectangle(work)
        if size < minimum_area or min(bw, bh) < 3:
            break
        piece = np.zeros((h, w), np.uint8)
        piece[y:y + bh, x:x + bw] = 255
        work[y:y + bh, x:x + bw] = False
        restored = cv2.warpPerspective(
            piece, inverse, mask.shape[::-1], flags=cv2.INTER_NEAREST
        )
        restored[mask == 0] = 0
        pieces.append(restored)
    return pieces or [mask]


class BlockDetector:
    def __init__(self):
        self.support = None
        self.exclusion = None
        self.table_only = None
        self.likelihood = None

    def detect(
        self, rgb, table_mask, hand_mask=None, *, table_color_lab=None,
        pixels_per_meter=None,
    ):
        height, width = rgb.shape[:2]
        table_mask = _resize_mask(table_mask, (height, width))
        hands = _resize_mask(hand_mask, (height, width))
        valid = valid_image_mask(rgb) != 0
        support = (filled_table_hull(table_mask) != 0) & valid
        self.support = support.astype(np.uint8) * 255
        self.exclusion = hands.copy()
        self.table_only = np.zeros_like(support)
        self.likelihood = np.zeros_like(support, np.float32)
        if support.sum() < height * width * 0.01:
            return []
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        samples = lab[support & (hands == 0)]
        if len(samples) < 16:
            return []
        if table_color_lab is None or not np.isfinite(table_color_lab).all():
            bins = (samples // 16).astype(int)
            codes = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
            center = np.median(samples[codes == np.bincount(codes).argmax()], axis=0)
        else:
            center = np.asarray(table_color_lab, np.float32)
        residual = np.linalg.norm(samples - center, axis=1)
        quiet = residual[residual <= np.median(residual)]
        tolerance = max(12.0, min(32.0, float(np.median(quiet)) * 2 + 6))
        support = extend_table_mask(
            rgb, table_mask, center, hands, color_threshold=tolerance
        ) != 0
        model, scatter, table_only = local_table_model(
            lab, support & (hands == 0), center
        )
        chroma = np.linalg.norm(lab[..., 1:] - 128, axis=-1)
        background_chroma = np.linalg.norm(model[..., 1:] - 128, axis=-1)
        ratio = chroma / np.maximum(background_chroma, 2)
        light_delta = lab[..., 0] - model[..., 0]
        light_gate = np.maximum(3 * scatter[..., 0], 0.12 * model[..., 0])
        chroma_drop = background_chroma - chroma
        low_chroma = (ratio < 0.82) & (
            chroma_drop > np.maximum(2.5, 2 * np.linalg.norm(scatter[..., 1:], axis=-1))
        )
        bright = (light_delta > light_gate) & (ratio < 1.65)
        distinct_skin = (
            (skin_mask(rgb) != 0)
            & (ratio >= 0.90)
            & (lab[..., 1] - model[..., 1] > np.maximum(3, 2.5 * scatter[..., 1]))
        )
        radius = max(2, round(min(height, width) * 0.006))
        excluded = cv2.dilate(
            ((hands != 0) | distinct_skin).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2),
        ) != 0
        allowed = support & valid & ~excluded
        likelihood = np.maximum(
            np.clip((1 - ratio) / 0.35, 0, 1),
            np.clip(light_delta / np.maximum(light_gate * 2, 1), 0, 1),
        )
        candidate = ((low_chroma | bright) & allowed).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate[~allowed] = 0
        self.support = support.astype(np.uint8) * 255
        self.exclusion = excluded.astype(np.uint8) * 255
        self.table_only = table_only & ~excluded & (candidate == 0)
        self.likelihood = likelihood * allowed

        minimum_area = height * width * 0.0006
        if pixels_per_meter is not None and np.isfinite(pixels_per_meter):
            minimum_area = max(minimum_area, (0.005 * pixels_per_meter) ** 2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        detections = []
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if not minimum_area <= area <= height * width * 0.12:
                continue
            component = (labels == label).astype(np.uint8) * 255
            for piece in split_component(component, minimum_area):
                points = cv2.findNonZero(piece)
                if points is None or len(points) < minimum_area:
                    continue
                rectangle = cv2.minAreaRect(points)
                rw, rh = rectangle[1]
                if min(rw, rh) < 3 or max(rw, rh) / min(rw, rh) > 8:
                    continue
                rectangularity = min(1.0, len(points) / max(rw * rh, 1))
                if rectangularity < 0.55:
                    continue
                x, y, w, h = cv2.boundingRect(points)
                confidence = float(np.clip(
                    rectangularity * np.mean(likelihood[piece != 0]), 0.05, 1
                ))
                detections.append(BlockDetection(
                    np.array([x, y, w, h], np.float32),
                    cv2.boxPoints(rectangle).astype(np.float32), confidence,
                ))
        return sorted(detections, key=lambda d: (d.bbox[0], d.bbox[1]))


def _iou(a, b):
    low, high = np.maximum(a[:2], b[:2]), np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    intersection = float(np.prod(np.maximum(0, high - low)))
    return intersection / max(float(np.prod(a[2:]) + np.prod(b[2:]) - intersection), 1e-9)


def warp_points(points, transform):
    return cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(-1, 1, 2),
        np.asarray(transform, float),
    ).reshape(-1, 2)


def region_fraction(corners, mask):
    if not np.isfinite(corners).all():
        return 0.0
    region = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(region, np.round(corners).astype(np.int32), 1)
    selected = region != 0
    return float(np.mean(mask[selected] != 0)) if selected.any() else 0.0


class BlockTracker:
    """Compatibility online detector tracker with causal confirmation.

    The scene stage uses PersistentBlockTracker and a full SOURCE-clip support
    audit. This small API remains available to existing standalone callers.
    """
    def __init__(self, max_missing=8):
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 0

    def update(self, detections, image_shape):
        ids = list(self.tracks)
        diagonal = float(np.hypot(*image_shape[:2]))
        for track in self.tracks.values():
            track["missing"] += 1
            track["history"].append(False)
        cost = np.full((len(ids), len(detections)), 1e6)
        for row, identity in enumerate(ids):
            before = self.tracks[identity]["bbox"]
            for col, detection in enumerate(detections):
                box = detection.bbox
                distance = np.linalg.norm(before[:2] + before[2:] / 2 - box[:2] - box[2:] / 2)
                ratio = np.prod(box[2:]) / max(np.prod(before[2:]), 1)
                if distance <= diagonal * 0.12 and 0.25 <= ratio <= 4:
                    cost[row, col] = distance / max(diagonal, 1) + 0.6 * (1 - _iou(before, box))
        assigned = {}
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            assigned = {int(c): ids[r] for r, c in zip(rows, cols) if cost[r, c] < 1e6}
        for index, detection in enumerate(detections):
            identity = assigned.get(index)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
                history = deque([True], maxlen=5)
            else:
                history = self.tracks[identity]["history"]
                history[-1] = True
            detection.track_id = identity
            detection.observed = sum(history) >= 3
            self.tracks[identity] = {
                "bbox": detection.bbox.copy(), "history": history, "missing": 0,
            }
        self.tracks = {
            key: value for key, value in self.tracks.items()
            if value["missing"] <= self.max_missing
        }
        return detections


class PersistentBlockTracker:
    """Constant-position tracks in a stabilized table-image coordinate chart."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections, image_from_table, exclusion, image_shape):
        table_from_image = np.linalg.inv(image_from_table)
        ids = list(self.tracks)
        canonical = [warp_points(d.corners, table_from_image) for d in detections]
        occluded = {}
        for identity in ids:
            track = self.tracks[identity]
            projected = warp_points(track["corners"], image_from_table)
            occluded[identity] = region_fraction(projected, exclusion) >= 0.12
            track["touched"] = track["touched"] or occluded[identity]
        cost = np.full((len(ids), len(detections)), 1e6)
        for row, identity in enumerate(ids):
            track = self.tracks[identity]
            before = track["corners"]
            old_size = np.sort(cv2.minAreaRect(before.astype(np.float32))[1])
            length = max(float(old_size[-1]), 5)
            gate = length * (3.0 if track["touched"] else 0.65)
            for col, points in enumerate(canonical):
                size = np.sort(cv2.minAreaRect(points.astype(np.float32))[1])
                ratio = np.maximum(size, 1) / np.maximum(old_size, 1)
                distance = np.linalg.norm(points.mean(axis=0) - before.mean(axis=0))
                if distance <= gate and np.all((ratio > 0.45) & (ratio < 2.2)):
                    cost[row, col] = distance / gate + np.linalg.norm(np.log(ratio))
        assigned = {}
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            assigned = {int(c): ids[r] for r, c in zip(rows, cols) if cost[r, c] < 1e6}
        result, matched = [], set()
        for col, detection in enumerate(detections):
            identity = assigned.get(col)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
            matched.add(identity)
            self.tracks[identity] = {
                "corners": canonical[col].copy(),
                "confidence": detection.confidence,
                "touched": False,
            }
            result.append(replace(
                detection, track_id=identity, observed=True, method="observed",
            ))
        for identity in ids:
            if identity in matched or not occluded[identity]:
                continue
            track = self.tracks[identity]
            corners = warp_points(track["corners"], image_from_table)
            x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
            result.append(BlockDetection(
                np.array([x, y, w, h], np.float32), corners,
                track["confidence"] * 0.25, identity, False, True, "occluded_hold",
            ))
        return result
