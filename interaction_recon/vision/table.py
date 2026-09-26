from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TableObservation:
    mask: np.ndarray
    contour: np.ndarray
    color_lab: np.ndarray
    confidence: float


def valid_image_mask(rgb: np.ndarray) -> np.ndarray:
    """Exclude dark pixels and a supported circular fisheye vignette."""
    value = rgb.max(axis=2)
    valid = (value >= 25).astype(np.uint8) * 255
    height, width = valid.shape
    contours, _ = cv2.findContours(
        valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return valid
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    corners = value[
        [0, 0, height - 1, height - 1],
        [0, width - 1, 0, width - 1],
    ]
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
    if (
        np.all(corners < 25)
        and area > width * height * 0.3
        and circularity > 0.72
    ):
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        yy, xx = np.ogrid[:height, :width]
        valid[(xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2] = 0
    return valid


def filled_table_hull(mask: np.ndarray) -> np.ndarray:
    """Fill holes in the largest connected table component."""
    binary = (mask != 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filled = np.zeros_like(binary)
    if count <= 1:
        return filled
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    points = cv2.findNonZero((labels == label).astype(np.uint8))
    if points is not None and len(points) >= 3:
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 255)
    return filled


def _lower_component(component: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Stop a bottom-up traversal at a sustained width collapse."""
    height = component.shape[0]
    widths = np.count_nonzero(component, axis=1).astype(np.float32)
    seed_rows = np.flatnonzero(np.any(seed & component, axis=1))
    if not len(seed_rows):
        return component
    anchor = int(np.median(seed_rows))
    window = max(1, round(height * 0.012))
    smooth = np.convolve(widths, np.ones(window) / window, mode="same")
    reference = float(np.percentile(smooth[seed_rows], 75))
    if reference <= 0:
        return component
    run = 0
    cutoff = 0
    for row in range(anchor, -1, -1):
        if smooth[row] < reference * 0.48:
            run += 1
        else:
            run = 0
            reference = max(reference, float(smooth[row]))
        if run >= window:
            cutoff = row + run
            break
    result = component.copy()
    result[:cutoff] = False
    return result


def extend_far_edge(
    hull: np.ndarray,
    table_colored: np.ndarray,
    valid: np.ndarray,
    exclusion: np.ndarray | None = None,
) -> np.ndarray:
    """Recover supported far-edge rows without following narrow wall bridges.

    Each added row needs substantial table-colored support on both sides.
    Foreground between those supports is included as tabletop occupancy, rather
    than being mistaken for the end of the table. No global convex hull is
    taken after this extension.
    """
    result = (hull != 0).astype(np.uint8) * 255
    valid = valid != 0
    colored = (table_colored != 0) & valid
    if exclusion is not None:
        colored &= exclusion == 0

    height, width = result.shape
    rows = np.flatnonzero(np.any(result != 0, axis=1))
    if not len(rows):
        return result

    top = int(rows[0])
    anchor = min(int(rows[-1]), top + max(2, round(height * 0.06)))
    anchor_points = np.flatnonzero(result[anchor])
    if len(anchor_points) < max(8, width * 0.08):
        return result

    left, right = int(anchor_points[0]), int(anchor_points[-1])
    reference_width = right - left + 1
    lower_limit = max(0, top - round(height * 0.18))
    horizontal_margin = max(2, round(width * 0.025))
    row_radius = max(1, round(height * 0.003))
    minimum_support = max(3, round(reference_width * 0.035))

    for row in range(anchor, lower_limit - 1, -1):
        lo = max(0, left - horizontal_margin)
        hi = min(width, right + horizontal_margin + 1)
        y0 = max(0, row - row_radius)
        y1 = min(height, row + row_radius + 1)
        support = np.mean(colored[y0:y1, lo:hi], axis=0) >= 0.5
        positions = np.flatnonzero(support) + lo
        if len(positions) < minimum_support * 2:
            break

        span = right - left + 1
        left_support = positions[
            (positions >= lo) & (positions <= left + span * 0.30)
        ]
        right_support = positions[
            (positions >= right - span * 0.30) & (positions < hi)
        ]
        if (
            len(left_support) < minimum_support
            or len(right_support) < minimum_support
        ):
            break

        new_left = int(left_support[0])
        new_right = int(right_support[-1])
        new_width = new_right - new_left + 1
        if new_width < reference_width * 0.55:
            break

        # A wood-colored post between table and wall must not count as a row.
        supported_fraction = (
            np.count_nonzero(
                (positions >= new_left) & (positions <= new_right)
            ) / new_width
        )
        if supported_fraction < 0.35:
            break

        # These spans deliberately include objects bounded by wood on both sides.
        result[row, new_left:new_right + 1] = 255
        left, right = new_left, new_right

    result[~valid] = 0
    return result


def extend_table_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_lab: np.ndarray,
    exclusion: np.ndarray | None = None,
    *,
    color_threshold: float = 18.0,
) -> np.ndarray:
    """Repair a cached/coarse hull using current-frame color evidence."""
    height, width = rgb.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
    if exclusion is not None and exclusion.shape != (height, width):
        exclusion = cv2.resize(
            exclusion, (width, height), interpolation=cv2.INTER_NEAREST
        )
    valid = valid_image_mask(rgb) != 0
    if not np.isfinite(color_lab).all():
        result = filled_table_hull(mask)
        result[~valid] = 0
        return result
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    colored = np.linalg.norm(lab - color_lab, axis=2) <= color_threshold
    return extend_far_edge(
        filled_table_hull(mask), colored, valid, exclusion
    )


class TableDetector:
    """Seed-connected tabletop with a color-supported far-edge extension."""

    def detect(
        self, rgb: np.ndarray, exclusion: np.ndarray | None = None
    ) -> TableObservation:
        height, width = rgb.shape[:2]
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        valid = valid_image_mask(rgb) != 0
        yy, xx = np.mgrid[:height, :width]
        seed = (
            (yy >= height * 0.60) & (yy < height * 0.90)
            & (xx >= width * 0.25) & (xx < width * 0.75)
            & valid
        )
        if exclusion is not None:
            seed &= exclusion == 0
        samples = lab[seed]
        empty = TableObservation(
            np.zeros((height, width), np.uint8),
            np.empty((0, 1, 2), np.int32),
            np.full(3, np.nan, np.float32),
            0.0,
        )
        if len(samples) < 16:
            return empty

        bins = (samples // 16).astype(np.int32)
        codes = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
        dominant = np.bincount(codes).argmax()
        selected = samples[codes == dominant]
        center = np.median(selected, axis=0)
        spread = np.percentile(np.linalg.norm(selected - center, axis=1), 95)
        threshold = max(12.0, min(32.0, float(spread) * 2.5))
        distance = np.linalg.norm(lab - center, axis=2)
        raw_colored = (distance <= threshold) & valid
        if exclusion is not None:
            raw_colored &= exclusion == 0

        size = max(3, round(min(width, height) * 0.012) | 1)
        colored = cv2.morphologyEx(
            raw_colored.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((size, size), np.uint8),
        )
        colored[~valid] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
        candidates = [
            label for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= width * height * 0.03
            and np.any(seed & (labels == label))
        ]
        if not candidates:
            return empty

        label = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        component = _lower_component(labels == label, seed)
        hull = extend_far_edge(
            filled_table_hull(component), raw_colored, valid, exclusion
        )
        contours, _ = cv2.findContours(
            hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return empty
        contour = max(contours, key=cv2.contourArea)
        coverage = np.count_nonzero(seed & (hull != 0)) / max(
            1, np.count_nonzero(seed)
        )
        dominance = len(selected) / len(samples)
        confidence = float(np.clip(coverage * np.sqrt(dominance), 0, 1))
        return TableObservation(hull, contour, center, confidence)
