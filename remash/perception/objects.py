"""Flood-fill object detection on 64x64 color-index grids.

Detects connected components of same-colored pixels (4-connected).
No scipy dependency - direct BFS on a 4096-cell grid is trivial.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import xxhash

from remash.perception.frame import Frame, FrameDiff


@dataclass(slots=True)
class GridObject:
    color: int
    pixels: set[tuple[int, int]]  # (x, y) coordinates
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    centroid: tuple[float, float]  # (cx, cy)
    area: int
    shape_hash: int  # position-independent shape hash


@dataclass(slots=True)
class ObjectDelta:
    obj: GridObject  # current frame version
    prev_obj: GridObject | None  # previous frame version, None if new
    moved: tuple[int, int] | None  # (dx, dy) displacement
    color_changed: bool
    shape_changed: bool
    is_new: bool
    is_gone: bool


def _compute_shape_hash(pixels: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int:
    """Hash pixel pattern relative to bbox origin for position-independent comparison."""
    x_min, y_min = bbox[0], bbox[1]
    # Sort for deterministic hashing
    relative = sorted((x - x_min, y - y_min) for x, y in pixels)
    return xxhash.xxh64(str(relative).encode()).intdigest()


def _compute_bbox(pixels: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return (min(xs), min(ys), max(xs), max(ys))


def _compute_centroid(pixels: set[tuple[int, int]]) -> tuple[float, float]:
    n = len(pixels)
    sx = sum(p[0] for p in pixels)
    sy = sum(p[1] for p in pixels)
    return (sx / n, sy / n)


def detect_background_color(frame: Frame) -> int:
    """Detect background as most frequent color in frame borders."""
    grid = frame.grid
    border = np.concatenate([
        grid[0, :], grid[63, :],  # top and bottom rows
        grid[1:63, 0], grid[1:63, 63],  # left and right columns (excluding corners)
    ])
    values, counts = np.unique(border, return_counts=True)
    return int(values[counts.argmax()])


def detect_objects(
    frame: Frame,
    background_color: int | None = None,
    ui_mask: np.ndarray | None = None,
    min_area: int = 1,
) -> list[GridObject]:
    """Run BFS flood fill to detect connected components of same-colored pixels.

    Args:
        frame: The frame to analyze.
        background_color: Color to skip. Auto-detected from borders if None.
        ui_mask: Optional (64,64) bool mask of pixels to exclude.
        min_area: Minimum pixel count to include an object.
    """
    grid = frame.grid
    h, w = grid.shape

    if background_color is None:
        background_color = detect_background_color(frame)

    visited = np.zeros((h, w), dtype=bool)
    if ui_mask is not None:
        visited |= ui_mask

    objects: list[GridObject] = []

    for start_y in range(h):
        for start_x in range(w):
            if visited[start_y, start_x]:
                continue
            color = int(grid[start_y, start_x])
            if color == background_color:
                visited[start_y, start_x] = True
                continue

            # BFS flood fill (4-connected)
            pixels: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque()
            queue.append((start_x, start_y))
            visited[start_y, start_x] = True

            while queue:
                x, y = queue.popleft()
                pixels.add((x, y))
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        if int(grid[ny, nx]) == color:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

            if len(pixels) < min_area:
                continue

            bbox = _compute_bbox(pixels)
            objects.append(GridObject(
                color=color,
                pixels=pixels,
                bbox=bbox,
                centroid=_compute_centroid(pixels),
                area=len(pixels),
                shape_hash=_compute_shape_hash(pixels, bbox),
            ))

    return objects


def track_objects(
    prev_objects: list[GridObject],
    curr_objects: list[GridObject],
    diff: FrameDiff,
) -> list[ObjectDelta]:
    """Match objects between frames using shape hash, color, and proximity.

    Simple greedy matching: for each current object, find best match in previous
    objects by shape_hash + color + centroid distance.
    """
    if not prev_objects and not curr_objects:
        return []

    deltas: list[ObjectDelta] = []
    unmatched_prev = list(range(len(prev_objects)))

    for curr_obj in curr_objects:
        best_match_idx: int | None = None
        best_score = float("inf")

        for i in unmatched_prev:
            prev_obj = prev_objects[i]
            # Score: lower is better
            score = 0.0
            # Shape match bonus
            if prev_obj.shape_hash != curr_obj.shape_hash:
                score += 100.0
            # Color match bonus
            if prev_obj.color != curr_obj.color:
                score += 50.0
            # Centroid distance
            dx = curr_obj.centroid[0] - prev_obj.centroid[0]
            dy = curr_obj.centroid[1] - prev_obj.centroid[1]
            score += (dx * dx + dy * dy) ** 0.5

            if score < best_score:
                best_score = score
                best_match_idx = i

        if best_match_idx is not None and best_score < 200.0:
            prev_obj = prev_objects[best_match_idx]
            unmatched_prev.remove(best_match_idx)

            # Compute displacement from centroid movement
            dx = round(curr_obj.centroid[0] - prev_obj.centroid[0])
            dy = round(curr_obj.centroid[1] - prev_obj.centroid[1])
            moved = (dx, dy) if (dx != 0 or dy != 0) else None

            deltas.append(ObjectDelta(
                obj=curr_obj,
                prev_obj=prev_obj,
                moved=moved,
                color_changed=curr_obj.color != prev_obj.color,
                shape_changed=curr_obj.shape_hash != prev_obj.shape_hash,
                is_new=False,
                is_gone=False,
            ))
        else:
            deltas.append(ObjectDelta(
                obj=curr_obj,
                prev_obj=None,
                moved=None,
                color_changed=False,
                shape_changed=False,
                is_new=True,
                is_gone=False,
            ))

    # Mark unmatched previous objects as gone
    for i in unmatched_prev:
        prev_obj = prev_objects[i]
        deltas.append(ObjectDelta(
            obj=prev_obj,  # use prev as obj since current doesn't exist
            prev_obj=prev_obj,
            moved=None,
            color_changed=False,
            shape_changed=False,
            is_new=False,
            is_gone=True,
        ))

    return deltas
