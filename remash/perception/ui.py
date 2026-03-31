"""Adaptive UI element detection for ARC-AGI-3 games.

Strategy: detect energy bars and lives indicators from frame patterns,
not hardcoded positions. Works by finding horizontal runs of consistent
color in the bottom rows that shrink between frames.

UI layout varies between games. Returns None for undetected elements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xxhash

from remash.perception.frame import Frame
from remash.perception.objects import GridObject


@dataclass(slots=True)
class UIState:
    energy: float | None  # 0.0-1.0, None if not detected
    lives: int | None
    target_shape: GridObject | None
    ui_region_mask: np.ndarray | None  # (64, 64) bool
    shape_display_hash: int | None  # hash of the bottom-left shape display region


@dataclass
class _BarConfig:
    """Cached energy bar configuration, detected from initial frames."""

    rows: list[int]  # which rows the bar occupies
    bar_color: int  # the color of the bar itself
    col_start: int  # leftmost column of bar region
    col_end: int  # rightmost column of bar region (exclusive)
    initial_length: int  # bar length per row at first detection
    indicator_color: int | None  # color of life indicators (small blocks)
    indicator_positions: list[tuple[int, int]]  # (x, y) of each indicator block


# Minimum horizontal run length to be considered an energy bar candidate
_MIN_BAR_LENGTH = 8

# How many bottom rows to scan for UI elements
_SCAN_ROWS = 10

# Fallback step budget when energy bar is not detected
_FALLBACK_ENERGY_BUDGET = 50

# Fallback UI mask: mark bottom N rows as UI when bar detection fails
_FALLBACK_UI_ROWS = 4


def _find_horizontal_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Find all maximal horizontal runs of same color in a row.

    Returns list of (color, start_col, length) sorted by length descending.
    """
    runs: list[tuple[int, int, int]] = []
    if len(row) == 0:
        return runs

    current_color = int(row[0])
    start = 0

    for i in range(1, len(row)):
        c = int(row[i])
        if c != current_color:
            runs.append((current_color, start, i - start))
            current_color = c
            start = i
    runs.append((current_color, start, len(row) - start))

    runs.sort(key=lambda r: -r[2])
    return runs


def _detect_bar_config(grid: np.ndarray) -> _BarConfig | None:
    """Detect the energy bar from a frame grid.

    Strategy: the energy bar is a color that appears as a long horizontal
    run in only 2-4 adjacent bottom rows (not a structural color that spans
    many rows). This distinguishes it from backgrounds, borders, and floors.
    """
    h, w = grid.shape
    start_row = max(0, h - _SCAN_ROWS)

    # For each color in the bottom rows, find:
    # 1. Which rows it appears in
    # 2. Its longest horizontal run
    color_row_presence: dict[int, set[int]] = {}
    color_best_run: dict[int, tuple[int, int, int]] = {}  # color -> (row, start_col, length)

    for y in range(start_row, h):
        runs = _find_horizontal_runs(grid[y, :])
        for color, start_col, length in runs:
            if length < _MIN_BAR_LENGTH:
                break
            # Track which rows this color appears in
            if color not in color_row_presence:
                color_row_presence[color] = set()
            color_row_presence[color].add(y)
            # Track best run per color
            if color not in color_best_run or length > color_best_run[color][2]:
                color_best_run[color] = (y, start_col, length)

    # Also check how many rows each color appears in across the FULL frame
    # (to filter out structural colors like borders/floors)
    full_frame_row_presence: dict[int, int] = {}
    for y in range(h):
        unique_colors = np.unique(grid[y, :])
        for c in unique_colors:
            c = int(c)
            full_frame_row_presence[c] = full_frame_row_presence.get(c, 0) + 1

    # Energy bar candidates: colors that appear in 2-4 adjacent bottom rows
    # AND don't span more than 8 rows in the full frame (not structural)
    best_run: tuple[int, int, int, int] | None = None  # (row, color, start_col, length)
    for color, rows in color_row_presence.items():
        # Skip colors that appear in too many rows overall (structural)
        if full_frame_row_presence.get(color, 0) > 10:
            continue
        # Skip full-width runs (borders)
        run_info = color_best_run[color]
        if run_info[2] >= w - 2:
            continue
        # Prefer colors in exactly 2-4 rows with long runs
        num_rows = len(rows)
        if num_rows < 2 or num_rows > 6:
            continue
        # Check rows are adjacent
        sorted_rows = sorted(rows)
        if sorted_rows[-1] - sorted_rows[0] != num_rows - 1:
            continue
        # Best = longest run among qualifying candidates
        if best_run is None or run_info[2] > best_run[3]:
            best_run = (run_info[0], color, run_info[1], run_info[2])

    if best_run is None:
        return None

    bar_row, bar_color, bar_start, bar_len = best_run

    # Find all adjacent rows with matching bar color and similar extent
    bar_rows = [bar_row]
    for dy in [-1, 1]:
        y = bar_row + dy
        while start_row <= y < h:
            row_runs = _find_horizontal_runs(grid[y, :])
            matching = [r for r in row_runs if r[0] == bar_color and r[2] >= _MIN_BAR_LENGTH]
            if matching:
                bar_rows.append(y)
                y += dy
            else:
                break
    bar_rows.sort()

    # Determine bar region extent (start and end columns across all bar rows)
    col_positions: list[int] = []
    for y in bar_rows:
        for x in range(w):
            if int(grid[y, x]) == bar_color:
                col_positions.append(x)
    if not col_positions:
        return None

    col_start = min(col_positions)
    col_end = max(col_positions) + 1

    # Detect life indicators: small same-sized blocks near the bar
    # Look for repeating patterns of a different color in the bar rows
    indicator_color = None
    indicator_positions: list[tuple[int, int]] = []

    # Scan the same rows for small blocks to the right of the bar
    for y in bar_rows[:1]:  # just check first bar row
        x = col_end
        while x < w:
            c = int(grid[y, x])
            if c != bar_color and c != int(grid[h - 1, 0]):  # not bar color, not border color
                # Check if it's a small block (2x2 or similar)
                block_w = 0
                while x + block_w < w and int(grid[y, x + block_w]) == c:
                    block_w += 1
                if 1 <= block_w <= 4:
                    if indicator_color is None:
                        indicator_color = c
                    if c == indicator_color:
                        indicator_positions.append((x, y))
                x += block_w
            else:
                x += 1

    return _BarConfig(
        rows=bar_rows,
        bar_color=bar_color,
        col_start=col_start,
        col_end=col_end,
        initial_length=bar_len,
        indicator_color=indicator_color,
        indicator_positions=indicator_positions,
    )


class UIDetector:
    """Stateful UI detector that calibrates on first frame and tracks changes.

    Usage:
        detector = UIDetector()
        ui = detector.detect(frame, prev_frame)  # call each step
    """

    def __init__(self) -> None:
        self._bar_config: _BarConfig | None = None
        self._calibrated: bool = False
        self._initial_bar_pixels: int = 0
        self._prev_bar_pixels: int = 0
        self._initial_lives: int = 0
        # Shape display region: (row_start, row_end, col_start, col_end)
        self._shape_region: tuple[int, int, int, int] | None = None
        # Step-based energy fallback when bar detection fails
        self._step_count: int = 0

    def detect(self, frame: Frame, prev_frame: Frame | None = None) -> UIState:
        grid = frame.grid
        self._step_count += 1

        # Calibrate on first call
        if not self._calibrated:
            self._bar_config = _detect_bar_config(grid)
            if self._bar_config is not None:
                self._initial_bar_pixels = self._count_bar_pixels(grid)
                self._prev_bar_pixels = self._initial_bar_pixels
                self._initial_lives = self._count_lives(grid)
            self._shape_region = self._detect_shape_region(grid)
            self._calibrated = True

        energy = self._detect_energy(grid)
        lives = self._count_lives(grid)
        ui_mask = self._build_ui_mask(grid)
        shape_hash = self._hash_shape_display(grid)

        return UIState(
            energy=energy,
            lives=lives,
            target_shape=None,
            ui_region_mask=ui_mask,
            shape_display_hash=shape_hash,
        )

    def _count_bar_pixels(self, grid: np.ndarray) -> int:
        """Count current bar pixels across all bar rows."""
        cfg = self._bar_config
        if cfg is None:
            return 0
        count = 0
        for y in cfg.rows:
            for x in range(grid.shape[1]):
                if int(grid[y, x]) == cfg.bar_color:
                    count += 1
        return count

    def _detect_energy(self, grid: np.ndarray) -> float | None:
        cfg = self._bar_config
        if cfg is None or self._initial_bar_pixels == 0:
            # Fallback: estimate energy from step count.
            # Use a cyclical estimate so we don't get stuck in permanent exploit mode.
            # Each "life" is ~FALLBACK_BUDGET steps; if we survive longer, assume
            # we got a new life and reset the counter.
            steps_in_life = self._step_count % _FALLBACK_ENERGY_BUDGET
            return max(0.05, 1.0 - steps_in_life / _FALLBACK_ENERGY_BUDGET)

        current = self._count_bar_pixels(grid)

        # Handle death/reset: if bar jumps UP significantly, it's a life reset.
        if current > self._prev_bar_pixels + 10:
            # Life was lost, bar reset. Update reference but don't change initial.
            pass

        self._prev_bar_pixels = current
        energy = current / self._initial_bar_pixels
        return max(0.0, min(1.0, energy))

    def _count_lives(self, grid: np.ndarray) -> int | None:
        cfg = self._bar_config
        if cfg is None or cfg.indicator_color is None:
            return None

        # Count how many indicator blocks still have the indicator color
        count = 0
        for x, y in cfg.indicator_positions:
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                if int(grid[y, x]) == cfg.indicator_color:
                    count += 1
        return count if count > 0 else None

    def _detect_shape_region(self, grid: np.ndarray) -> tuple[int, int, int, int] | None:
        """Find a small bordered display region in the bottom portion of the frame.

        Looks for a rectangular region to the LEFT of the energy bar that contains
        a distinct bordered pattern (different from the game area background).
        """
        cfg = self._bar_config
        if cfg is None:
            return None

        h, w = grid.shape
        bar_col_start = cfg.col_start

        # The shape display is typically to the left of the energy bar,
        # in the bottom rows. Scan for a rectangular bordered region.
        # Look in columns 0 to bar_col_start, rows from ~(h-15) to h
        if bar_col_start < 6:
            return None  # no room for a display left of the bar

        scan_top = max(0, h - 15)
        # Find the region: look for columns that have a consistent border color
        # forming a rectangle in the bottom-left
        region_right = min(bar_col_start, 14)  # don't go past col 14
        region_left = 0

        # Find top of the display: scan upward from bar rows looking for
        # the first row where the left columns aren't all background
        bg_vals, bg_counts = np.unique(grid[0, :], return_counts=True)
        bg_color = int(bg_vals[bg_counts.argmax()])

        region_top = scan_top
        for y in range(min(cfg.rows) - 1, scan_top - 1, -1):
            left_strip = grid[y, region_left:region_right]
            non_bg = np.sum(left_strip != bg_color)
            if non_bg > 3:
                region_top = y
            elif region_top < y:
                break

        region_bottom = min(cfg.rows) + len(cfg.rows)  # include bar rows

        if region_bottom - region_top < 4:
            return None

        return (region_top, region_bottom, region_left, region_right)

    def _hash_shape_display(self, grid: np.ndarray) -> int | None:
        """Hash the shape display region from the FULL grid (not UI-masked)."""
        if self._shape_region is None:
            return None
        r0, r1, c0, c1 = self._shape_region
        region = grid[r0:r1, c0:c1].copy()
        return xxhash.xxh64(region.tobytes()).intdigest()

    def _build_ui_mask(self, grid: np.ndarray) -> np.ndarray | None:
        h, w = grid.shape
        cfg = self._bar_config
        if cfg is None:
            # Fallback: mark bottom rows as UI to prevent UI flicker
            # from corrupting state hashes
            mask = np.zeros((h, w), dtype=bool)
            mask[h - _FALLBACK_UI_ROWS:, :] = True
            return mask

        mask = np.zeros((h, w), dtype=bool)
        min_row = min(cfg.rows)
        # Mark everything from the bar rows to the bottom as UI
        mask[min_row:, :] = True
        return mask

    def reset_energy(self) -> None:
        """Reset step-based energy counter (call on level complete or game reset)."""
        self._step_count = 0

    def reset(self) -> None:
        """Reset calibration (call on new game or level with different UI)."""
        self._bar_config = None
        self._calibrated = False
        self._initial_bar_pixels = 0
        self._prev_bar_pixels = 0
        self._initial_lives = 0
        self._shape_region = None
        self._step_count = 0


# Module-level convenience for simple usage
def detect_ui(frame: Frame, prev_frame: Frame | None = None) -> UIState:
    """Stateless detect (no calibration). Use UIDetector for tracking."""
    return UIState(
        energy=None,
        lives=None,
        target_shape=None,
        ui_region_mask=None,
        shape_display_hash=None,
    )
