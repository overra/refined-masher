"""Frame: wrapper around a 64x64 numpy array of color indices (0-15).

Based on FRAME_FORMAT.md findings:
- obs.frame is list[ndarray], typically length 1
- Each array is (64, 64) int8 with values 0-15
- Already color indices, no RGB conversion needed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xxhash

if TYPE_CHECKING:
    from arcengine.enums import FrameDataRaw


@dataclass(frozen=True, slots=True)
class CellChange:
    x: int
    y: int
    old_color: int
    new_color: int


@dataclass(slots=True)
class FrameDiff:
    changed_mask: np.ndarray  # (64, 64) bool
    num_changed: int
    changes: list[CellChange]


class Frame:
    __slots__ = ("grid", "_hash", "_game_hash", "_ui_mask")

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid  # (64, 64) uint8
        self._hash: int | None = None
        self._game_hash: int | None = None
        self._ui_mask: np.ndarray | None = None

    @classmethod
    def from_raw(cls, obs: FrameDataRaw) -> Frame:
        """Construct from ARC-AGI-3 FrameDataRaw observation."""
        layers = obs.frame
        if not layers:
            raise ValueError("FrameDataRaw.frame is empty")
        grid = layers[0]
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid, dtype=np.uint8)
        elif grid.dtype != np.uint8:
            grid = grid.astype(np.uint8)
        return cls(grid)

    def hash(self) -> int:
        """Fast deterministic hash of full grid contents."""
        if self._hash is None:
            self._hash = xxhash.xxh64(self.grid.tobytes()).intdigest()
        return self._hash

    def game_hash(self, ui_mask: np.ndarray | None = None) -> int:
        """Hash of game area only, excluding UI pixels.

        Critical: the energy bar changes every step, so hashing the full
        frame produces unique hashes even when the game position is identical.
        Masking UI pixels before hashing gives stable state identity.
        """
        if ui_mask is not None:
            self._ui_mask = ui_mask
            self._game_hash = None  # invalidate cache if mask changes

        if self._game_hash is None:
            if self._ui_mask is not None:
                masked = self.grid.copy()
                masked[self._ui_mask] = 0  # zero out UI pixels
                self._game_hash = xxhash.xxh64(masked.tobytes()).intdigest()
            else:
                self._game_hash = self.hash()
        return self._game_hash

    def diff(self, other: Frame) -> FrameDiff:
        """Element-wise comparison with another frame."""
        changed_mask = self.grid != other.grid
        num_changed = int(changed_mask.sum())

        changes: list[CellChange] = []
        if num_changed > 0:
            ys, xs = np.where(changed_mask)
            for x, y in zip(xs, ys):
                changes.append(CellChange(
                    x=int(x),
                    y=int(y),
                    old_color=int(other.grid[y, x]),
                    new_color=int(self.grid[y, x]),
                ))

        return FrameDiff(
            changed_mask=changed_mask,
            num_changed=num_changed,
            changes=changes,
        )

    def color_masks(self) -> dict[int, np.ndarray]:
        """Return {color_id: (64,64) bool mask} for each color present."""
        unique_colors = np.unique(self.grid)
        return {int(c): self.grid == c for c in unique_colors}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return self.hash() == other.hash()

    def __repr__(self) -> str:
        unique = np.unique(self.grid)
        return f"Frame(hash={self.hash():#018x}, colors={list(unique)})"
