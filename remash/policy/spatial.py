"""Spatial awareness: player detection, goal identification, directional guidance.

The "toddler heuristic": detect interesting objects, figure out where you are,
walk toward them. No game-specific knowledge required.

When a direct path is blocked, tracks which approaches have failed so the
explorer can find alternate corridors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arcengine import GameAction

if TYPE_CHECKING:
    from remash.perception.frame import Frame, FrameDiff
    from remash.perception.objects import GridObject, ObjectDelta

# Minimum movement to count as a real directional shift (in pixels)
_MIN_MOVEMENT = 3
# Steps needed before declaring calibration done
_CALIBRATION_STEPS = 4


@dataclass(slots=True)
class GoalCandidate:
    obj: GridObject
    salience: float  # higher = more interesting
    distance: float  # from player, if known


class SpatialTracker:
    """Tracks the player object, maps actions to directions, identifies goals."""

    def __init__(self) -> None:
        # Action→direction mapping: ACTION1 → (dx, dy) per step
        self.action_dirs: dict[GameAction, tuple[float, float]] = {}
        # Player identification
        self.player_colors: set[int] = set()
        self.player_centroid: tuple[float, float] | None = None
        # Calibration state
        self._calibrated: bool = False
        self._move_observations: list[tuple[GameAction, list[ObjectDelta]]] = []
        # Per-color movement tallies during calibration
        self._color_moved_on: dict[int, list[tuple[GameAction, float, float]]] = defaultdict(list)
        # Structural colors to exclude from goal detection
        self.structural_colors: set[int] = set()
        # Blocked approach tracking: maps goal centroid (rounded) to set of
        # blocked approach sides ("east", "west", "north", "south")
        self._blocked_sides: dict[tuple[int, int], set[str]] = defaultdict(set)
        # Cache player position per state hash for frontier scoring
        self.state_player_pos: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self.action_dirs.clear()
        self.player_colors.clear()
        self.player_centroid = None
        self._calibrated = False
        self._move_observations.clear()
        self._color_moved_on.clear()
        self.structural_colors.clear()
        self._blocked_sides.clear()
        self.state_player_pos.clear()

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def has_direction_info(self) -> bool:
        """True if we have at least one action→direction mapping."""
        return len(self.action_dirs) > 0

    def cache_state_position(self, state_hash: int) -> None:
        """Call each step so frontier scorer can look up player positions per state."""
        if self.player_centroid is not None:
            self.state_player_pos[state_hash] = self.player_centroid

    def on_step(
        self,
        action: GameAction,
        object_deltas: list[ObjectDelta],
        diff_pixels: int,
        objects: list[GridObject],
        bg_color: int,
    ) -> None:
        """Call after each step with the action taken and resulting object deltas."""
        # Identify structural colors from large objects (always useful)
        for obj in objects:
            if obj.area > 200:
                self.structural_colors.add(obj.color)
        self.structural_colors.add(bg_color)

        # Skip clicks and no-ops for direction mapping
        if action.is_complex() or diff_pixels < _MIN_MOVEMENT:
            if self._calibrated:
                self._update_player_position(objects)
            return

        # Record which objects moved and how
        for delta in object_deltas:
            if delta.moved and delta.prev_obj is not None:
                dx, dy = delta.moved
                if abs(dx) >= _MIN_MOVEMENT or abs(dy) >= _MIN_MOVEMENT:
                    self._color_moved_on[delta.obj.color].append((action, float(dx), float(dy)))

        self._move_observations.append((action, object_deltas))

        if self._calibrated:
            self._update_action_dirs_incremental(action, object_deltas)
            self._update_player_position(objects)
        elif len(self._move_observations) >= _CALIBRATION_STEPS:
            self._try_calibrate(objects)

    def _try_calibrate(self, objects: list[GridObject]) -> None:
        actions_per_color: dict[int, set[GameAction]] = defaultdict(set)
        avg_dir_per_color_action: dict[tuple[int, GameAction], tuple[float, float]] = {}

        for color, moves in self._color_moved_on.items():
            for action, dx, dy in moves:
                actions_per_color[color].add(action)
                key = (color, action)
                if key not in avg_dir_per_color_action:
                    avg_dir_per_color_action[key] = (dx, dy)
                else:
                    old_dx, old_dy = avg_dir_per_color_action[key]
                    avg_dir_per_color_action[key] = ((old_dx + dx) / 2, (old_dy + dy) / 2)

        if not actions_per_color:
            return

        max_actions = max(len(acts) for acts in actions_per_color.values())
        if max_actions < 1:
            return

        self.player_colors = {
            color for color, acts in actions_per_color.items()
            if len(acts) >= max_actions
        }

        for (color, action), (dx, dy) in avg_dir_per_color_action.items():
            if color in self.player_colors:
                if action not in self.action_dirs:
                    self.action_dirs[action] = (dx, dy)

        self._update_player_position(objects)
        self._calibrated = bool(self.action_dirs)

    def _update_action_dirs_incremental(
        self,
        action: GameAction,
        object_deltas: list[ObjectDelta],
    ) -> None:
        if action in self.action_dirs:
            return
        for delta in object_deltas:
            if delta.moved and delta.obj.color in self.player_colors:
                dx, dy = delta.moved
                if abs(dx) >= _MIN_MOVEMENT or abs(dy) >= _MIN_MOVEMENT:
                    self.action_dirs[action] = (float(dx), float(dy))
                    return

    def _update_player_position(self, objects: list[GridObject]) -> None:
        if not self.player_colors:
            return
        player_objs = [o for o in objects if o.color in self.player_colors and o.area < 100]
        if player_objs:
            total_area = sum(o.area for o in player_objs)
            cx = sum(o.centroid[0] * o.area for o in player_objs) / total_area
            cy = sum(o.centroid[1] * o.area for o in player_objs) / total_area
            self.player_centroid = (cx, cy)

    # --- Blocked approach tracking ---

    def record_blocked_approach(self, goal_pos: tuple[float, float]) -> None:
        """Record that the current approach direction to the goal is a wall.

        Determines the approach side from the relative position of the player
        to the goal. If the player is east of the goal and can't go left,
        the "east" approach is blocked.
        """
        if self.player_centroid is None:
            return
        px, py = self.player_centroid
        gx, gy = goal_pos
        goal_key = (round(gx), round(gy))

        dx = px - gx
        dy = py - gy
        # Determine which side we're approaching from
        if abs(dx) >= abs(dy):
            side = "east" if dx > 0 else "west"
        else:
            side = "south" if dy > 0 else "north"

        self._blocked_sides[goal_key].add(side)

    def is_approach_blocked(self, goal_pos: tuple[float, float]) -> bool:
        """True if the current approach direction to this goal is blocked."""
        if self.player_centroid is None:
            return False
        px, py = self.player_centroid
        gx, gy = goal_pos
        goal_key = (round(gx), round(gy))

        blocked = self._blocked_sides.get(goal_key, set())
        if not blocked:
            return False

        dx = px - gx
        dy = py - gy
        if abs(dx) >= abs(dy):
            side = "east" if dx > 0 else "west"
        else:
            side = "south" if dy > 0 else "north"

        return side in blocked

    def get_blocked_sides(self, goal_pos: tuple[float, float]) -> set[str]:
        goal_key = (round(goal_pos[0]), round(goal_pos[1]))
        return self._blocked_sides.get(goal_key, set())

    def get_preferred_approach_offset(self, goal_pos: tuple[float, float]) -> tuple[float, float]:
        """Return an (dx, dy) offset from the goal indicating the preferred approach direction.

        If east is blocked, prefer approaching from the west (dx < 0).
        Used by frontier scoring to prefer states on the unblocked side.
        """
        blocked = self.get_blocked_sides(goal_pos)
        if not blocked:
            return (0.0, 0.0)

        # Compute preferred direction: opposite of blocked sides
        dx, dy = 0.0, 0.0
        if "east" in blocked:
            dx -= 1.0  # prefer states west of goal
        if "west" in blocked:
            dx += 1.0  # prefer states east of goal
        if "south" in blocked:
            dy -= 1.0  # prefer states north of goal
        if "north" in blocked:
            dy += 1.0  # prefer states south of goal
        return (dx, dy)

    # --- Goal candidates ---

    def get_goal_candidates(
        self,
        objects: list[GridObject],
        ui_mask_rows: int | None = None,
    ) -> list[GoalCandidate]:
        if not objects:
            return []

        exclude_colors = self.structural_colors | self.player_colors

        color_area: dict[int, int] = {}
        for obj in objects:
            color_area[obj.color] = color_area.get(obj.color, 0) + obj.area
        max_area = max(color_area.values()) if color_area else 1

        candidates: list[GoalCandidate] = []
        for obj in objects:
            if obj.color in exclude_colors:
                continue
            if ui_mask_rows is not None and obj.centroid[1] > (64 - ui_mask_rows):
                continue
            if obj.area > 80:
                continue

            salience = 0.0
            if obj.area <= 6:
                salience += 40.0
            elif obj.area <= 20:
                salience += 25.0
            else:
                salience += 8.0

            rarity = 1.0 - (color_area.get(obj.color, 0) / max_area)
            salience += rarity * 20.0
            salience += 5.0

            dist = 999.0
            if self.player_centroid is not None:
                ddx = obj.centroid[0] - self.player_centroid[0]
                ddy = obj.centroid[1] - self.player_centroid[1]
                dist = (ddx * ddx + ddy * ddy) ** 0.5

            candidates.append(GoalCandidate(obj=obj, salience=salience, distance=dist))

        candidates.sort(key=lambda c: -c.salience)
        return candidates

    def get_action_toward(self, target: tuple[float, float]) -> GameAction | None:
        if not self.action_dirs or self.player_centroid is None:
            return None

        px, py = self.player_centroid
        tx, ty = target
        dx_want = tx - px
        dy_want = ty - py

        best_action: GameAction | None = None
        best_dot = -999.0

        for action, (adx, ady) in self.action_dirs.items():
            mag = (adx * adx + ady * ady) ** 0.5
            if mag < 0.1:
                continue
            dot = (dx_want * adx + dy_want * ady) / mag
            if dot > best_dot:
                best_dot = dot
                best_action = action

        if best_dot > 0:
            return best_action
        return None

    def format_status(self) -> str:
        if not self._calibrated:
            return f"calibrating({len(self._move_observations)}/{_CALIBRATION_STEPS})"
        parts = []
        if self.player_centroid:
            px, py = self.player_centroid
            parts.append(f"player=({px:.0f},{py:.0f})")
        dirs = []
        for a, (dx, dy) in sorted(self.action_dirs.items(), key=lambda x: x[0].value):
            dirs.append(f"{a.name[-1]}:({dx:+.0f},{dy:+.0f})")
        if dirs:
            parts.append(" ".join(dirs))
        if self._blocked_sides:
            for gk, sides in self._blocked_sides.items():
                parts.append(f"blocked@{gk}:{','.join(sorted(sides))}")
        return " ".join(parts)
