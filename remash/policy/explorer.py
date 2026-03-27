"""Phase 1: Graph-guided systematic exploration policy.

Priority order:
1. Win state known and reachable -> execute shortest path.
2. Productive toggle detection.
3. Spatial goal pursuit (suppressed when approach is blocked).
4. Energy-aware explore/exploit mode switching.
5. Untested actions from current state.
6. Navigate to best frontier (corridor discovery when goal is blocked).
7. Random fallback.

When the direct path to a goal is blocked by a wall, the explorer switches
to corridor discovery mode: it prioritizes exploring from high-diff states
(room transitions) and prefers frontiers on the opposite side of the goal
from the blocked approach.
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING

from arcengine import GameAction

from remash.policy.base import Policy
from remash.policy.spatial import SpatialTracker
from remash.utils.logging import logger

if TYPE_CHECKING:
    import numpy as np

    from remash.memory.cross_level import CrossLevelMemory
    from remash.memory.episode import EpisodeBuffer
    from remash.memory.state_graph import StateGraph
    from remash.perception.frame import Frame
    from remash.perception.objects import GridObject
    from remash.perception.ui import UIState
    from remash.world_model.base import WorldModel


_LOOP_WINDOW = 20

_ENERGY_EXPLORE_THRESHOLD = 0.70
_ENERGY_EXPLOIT_THRESHOLD = 0.30

_REVERSE_PROBES: dict[GameAction, GameAction] = {
    GameAction.ACTION1: GameAction.ACTION3,
    GameAction.ACTION3: GameAction.ACTION1,
    GameAction.ACTION2: GameAction.ACTION4,
    GameAction.ACTION4: GameAction.ACTION2,
}


def score_objects(objects: list[GridObject], frame: Frame) -> dict[int, float]:
    if not objects:
        return {}
    color_total_area: dict[int, int] = {}
    for obj in objects:
        color_total_area[obj.color] = color_total_area.get(obj.color, 0) + obj.area
    max_color_area = max(color_total_area.values()) if color_total_area else 1
    scores: dict[int, float] = {}
    for i, obj in enumerate(objects):
        score = 0.0
        if obj.area <= 6:
            score += 30.0
        elif obj.area <= 20:
            score += 20.0
        elif obj.area <= 80:
            score += 5.0
        else:
            score += 1.0
        rarity = 1.0 - (color_total_area.get(obj.color, 0) / max_color_area)
        score += rarity * 15.0
        scores[i] = score
    return scores


class ExplorerPolicy(Policy):
    def __init__(self, max_steps_per_level: int = 500) -> None:
        self.max_steps_per_level = max_steps_per_level
        self._level_steps: int = 0
        self._recent_actions: deque[tuple[int, GameAction]] = deque(maxlen=_LOOP_WINDOW)
        self._win_path: list[GameAction] | None = None
        self._win_path_idx: int = 0
        self._nav_path: list[GameAction] | None = None
        self._nav_path_idx: int = 0
        self._mode: str = "explore"
        self._pending_reverse: GameAction | None = None
        self._last_action: GameAction | None = None
        self.last_reason: str = ""
        self.spatial = SpatialTracker()
        self._state_interest: dict[int, float] = {}
        # Toggle detector
        self._toggle_action: GameAction | None = None
        self._toggle_return_action: GameAction | None = None
        self._toggle_state_a: int | None = None
        self._toggle_state_b: int | None = None
        self._toggle_shape_hashes: set[int] = set()
        self._toggle_stale_count: int = 0
        self._toggle_active: bool = False
        # Room commitment: after crossing a doorway, stay for N steps
        self._room_commit_remaining: int = 0
        self._room_entry_state: int | None = None
        self._last_state_hash: int | None = None
        self._last_diff_pixels: int = 0

    def on_level_start(self, level_num: int) -> None:
        self._level_steps = 0
        self._recent_actions.clear()
        self._win_path = None
        self._win_path_idx = 0
        self._nav_path = None
        self._nav_path_idx = 0
        self._mode = "explore"
        self._pending_reverse = None
        self._last_action = None
        self.last_reason = ""
        self.spatial.reset()
        self._state_interest.clear()
        self._toggle_action = None
        self._toggle_return_action = None
        self._toggle_state_a = None
        self._toggle_state_b = None
        self._toggle_shape_hashes.clear()
        self._toggle_stale_count = 0
        self._toggle_active = False
        self._room_commit_remaining = 0
        self._room_entry_state = None
        self._last_state_hash = None
        self._last_diff_pixels = 0

    def on_level_complete(self, level_num: int) -> None:
        pass

    def on_step_result(self, state_hash: int, diff_pixels: int) -> None:
        """Called by the agent after each step with the resulting state and diff."""
        self._last_state_hash = state_hash
        self._last_diff_pixels = diff_pixels

    def select_action(
        self,
        state_hash: int,
        frame: Frame,
        objects: list[GridObject],
        ui_state: UIState | None,
        world_model: WorldModel,
        episode: EpisodeBuffer,
        graph: StateGraph,
        cross_level: CrossLevelMemory,
        context: np.ndarray | None = None,
    ) -> GameAction:
        self._level_steps += 1
        energy = ui_state.energy if ui_state else None

        # Cache state interest and player position
        if state_hash not in self._state_interest:
            obj_scores = score_objects(objects, frame)
            self._state_interest[state_hash] = max(obj_scores.values()) if obj_scores else 0.0
        self.spatial.cache_state_position(state_hash)

        # Detect doorway crossing: did we just arrive through a large-diff transition?
        if (self._last_state_hash is not None
                and self._last_state_hash != state_hash
                and self._last_diff_pixels >= 80
                and self._room_commit_remaining <= 0):
            self._room_commit_remaining = 7
            self._room_entry_state = state_hash
            # Clear blocked approaches — new room might offer new angles
            goal_pos = self._get_primary_goal(objects, ui_state)
            if goal_pos and self.spatial.get_blocked_sides(goal_pos):
                logger.info(
                    "Doorway crossed (%dpx). Entering new room. Committing %d steps. Clearing blocked sides.",
                    self._last_diff_pixels, self._room_commit_remaining,
                )
                self.spatial._blocked_sides.clear()
            else:
                logger.info(
                    "Doorway crossed (%dpx). Committing %d steps in new room.",
                    self._last_diff_pixels, self._room_commit_remaining,
                )
            self._nav_path = None  # cancel any path back

        if self._room_commit_remaining > 0:
            self._room_commit_remaining -= 1

        # Update mode
        old_mode = self._mode
        self._update_mode(energy)
        if self._mode != old_mode:
            logger.info(
                "Policy mode: %s -> %s (energy=%.2f)",
                old_mode, self._mode, energy if energy is not None else -1.0,
            )
            if self._mode == "exploit":
                self._nav_path = None

        # --- Priority 1: Win path ---
        if self._win_path and self._win_path_idx < len(self._win_path):
            action = self._win_path[self._win_path_idx]
            self._win_path_idx += 1
            self._record(state_hash, action)
            self.last_reason = f"win-path({len(self._win_path) - self._win_path_idx} left)"
            return action

        # --- Priority 2: Reverse probe (suppressed during room commitment) ---
        if self._pending_reverse is not None and self._room_commit_remaining <= 0:
            action = self._pending_reverse
            self._pending_reverse = None
            if action in graph.available_actions:
                untested = graph.get_untested_actions(state_hash)
                if action in untested:
                    self._record(state_hash, action)
                    self.last_reason = "reverse-probe"
                    return action
        elif self._room_commit_remaining > 0:
            self._pending_reverse = None  # don't probe back through the doorway

        # --- Priority 3: Known win path ---
        win_path = graph.get_path_to_win(state_hash)
        if win_path:
            self._win_path = win_path
            self._win_path_idx = 0
            action = self._win_path[self._win_path_idx]
            self._win_path_idx += 1
            self._record(state_hash, action)
            self.last_reason = f"win-path({len(win_path)} steps)"
            return action

        # --- Priority 4: Productive toggle ---
        toggle_action = self._check_toggle(state_hash, ui_state, graph)
        if toggle_action is not None:
            self._record(state_hash, toggle_action)
            return toggle_action

        # --- Priority 5: Spatial goal pursuit (only if approach not blocked) ---
        goal_action = self._spatial_goal_action(objects, graph, state_hash, ui_state)
        if goal_action is not None:
            self._record(state_hash, goal_action)
            return goal_action

        # --- Priority 6: Room commitment — stay and explore locally ---
        if self._room_commit_remaining > 0:
            untested = graph.get_untested_actions(state_hash)
            if untested:
                action = self._pick_untested(untested, state_hash, cross_level)
                self._record(state_hash, action)
                self.last_reason = f"room-explore({self._room_commit_remaining} left, {len(untested)} untested)"
                return action
            # All actions tested here — navigate to nearest frontier within this room
            local_frontier = graph.nearest_unexplored(state_hash)
            if local_frontier:
                path, target_hash = local_frontier
                if path and len(path) <= 4:
                    self._nav_path = path
                    self._nav_path_idx = 0
                    action = self._nav_path[self._nav_path_idx]
                    self._nav_path_idx += 1
                    self._record(state_hash, action)
                    self.last_reason = f"room-nav({self._room_commit_remaining} left, dist={len(path)})"
                    return action

        # --- Priority 7: Exploit mode ---
        if self._mode == "exploit":
            return self._exploit_action(state_hash, graph, cross_level, objects, ui_state)

        # --- Priority 8: Follow nav path ---
        if self._nav_path and self._nav_path_idx < len(self._nav_path):
            action = self._nav_path[self._nav_path_idx]
            self._nav_path_idx += 1
            self._record(state_hash, action)
            self.last_reason = f"nav({len(self._nav_path) - self._nav_path_idx} left)"
            return action
        self._nav_path = None

        # --- Priority 9: Untested actions ---
        untested = graph.get_untested_actions(state_hash)
        node = graph.nodes.get(state_hash)
        tested_count = len(node.transitions) if node else 0

        if untested:
            if tested_count > 0 or self._level_steps % 3 != 0:
                action = self._pick_untested(untested, state_hash, cross_level)
                self._record(state_hash, action)
                label = "consolidate" if tested_count > 0 else "discover"
                self.last_reason = f"untested-{label}({len(untested)} left)"
                return action

        # --- Priority 10: Deep frontier (doorway exploration) ---
        # When goal is blocked and local frontier is depleted, find paths through doorways
        goal_info = self._get_primary_goal(objects, ui_state)
        goal_blocked = goal_info is not None and bool(self.spatial.get_blocked_sides(goal_info))
        frontier = graph.frontier_count()

        if goal_blocked or frontier < 10:
            doorway_targets = graph.get_doorway_frontiers(state_hash)
            if doorway_targets:
                path, target_hash = doorway_targets[0]
                self._nav_path = path
                self._nav_path_idx = 0
                action = self._nav_path[self._nav_path_idx]
                self._nav_path_idx += 1
                self._record(state_hash, action)
                n_behind = len(doorway_targets)
                self.last_reason = f"doorway-frontier(dist={len(path)},{n_behind} behind doors)"
                return action

        # --- Priority 11: Standard frontier ---
        max_nav_dist = None
        if self._mode == "mixed" and energy is not None:
            max_nav_dist = max(2, int(energy * 20))

        target = self._find_best_frontier(state_hash, graph, max_nav_dist, energy, goal_info)
        if target:
            path, target_hash = target
            if path:
                self._nav_path = path
                self._nav_path_idx = 0
                action = self._nav_path[self._nav_path_idx]
                self._nav_path_idx += 1
                self._record(state_hash, action)
                interest = self._state_interest.get(target_hash, 0)
                self.last_reason = f"nav-frontier(dist={len(path)},int={interest:.0f})"
                return action
            elif untested:
                action = self._pick_untested(untested, state_hash, cross_level)
                self._record(state_hash, action)
                self.last_reason = f"at-frontier({len(untested)} untested)"
                return action

        if untested:
            action = self._pick_untested(untested, state_hash, cross_level)
            self._record(state_hash, action)
            self.last_reason = f"untested-fallback({len(untested)} left)"
            return action

        # --- Priority 10: Loop avoidance ---
        changed = graph.get_changed_actions(state_hash)
        if changed:
            action = self._avoid_loops(changed, state_hash)
            self._record(state_hash, action)
            self.last_reason = "loop-avoid"
            return action

        action = random.choice(graph.available_actions)
        self._record(state_hash, action)
        self.last_reason = "random"
        return action

    # --- Toggle detection ---

    def _check_toggle(self, state_hash: int, ui_state: UIState | None, graph: StateGraph | None = None) -> GameAction | None:
        shape_hash = ui_state.shape_display_hash if ui_state else None

        if self._toggle_active:
            if shape_hash is not None:
                if shape_hash not in self._toggle_shape_hashes:
                    self._toggle_shape_hashes.add(shape_hash)
                    self._toggle_stale_count = 0
                    logger.info("Toggle: new shape %#018x (total: %d)", shape_hash, len(self._toggle_shape_hashes))
                else:
                    self._toggle_stale_count += 1

            if self._toggle_stale_count > 6:
                logger.info("Toggle: exhausted after %d unique shapes", len(self._toggle_shape_hashes))
                self._toggle_active = False
                return None

            # Verify action actually leads to the OTHER state (not a self-loop)
            if state_hash == self._toggle_state_a and self._toggle_action:
                if graph:
                    dest = graph.get_transition(state_hash, self._toggle_action)
                    if dest is not None and dest != self._toggle_state_b:
                        self._toggle_active = False
                        return None
                self.last_reason = f"toggle→B(shapes={len(self._toggle_shape_hashes)},stale={self._toggle_stale_count})"
                return self._toggle_action
            elif state_hash == self._toggle_state_b and self._toggle_return_action:
                if graph:
                    dest = graph.get_transition(state_hash, self._toggle_return_action)
                    if dest is not None and dest != self._toggle_state_a:
                        self._toggle_active = False
                        return None
                self.last_reason = f"toggle→A(shapes={len(self._toggle_shape_hashes)},stale={self._toggle_stale_count})"
                return self._toggle_return_action
            else:
                self._toggle_active = False
                return None

        # Check if we should START a toggle
        if len(self._recent_actions) < 4 or shape_hash is None:
            return None

        recent = list(self._recent_actions)[-4:]
        states = [s for s, _ in recent]
        unique_states = set(states)

        # Need exactly 2 DIFFERENT states strictly alternating
        if len(unique_states) != 2:
            return None
        # Verify strict alternation (A-B-A-B), which also excludes self-loops
        if not (states[0] != states[1] and states[1] != states[2] and states[2] != states[3]):
            return None

        state_a, state_b = sorted(unique_states)

        self._toggle_active = True
        self._toggle_state_a = state_a
        self._toggle_state_b = state_b
        self._toggle_shape_hashes = {shape_hash}
        self._toggle_stale_count = 0

        for s, a in recent:
            if s == state_a:
                self._toggle_action = a
            elif s == state_b:
                self._toggle_return_action = a

        logger.info("Toggle: started between %#06x and %#06x", state_a & 0xFFFF, state_b & 0xFFFF)

        if state_hash == state_a and self._toggle_action:
            self.last_reason = "toggle-start→B"
            return self._toggle_action
        elif state_hash == state_b and self._toggle_return_action:
            self.last_reason = "toggle-start→A"
            return self._toggle_return_action
        return None

    # --- Mode management ---

    def _update_mode(self, energy: float | None) -> None:
        if energy is None:
            self._mode = "explore"
            return
        if energy >= _ENERGY_EXPLORE_THRESHOLD:
            self._mode = "explore"
        elif energy <= _ENERGY_EXPLOIT_THRESHOLD:
            self._mode = "exploit"
        else:
            self._mode = "mixed"

    def _is_oscillating(self, state_hash: int) -> bool:
        if len(self._recent_actions) < 6:
            return False
        recent = list(self._recent_actions)[-6:]
        states = [s for s, _ in recent]
        return len(set(states)) <= 2

    # --- Spatial goal pursuit ---

    def _get_primary_goal(
        self,
        objects: list[GridObject],
        ui_state: UIState | None,
    ) -> tuple[float, float] | None:
        """Return the position of the best goal candidate, or None."""
        if not self.spatial.calibrated:
            return None
        ui_rows = 10 if ui_state and ui_state.ui_region_mask is not None else None
        goals = self.spatial.get_goal_candidates(objects, ui_rows)
        if goals and goals[0].salience >= 30.0:
            return goals[0].obj.centroid
        return None

    def _spatial_goal_action(
        self,
        objects: list[GridObject],
        graph: StateGraph,
        state_hash: int,
        ui_state: UIState | None,
    ) -> GameAction | None:
        if not self.spatial.calibrated:
            return None
        if self._level_steps % 3 == 0:
            return None
        if self._is_oscillating(state_hash):
            return None

        ui_rows = 10 if ui_state and ui_state.ui_region_mask is not None else None
        goals = self.spatial.get_goal_candidates(objects, ui_rows)
        if not goals:
            return None

        best_goal = goals[0]
        if best_goal.salience < 30.0:
            return None

        target = best_goal.obj.centroid

        # Check if this approach direction is blocked
        if self.spatial.is_approach_blocked(target):
            blocked = self.spatial.get_blocked_sides(target)
            self.last_reason = f"goal-blocked({','.join(sorted(blocked))})"
            # Don't return an action — fall through to corridor discovery
            return None

        action = self.spatial.get_action_toward(target)
        if action is None:
            return None

        # Check if this action is a known no-op from this state
        node = graph.nodes.get(state_hash)
        if node and action in node.no_change:
            # This direction is a wall. Record it as blocked.
            self.spatial.record_blocked_approach(target)
            blocked = self.spatial.get_blocked_sides(target)
            logger.info(
                "Goal at (%.0f,%.0f) blocked from %s — switching to corridor discovery",
                target[0], target[1], ",".join(sorted(blocked)),
            )
            return None

        self.last_reason = (
            f"goal→({target[0]:.0f},{target[1]:.0f})"
            f" c{best_goal.obj.color} sal={best_goal.salience:.0f}"
            f" d={best_goal.distance:.0f}"
        )
        return action

    # --- Exploit mode ---

    def _exploit_action(
        self,
        state_hash: int,
        graph: StateGraph,
        cross_level: CrossLevelMemory,
        objects: list[GridObject] | None = None,
        ui_state: UIState | None = None,
    ) -> GameAction:
        # Spatial goal pursuit in exploit (only if not blocked and not oscillating)
        if self.spatial.calibrated and objects and not self._is_oscillating(state_hash):
            ui_rows = 10 if ui_state and ui_state.ui_region_mask is not None else None
            goals = self.spatial.get_goal_candidates(objects, ui_rows)
            if goals and goals[0].salience >= 25.0:
                target = goals[0].obj.centroid
                if not self.spatial.is_approach_blocked(target):
                    action = self.spatial.get_action_toward(target)
                    node = graph.nodes.get(state_hash)
                    if action and not (node and action in node.no_change):
                        self._record(state_hash, action)
                        self.last_reason = f"exploit-goal→({target[0]:.0f},{target[1]:.0f}) d={goals[0].distance:.0f}"
                        return action

        changed = graph.get_changed_actions(state_hash)
        if changed:
            scored: list[tuple[GameAction, float]] = []
            for action in changed:
                next_hash = graph.get_transition(state_hash, action)
                if next_hash is None:
                    continue
                next_node = graph.nodes.get(next_hash)
                if next_node is None:
                    continue
                untested_count = len(graph.get_untested_actions(next_hash))
                visit_penalty = next_node.visit_count * 0.5
                interest = self._state_interest.get(next_hash, 0)
                diff_bonus = min(next_node.max_diff_into / 10.0, 15.0)
                score = untested_count * 10.0 + interest * 0.3 + diff_bonus - visit_penalty
                scored.append((action, score))

            if scored:
                scored.sort(key=lambda x: -x[1])
                action = scored[0][0]
                self._record(state_hash, action)
                self.last_reason = f"exploit(score={scored[0][1]:.0f})"
                return action

        untested = graph.get_untested_actions(state_hash)
        if untested:
            action = self._pick_untested(untested, state_hash, cross_level)
            self._record(state_hash, action)
            self.last_reason = "exploit-untested"
            return action

        action = self._avoid_loops(graph.available_actions, state_hash)
        self._record(state_hash, action)
        self.last_reason = "exploit-loop-avoid"
        return action

    # --- Frontier scoring ---

    def _find_best_frontier(
        self,
        from_hash: int,
        graph: StateGraph,
        max_dist: int | None,
        energy: float | None = None,
        goal_pos: tuple[float, float] | None = None,
    ) -> tuple[list[GameAction], int] | None:
        """BFS to find a state worth exploring, with corridor discovery.

        When a goal is blocked, boosts frontiers that are:
        - High-diff (room transitions, likely corridor connections)
        - On the opposite side of the goal from the blocked approach
        - Have many untested actions (corridor junctions)

        When no goal is blocked, falls back to standard diff+interest scoring.
        """
        if from_hash not in graph.nodes:
            return None

        # Check current state first
        untested_here = len(graph.get_untested_actions(from_hash))
        if untested_here > 0:
            tested_here = len(graph.nodes[from_hash].transitions)
            if tested_here > 0:
                return ([], from_hash)

        # Determine if we're in corridor discovery mode
        goal_blocked = False
        preferred_offset = (0.0, 0.0)
        if goal_pos is not None and self.spatial.get_blocked_sides(goal_pos):
            goal_blocked = True
            preferred_offset = self.spatial.get_preferred_approach_offset(goal_pos)

        # Distance penalty — relaxed when goal is blocked (willing to travel far)
        if goal_blocked:
            dist_penalty = 0.5  # very relaxed: explore distant corridors
        elif energy is not None and energy < 0.5:
            dist_penalty = 4.0
        else:
            dist_penalty = 1.5

        visited: set[int] = {from_hash}
        queue: deque[tuple[int, list[GameAction]]] = deque()
        queue.append((from_hash, []))
        candidates: list[tuple[float, list[GameAction], int]] = []

        while queue:
            current, path = queue.popleft()
            if max_dist is not None and len(path) > max_dist:
                break

            node = graph.nodes[current]
            for action, next_hash in node.transitions.items():
                if next_hash in visited:
                    continue
                new_path = path + [action]
                visited.add(next_hash)

                untested = graph.get_untested_actions(next_hash)
                if untested:
                    next_node = graph.nodes[next_hash]
                    tested = len(next_node.transitions)
                    interest = self._state_interest.get(next_hash, 0)
                    diff_score = min(next_node.max_diff_into / 8.0, 15.0)

                    score = diff_score + interest * 0.3
                    score += len(untested) * 3.0
                    if tested > 0:
                        score += 20.0
                    score -= len(new_path) * dist_penalty

                    # Corridor discovery bonus: when goal is blocked, boost states
                    # on the preferred side and near room transitions
                    if goal_blocked and goal_pos is not None:
                        state_pos = self.spatial.state_player_pos.get(next_hash)
                        if state_pos is not None:
                            gx, gy = goal_pos
                            sx, sy = state_pos
                            # Is this state on the preferred side of the goal?
                            dx_from_goal = sx - gx
                            dy_from_goal = sy - gy
                            alignment = (
                                dx_from_goal * preferred_offset[0]
                                + dy_from_goal * preferred_offset[1]
                            )
                            if alignment > 0:
                                score += 15.0  # on the right side
                        # Extra boost for high-diff states (room transitions)
                        if next_node.max_diff_into >= 80:
                            score += 20.0

                    candidates.append((score, new_path, next_hash))

                queue.append((next_hash, new_path))

        if not candidates:
            return None

        candidates.sort(key=lambda c: -c[0])
        return (candidates[0][1], candidates[0][2])

    # --- Helpers ---

    def _pick_untested(
        self,
        untested: list[GameAction],
        state_hash: int,
        cross_level: CrossLevelMemory,
    ) -> GameAction:
        priors = cross_level.get_action_priors()
        if priors:
            scored = [(a, priors.get(a, 0.0)) for a in untested]
            scored.sort(key=lambda x: -x[1])
            return scored[0][0]
        sorted_actions = sorted(untested, key=lambda a: a.value)
        offset = self._level_steps % len(sorted_actions)
        return sorted_actions[offset]

    def _avoid_loops(self, candidates: list[GameAction], state_hash: int) -> GameAction:
        recent_set = set(self._recent_actions)
        non_recent = [a for a in candidates if (state_hash, a) not in recent_set]
        if non_recent:
            return random.choice(non_recent)
        return random.choice(candidates)

    def _record(self, state_hash: int, action: GameAction) -> None:
        self._recent_actions.append((state_hash, action))
        reverse = _REVERSE_PROBES.get(action)
        if reverse is not None and self._last_action != reverse:
            self._pending_reverse = reverse
        else:
            self._pending_reverse = None
        self._last_action = action
