"""Main ReMash agent. Integrates all components."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arcengine import GameAction, GameState

from remash.memory.cross_level import CrossLevelMemory
from remash.memory.episode import EpisodeBuffer
from remash.memory.state_graph import StateGraph
from remash.perception.frame import Frame
from remash.perception.objects import detect_background_color, detect_objects, track_objects
from remash.perception.ui import UIDetector, UIState
from remash.utils.logging import EpisodeLogger, logger
from remash.world_model.base import WorldModel
from remash.world_model.graph_model import GraphWorldModel

try:
    from remash.world_model.neural_model import NeuralWorldModel
except ImportError:
    NeuralWorldModel = None  # torch/ncps not installed

if TYPE_CHECKING:
    from remash.perception.objects import GridObject
    from remash.policy.base import Policy


@dataclass
class GameResult:
    game_id: str
    levels_completed: int
    win_levels: int
    total_steps: int
    level_steps: list[int] = field(default_factory=list)
    graph_stats: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Simple completion ratio."""
        if self.win_levels == 0:
            return 0.0
        return self.levels_completed / self.win_levels


class ClickTargetManager:
    """Cycles through click targets, escalating from targeted to grid sampling.

    Strategy:
    1. Try previously responsive positions first (highest priority)
    2. Try untried object centroids ranked by salience (small, unusual color)
    3. After N non-responsive clicks, add grid sampling to fill gaps
    4. When a responsive region is found, explore nearby positions

    "Responsive" = diff >= RESPONSIVE_THRESHOLD (matches graph no-change threshold).
    Small diffs like countdown bar ticks (1-2px) don't count.
    """

    _NOOP_THRESHOLD = 4  # same as graph no_change_threshold
    _GRID_FALLBACK_AFTER = 10  # add grid targets after this many noops
    _GRID_STEP = 8

    def __init__(self) -> None:
        self._tried_global: set[tuple[int, int]] = set()
        self._tried_per_state: dict[int, set[tuple[int, int]]] = defaultdict(set)
        self._consecutive_noop: int = 0
        self._responsive_positions: list[tuple[int, int]] = []
        self._responsive_colors: set[int] = set()  # colors that responded (for cross-level)
        self._stale_positions: set[tuple[int, int]] = set()
        self._position_stale: dict[tuple[int, int], int] = {}
        self._last_target: tuple[int, int] | None = None
        self._last_target_color: int | None = None
        self._grid_targets_generated: bool = False
        self._grid_targets: list[tuple[int, int]] = []
        self._grid_idx: int = 0
        # Cross-level priors: colors known to be interactive from previous levels
        self._color_priors: set[int] = set()

    def report_result(self, pixels_changed: int) -> None:
        """Call after each click with the diff result."""
        if pixels_changed >= self._NOOP_THRESHOLD:
            self._consecutive_noop = 0
            if self._last_target is not None:
                if self._last_target not in self._responsive_positions:
                    self._responsive_positions.append(self._last_target)
                if self._last_target_color is not None:
                    self._responsive_colors.add(self._last_target_color)
                self._position_stale.pop(self._last_target, None)
            # Stage transition: massive change means the frame restructured.
            # Clear stale positions so previously-exhausted targets can be retried.
            if pixels_changed >= 200:
                self._stale_positions.clear()
                self._tried_global.clear()
                self._position_stale.clear()
        else:
            self._consecutive_noop += 1
            if self._consecutive_noop >= self._GRID_FALLBACK_AFTER and not self._grid_targets_generated:
                self._build_grid_targets()
            # Track stale responsive positions
            if self._last_target in self._responsive_positions:
                count = self._position_stale.get(self._last_target, 0) + 1
                self._position_stale[self._last_target] = count
                if count >= 2:
                    # Demote after just 2 noops — faster cycling through targets
                    self._responsive_positions.remove(self._last_target)
                    self._stale_positions.add(self._last_target)

    def set_color_priors(self, colors: set[int]) -> None:
        """Set cross-level color priors from previous levels."""
        self._color_priors = set(colors)

    def get_responsive_colors(self) -> set[int]:
        """Colors that were responsive in this level (for cross-level transfer)."""
        return set(self._responsive_colors)

    def pick_target(
        self,
        state_hash: int,
        objects: list[GridObject],
    ) -> tuple[int, int]:
        """Pick the best click target."""
        tried = self._tried_per_state[state_hash]

        # Priority 0: objects matching cross-level color priors (HIGHEST priority)
        if self._color_priors:
            for obj in sorted(objects, key=lambda o: o.area):
                if obj.color in self._color_priors:
                    target = (int(obj.centroid[0]), int(obj.centroid[1]))
                    if target not in self._tried_global:
                        self._commit(target, tried, obj.color)
                        return target

        # Priority 1: revisit known responsive positions (they work!)
        for pos in self._responsive_positions:
            if pos not in tried:
                self._commit(pos, tried)
                return pos

        # Priority 2: explore near responsive positions
        near = self._pick_near_responsive(tried)
        if near is not None:
            self._commit(near, tried)
            return near

        # Priority 3: untried object centroids, SMALL first (most likely interactive)
        for obj in sorted(objects, key=lambda o: o.area):
            target = (int(obj.centroid[0]), int(obj.centroid[1]))
            if target not in self._tried_global:
                self._commit(target, tried, obj.color)
                return target

        # Priority 4: untried bbox corners
        for obj in objects:
            if obj.area >= 4:
                for corner in [(obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3])]:
                    if corner not in self._tried_global:
                        self._commit(corner, tried)
                        return corner

        # Priority 5: grid sampling fallback
        while self._grid_idx < len(self._grid_targets):
            target = self._grid_targets[self._grid_idx]
            self._grid_idx += 1
            if target not in self._tried_global:
                self._commit(target, tried)
                return target

        # Exhausted: cycle responsive positions for this state
        if self._responsive_positions:
            target = self._responsive_positions[0]
            self._commit(target, tried)
            return target

        target = (32, 32)
        self._commit(target, tried)
        return target

    def _commit(self, target: tuple[int, int], tried: set[tuple[int, int]], color: int | None = None) -> None:
        tried.add(target)
        self._tried_global.add(target)
        self._last_target = target
        self._last_target_color = color

    def _build_grid_targets(self) -> None:
        """Generate grid targets, starting near detected objects then filling gaps."""
        self._grid_targets = []
        step = self._GRID_STEP
        for y in range(step // 2, 64, step):
            for x in range(step // 2, 64, step):
                self._grid_targets.append((x, y))
        self._grid_targets_generated = True
        self._grid_idx = 0

    def _pick_near_responsive(self, tried: set[tuple[int, int]]) -> tuple[int, int] | None:
        """Explore positions adjacent to known responsive ones."""
        for cx, cy in self._responsive_positions:
            for dx in range(-4, 8, 4):
                for dy in range(-4, 8, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = max(0, min(63, cx + dx))
                    ny = max(0, min(63, cy + dy))
                    if (nx, ny) not in self._tried_global:
                        return (nx, ny)
        return None

    def clear(self) -> None:
        """Reset for a new level. Color priors are preserved (cross-level)."""
        self._tried_global.clear()
        self._tried_per_state.clear()
        self._consecutive_noop = 0
        self._responsive_positions.clear()
        self._responsive_colors.clear()
        self._stale_positions.clear()
        self._position_stale.clear()
        self._last_target = None
        self._last_target_color = None
        self._grid_targets_generated = False
        self._grid_targets.clear()
        self._grid_idx = 0
        # NOTE: _color_priors intentionally NOT cleared — they're cross-level


class ReMashAgent:
    def __init__(self, policy: Policy, max_total_steps: int = 2000, use_neural: bool = False) -> None:
        self.policy = policy
        self.max_total_steps = max_total_steps
        self.use_neural = use_neural

    def play_game(self, env, game_id: str = "") -> GameResult:
        """Play a full game (all levels) using the given environment."""
        episode_logger = EpisodeLogger(game_id)

        # Reset environment
        obs = env.reset()
        frame = Frame.from_raw(obs)
        game_id = game_id or obs.game_id
        win_levels = obs.win_levels

        # Get available actions from the environment
        available_actions = [GameAction.from_id(a) for a in obs.available_actions]
        logger.info(
            "Starting game %s: %d levels, actions=%s",
            game_id, win_levels, [a.name for a in available_actions],
        )

        # Initialize components
        graph = StateGraph(available_actions=available_actions)
        if self.use_neural:
            world_model: WorldModel = NeuralWorldModel(graph)
            logger.info("Using neural world model (CfC ensemble)")
        else:
            world_model = GraphWorldModel(graph)
        cross_level = CrossLevelMemory()
        episode = EpisodeBuffer()
        click_mgr = ClickTargetManager()
        ui_detector = UIDetector()

        level_num = 0
        total_steps = 0
        level_steps: list[int] = []
        current_level_steps = 0
        prev_frame: Frame | None = None
        prev_objects: list[GridObject] = []
        prev_state_hash: int = 0

        self.policy.on_level_start(level_num)

        while total_steps < self.max_total_steps:
            # Perception
            objects = detect_objects(frame)
            ui_state = ui_detector.detect(frame, prev_frame)
            # Use game_hash (excluding UI) for state identity so the energy bar
            # doesn't make every frame unique
            state_hash = frame.game_hash(ui_state.ui_region_mask)
            graph.ensure_node(state_hash)

            # Cache frame for neural world model training
            if NeuralWorldModel is not None and isinstance(world_model, NeuralWorldModel):
                world_model.cache_frame(state_hash, frame.grid)

            # Check game state from observation
            # Detect level transition via WIN state OR levels_completed increment
            level_completed = (
                obs.state == GameState.WIN
                or obs.levels_completed > level_num
            )
            if level_completed:
                if prev_frame is not None:
                    graph.mark_win_state(prev_state_hash)
                # Pass responsive click colors to cross-level memory
                resp_colors = click_mgr.get_responsive_colors()
                cross_level.on_level_complete(
                    level_num, episode, graph,
                    responsive_click_colors=resp_colors,
                )
                self.policy.on_level_complete(level_num)
                episode_logger.log_level_complete(
                    level_num, current_level_steps, graph.get_stats(),
                )
                level_steps.append(current_level_steps)
                level_num = obs.levels_completed
                current_level_steps = 0
                episode.clear()
                click_mgr.clear()
                # Load cross-level color priors for the new level
                color_priors = cross_level.get_responsive_click_colors()
                click_mgr.set_color_priors(color_priors)
                self.policy.on_level_start(level_num)
                logger.info(
                    "Level %d complete! Now on level %d (click priors: %s)",
                    level_num - 1, level_num, sorted(color_priors),
                )

                # Check if all levels done
                if obs.levels_completed >= win_levels:
                    logger.info("All %d levels completed!", win_levels)
                    break

            if obs.state == GameState.GAME_OVER:
                # Game failed
                logger.info("Game over at level %d (step %d)", level_num, total_steps)
                episode_logger.log_game_over(obs.levels_completed, total_steps)
                level_steps.append(current_level_steps)
                break

            # Select action
            action = self.policy.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level,
            )

            prev_frame = frame
            prev_objects = objects
            prev_state_hash = state_hash

            # Execute action
            click_target: tuple[int, int] | None = None
            if action.is_complex():
                # EFE policy may have scored click candidates itself
                efe_target = getattr(self.policy, "selected_click_target", None)
                if efe_target is not None:
                    click_target = efe_target
                else:
                    click_target = click_mgr.pick_target(state_hash, objects)
                obs = env.step(action, data={"x": click_target[0], "y": click_target[1]})
            else:
                obs = env.step(action)

            total_steps += 1
            current_level_steps += 1
            frame = Frame.from_raw(obs)

            # Update memory
            diff = frame.diff(prev_frame)
            if click_target is not None:
                click_mgr.report_result(diff.num_changed)
            new_ui = ui_detector.detect(frame, prev_frame)
            new_objects = detect_objects(frame)
            new_deltas = track_objects(prev_objects, new_objects, diff)
            episode.add_step(
                prev_frame, action, frame, diff,
                prev_objects, new_deltas, ui_state,
            )
            new_state_hash = frame.game_hash(new_ui.ui_region_mask)
            # Cache new frame BEFORE update so the replay buffer can pair them
            if NeuralWorldModel is not None and isinstance(world_model, NeuralWorldModel):
                world_model.cache_frame(new_state_hash, frame.grid)
            if NeuralWorldModel is not None and isinstance(world_model, NeuralWorldModel):
                world_model.update(prev_state_hash, action, new_state_hash, diff, click_xy=click_target)
            else:
                world_model.update(prev_state_hash, action, new_state_hash, diff)

            # Feed spatial tracker and step result
            spatial = getattr(self.policy, "spatial", None)
            if spatial is not None:
                bg = detect_background_color(frame)
                spatial.on_step(action, new_deltas, diff.num_changed, new_objects, bg)
            if hasattr(self.policy, "on_step_result"):
                self.policy.on_step_result(new_state_hash, diff.num_changed)

            # Per-step terminal log
            h_from = f"{prev_state_hash:016x}"[:4]
            h_to = f"{new_state_hash:016x}"[:4]
            e_str = f"{new_ui.energy:.2f}" if new_ui and new_ui.energy is not None else "?.??"
            reason = getattr(self.policy, "last_reason", "")
            frontier = graph.frontier_count()
            spatial_str = spatial.format_status() if spatial else ""
            shape_str = ""
            if new_ui and new_ui.shape_display_hash is not None:
                shape_str = f" sh:{new_ui.shape_display_hash:016x}"[:8]
            nn_str = ""
            if NeuralWorldModel is not None and isinstance(world_model, NeuralWorldModel) and world_model.avg_loss > 0:
                nn_str = f" loss:{world_model.avg_loss:.4f}"
            print(
                f"Step {total_steps:3d} | #{h_from}→#{h_to}"
                f" | {action.name:<7s}"
                f" | {diff.num_changed:3d}px"
                f" | {len(graph.nodes):3d} states"
                f" | energy: {e_str}"
                f" | {self.policy._mode:<7s}"
                f" | frontier: {frontier:2d}"
                f" | {reason}"
                + (f"{shape_str}" if shape_str else "")
                + nn_str
                + (f" | {spatial_str}" if spatial_str else ""),
            )

            # File log
            episode_logger.log_step(
                total_steps, action, prev_state_hash, new_state_hash,
                diff.num_changed, obs.state.value, obs.levels_completed,
            )

        episode_logger.save()

        return GameResult(
            game_id=game_id,
            levels_completed=obs.levels_completed,
            win_levels=win_levels,
            total_steps=total_steps,
            level_steps=level_steps,
            graph_stats=graph.get_stats(),
        )
