"""Phase 3: Expected Free Energy (EFE) active inference policy.

For each action a, compute:
    G(a) = -epistemic_value(a) - pragmatic_value(a)

Select action by sampling from softmax over -G(a) with adaptive precision.

Epistemic value = world_model.get_uncertainty(state, a)
    High uncertainty → trying this action teaches the agent something.

Pragmatic value = predicted magnitude of change (big diff = important event).
    Known transitions: use recorded diff magnitude from graph.
    Unknown transitions: mild default (encourages exploration).

Precision (inverse temperature) scales with model confidence:
    Low loss → high precision → exploit known good actions.
    High loss → low precision → broad exploration.

Falls back to the graph-based explorer when the neural model has
insufficient data (< MIN_NEURAL_TRANSITIONS).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from arcengine import GameAction

from remash.policy.base import Policy
from remash.policy.explorer import ExplorerPolicy
from remash.policy.spatial import SpatialTracker
from remash.utils.logging import logger
from remash.world_model.neural_model import NeuralWorldModel

if TYPE_CHECKING:
    import numpy as np

    from remash.memory.cross_level import CrossLevelMemory
    from remash.memory.episode import EpisodeBuffer
    from remash.memory.state_graph import StateGraph
    from remash.perception.frame import Frame
    from remash.perception.objects import GridObject
    from remash.perception.ui import UIState
    from remash.world_model.base import WorldModel

# Minimum replay buffer size before trusting neural uncertainty
MIN_NEURAL_TRANSITIONS = 20

# Precision bounds
PRECISION_MIN = 0.5   # very exploratory
PRECISION_MAX = 10.0  # very exploitative
PRECISION_LOSS_SCALE = 0.05  # precision = PRECISION_LOSS_SCALE / (avg_loss + eps)

# Pragmatic value scaling
PRAGMATIC_DIFF_SCALE = 1.0 / 100.0  # normalize diff magnitude to ~0-1 range


class EFEPolicy(Policy):
    """Expected Free Energy policy with explorer fallback.

    For click-only games, generates candidates from object centroids and
    scores each by neural uncertainty. selected_click_target is read by
    the agent to override the ClickTargetManager.
    """

    def __init__(self) -> None:
        self._explorer = ExplorerPolicy()
        self.spatial = self._explorer.spatial  # share the spatial tracker
        self.last_reason: str = ""
        self._mode: str = "efe"  # for step log compat
        self._precision: float = PRECISION_MIN
        self._using_explorer: bool = True
        self._level_steps: int = 0
        # When set, the agent should use this click target instead of the ClickTargetManager
        self.selected_click_target: tuple[int, int] | None = None

    def on_level_start(self, level_num: int) -> None:
        self._explorer.on_level_start(level_num)
        self._level_steps = 0

    def on_level_complete(self, level_num: int) -> None:
        self._explorer.on_level_complete(level_num)

    def on_step_result(self, state_hash: int, diff_pixels: int) -> None:
        """Forwarded from agent — pass to explorer for doorway detection."""
        self._explorer.on_step_result(state_hash, diff_pixels)

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

        # Check if neural model has enough data
        neural = isinstance(world_model, NeuralWorldModel)
        has_enough_data = neural and len(world_model.replay) >= MIN_NEURAL_TRANSITIONS

        if not has_enough_data:
            # Fall back to explorer
            self._using_explorer = True
            self._mode = "explore"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"fallback:{self._explorer.last_reason}"
            return action

        # --- EFE action selection ---
        self._using_explorer = False

        # First check: does the explorer have a win path? Always defer to that.
        win_path = graph.get_path_to_win(state_hash)
        if win_path:
            self._mode = "efe-win"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"efe-win:{self._explorer.last_reason}"
            return action

        # Compute adaptive precision from model loss
        avg_loss = world_model.avg_loss if neural else 1.0
        self._precision = min(
            PRECISION_MAX,
            max(PRECISION_MIN, PRECISION_LOSS_SCALE / (avg_loss + 1e-6)),
        )

        # Compute G(a) for each available action
        available = graph.available_actions
        if not available:
            self.last_reason = "efe-no-actions"
            return GameAction.ACTION1

        # Click-only game: let the ClickTargetManager handle target selection
        # (it has grid fallback and tried-set tracking). EFE just selects ACTION6.
        self.selected_click_target = None
        is_click_only = len(available) == 1 and available[0] == GameAction.ACTION6
        if is_click_only:
            # The ClickTargetManager will pick the actual (x,y)
            self._mode = "efe-click"
            self.last_reason = "click-delegate"
            return GameAction.ACTION6

        g_values: dict[GameAction, float] = {}
        epistemic_values: dict[GameAction, float] = {}
        pragmatic_values: dict[GameAction, float] = {}

        node = graph.nodes.get(state_hash)

        for action in available:
            # --- Epistemic value ---
            # Direct: uncertainty of this transition
            direct_epistemic = world_model.get_uncertainty(state_hash, action)

            # Transitive: average uncertainty of actions from the destination state.
            # "Going somewhere with lots of unknowns teaches us things."
            transitive_epistemic = 0.0
            if node and action in node.transitions:
                dest_hash = node.transitions[action]
                dest_uncertainties = [
                    world_model.get_uncertainty(dest_hash, a)
                    for a in graph.available_actions
                ]
                transitive_epistemic = sum(dest_uncertainties) / len(dest_uncertainties) if dest_uncertainties else 0.0
                # Also boost destinations with untested actions in the graph
                untested_count = len(graph.get_untested_actions(dest_hash))
                transitive_epistemic += untested_count * 0.15

            epistemic = direct_epistemic + transitive_epistemic * 0.5

            # --- Pragmatic value ---
            pragmatic = 0.0
            if node and action in node.transition_diffs:
                diff_px = node.transition_diffs[action]
                pragmatic = diff_px * PRAGMATIC_DIFF_SCALE
                # Bonus for large diffs (doorways/room transitions)
                if diff_px >= 80:
                    pragmatic += 0.3
                # Penalize no-change actions
                if action in node.no_change:
                    pragmatic = -0.5
                # Visit penalty: reduce pragmatic value for heavily visited destinations
                if action in node.transitions:
                    dest_node = graph.nodes.get(node.transitions[action])
                    if dest_node:
                        pragmatic -= dest_node.visit_count * 0.03
            else:
                # Unknown transition: strong encouragement
                pragmatic = 0.3

            g = -epistemic - pragmatic
            g_values[action] = g
            epistemic_values[action] = epistemic
            pragmatic_values[action] = pragmatic

        # Softmax selection with precision
        action = self._softmax_sample(g_values, self._precision)

        # Energy-aware mode label
        energy = ui_state.energy if ui_state else None
        if energy is not None and energy < 0.3:
            self._mode = "efe-low"
        elif energy is not None and energy > 0.7:
            self._mode = "efe-hi"
        else:
            self._mode = "efe"

        # Build reason string
        ep = epistemic_values[action]
        pr = pragmatic_values[action]
        g = g_values[action]
        self.last_reason = (
            f"G={g:.2f}(ep={ep:.2f},pr={pr:.2f})"
            f" prec={self._precision:.1f}"
        )

        return action

    def _softmax_sample(
        self,
        g_values: dict[GameAction, float],
        precision: float,
    ) -> GameAction:
        """Sample action from softmax over -G(a) * precision."""
        actions = list(g_values.keys())
        neg_g = [-g_values[a] for a in actions]

        # Compute softmax probabilities
        max_val = max(neg_g)
        exp_vals = [math.exp(precision * (v - max_val)) for v in neg_g]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]

        # Sample
        r = random.random()
        cumulative = 0.0
        for action, p in zip(actions, probs):
            cumulative += p
            if r <= cumulative:
                return action
        return actions[-1]

    def _score_click_candidates(
        self,
        state_hash: int,
        objects: list[GridObject],
        world_model: NeuralWorldModel,
        graph: StateGraph,
        ui_state: UIState | None,
    ) -> GameAction:
        """For click-only games: generate click candidates and pick by EFE.

        Generates up to 10 candidates from object centroids and bbox corners.
        Scores each by neural uncertainty (epistemic) + object salience (pragmatic).
        """
        # Generate candidates from objects
        ui_rows = 10 if ui_state and ui_state.ui_region_mask is not None else None
        candidates: list[tuple[int, int]] = []
        saliences: list[float] = []

        for obj in sorted(objects, key=lambda o: o.area):
            if ui_rows is not None and obj.centroid[1] > (64 - ui_rows):
                continue
            cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
            if (cx, cy) not in candidates:
                candidates.append((cx, cy))
                saliences.append(obj.area)  # small objects = low area = high interest
            # Add bbox corners for larger objects
            if obj.area >= 4 and len(candidates) < 10:
                for corner in [(obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3])]:
                    if corner not in candidates:
                        candidates.append(corner)
                        saliences.append(obj.area)

        # Add grid samples to fill up to 10 candidates
        if len(candidates) < 10:
            for gy in range(4, 64, 8):
                for gx in range(4, 64, 8):
                    if len(candidates) >= 10:
                        break
                    if (gx, gy) not in candidates:
                        candidates.append((gx, gy))
                        saliences.append(200.0)  # large "area" = low salience = grid fallback

        if not candidates:
            candidates = [(32, 32)]
            saliences = [100.0]

        candidates = candidates[:10]
        saliences = saliences[:10]

        # Get neural uncertainties in a single batch
        uncertainties = world_model.get_click_uncertainties(state_hash, candidates)

        # Score: epistemic (neural uncertainty) + pragmatic (inverse object size)
        best_idx = 0
        best_score = -999.0
        for i, ((cx, cy), unc, area) in enumerate(zip(candidates, uncertainties, saliences)):
            # Smaller objects = more interesting targets
            pragmatic = max(0.0, 1.0 - area / 200.0)
            score = unc + pragmatic * 0.3
            if score > best_score:
                best_score = score
                best_idx = i

        best_xy = candidates[best_idx]
        best_unc = uncertainties[best_idx]
        self.selected_click_target = best_xy
        self._mode = "efe-click"
        self.last_reason = (
            f"click({best_xy[0]},{best_xy[1]})"
            f" unc={best_unc:.2f}"
            f" of {len(candidates)} candidates"
        )
        return GameAction.ACTION6
