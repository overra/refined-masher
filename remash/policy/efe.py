"""Policy dispatcher: routes to ActorCritic (with neural model) or Explorer (without).

The Kaggle notebook creates EFEPolicy(). This module ensures the right
policy runs depending on what's available:
- EnsembleWorldModel present → ActorCriticPolicy (model-based RL)
- GraphWorldModel only → ExplorerPolicy (reactive heuristics)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcengine import GameAction

from remash.policy.base import Policy
from remash.policy.explorer import ExplorerPolicy
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


class EFEPolicy(Policy):
    """Dispatcher: ActorCritic when neural model available, Explorer otherwise.

    Maintains backward compatibility with existing Kaggle notebook code
    that creates EFEPolicy() directly.
    """

    def __init__(self) -> None:
        self._explorer = ExplorerPolicy()
        self._actor_critic = None  # lazy init
        self._active_policy: Policy = self._explorer
        self.spatial = self._explorer.spatial
        self.last_reason: str = ""
        self._mode: str = "explore"
        self.selected_click_target: tuple[int, int] | None = None
        self._detected: bool = False

    def on_level_start(self, level_num: int) -> None:
        self._explorer.on_level_start(level_num)
        if self._actor_critic is not None:
            self._actor_critic.on_level_start(level_num)

    def on_level_complete(self, level_num: int) -> None:
        self._explorer.on_level_complete(level_num)
        if self._actor_critic is not None:
            self._actor_critic.on_level_complete(level_num)

    def on_step_result(self, state_hash: int, diff_pixels: int) -> None:
        self._explorer.on_step_result(state_hash, diff_pixels)
        if self._actor_critic is not None:
            self._actor_critic.on_step_result(state_hash, diff_pixels)

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
        # One-time detection of world model type
        if not self._detected:
            self._detected = True
            try:
                from remash.world_model.ensemble_model import EnsembleWorldModel
                if isinstance(world_model, EnsembleWorldModel):
                    from remash.policy.actor_critic import ActorCriticPolicy
                    self._actor_critic = ActorCriticPolicy()
                    self._actor_critic.spatial = self.spatial
                    self._actor_critic._explorer = self._explorer
                    logger.info("EFE dispatcher: using ActorCritic policy (ensemble model detected)")
                else:
                    logger.info("EFE dispatcher: using Explorer policy (no ensemble model)")
            except ImportError:
                logger.info("EFE dispatcher: using Explorer policy (torch not available)")

        # Route to the right policy
        if self._actor_critic is not None:
            action = self._actor_critic.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self._mode = self._actor_critic._mode
            self.last_reason = self._actor_critic.last_reason
            self.selected_click_target = self._actor_critic.selected_click_target
        else:
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self._mode = self._explorer._mode
            self.last_reason = self._explorer.last_reason
            self.selected_click_target = None

        return action
