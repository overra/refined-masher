"""Abstract interface for policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from arcengine import GameAction

if TYPE_CHECKING:
    import numpy as np

    from remash.memory.cross_level import CrossLevelMemory
    from remash.memory.episode import EpisodeBuffer
    from remash.memory.state_graph import StateGraph
    from remash.perception.frame import Frame
    from remash.perception.objects import GridObject
    from remash.perception.ui import UIState
    from remash.world_model.base import WorldModel


class Policy(ABC):
    @abstractmethod
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
    ) -> GameAction: ...

    def on_level_start(self, level_num: int) -> None:
        pass

    def on_level_complete(self, level_num: int) -> None:
        pass
