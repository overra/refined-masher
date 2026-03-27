"""Phase 1 world model: exact knowledge from the state graph.

Known transitions have uncertainty 0.0, unknown have 1.0.
"""

from __future__ import annotations

from arcengine import GameAction

from remash.memory.state_graph import StateGraph
from remash.perception.frame import FrameDiff
from remash.world_model.base import WorldModel, WorldModelPrediction


class GraphWorldModel(WorldModel):
    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph

    def predict(self, state_hash: int, action: GameAction) -> WorldModelPrediction:
        next_hash = self.graph.get_transition(state_hash, action)
        if next_hash is not None:
            frame_changes = action not in self.graph.nodes[state_hash].no_change
            return WorldModelPrediction(
                predicted_next_hash=next_hash,
                confidence=1.0,
                predicted_frame_changes=frame_changes,
                source="graph",
            )
        return WorldModelPrediction(
            predicted_next_hash=None,
            confidence=0.0,
            predicted_frame_changes=None,
            source="unknown",
        )

    def update(self, state_hash: int, action: GameAction, next_state_hash: int, diff: FrameDiff) -> None:
        self.graph.add_transition(state_hash, action, next_state_hash, diff_pixels=diff.num_changed)

    def get_uncertainty(self, state_hash: int, action: GameAction) -> float:
        if self.graph.get_transition(state_hash, action) is not None:
            return 0.0
        return 1.0

    def get_frontier_actions(self, state_hash: int) -> list[tuple[GameAction, float]]:
        """Untested actions first (uncertainty 1.0), no-change last (0.0)."""
        result: list[tuple[GameAction, float]] = []
        for action in self.graph.available_actions:
            uncertainty = self.get_uncertainty(state_hash, action)
            result.append((action, uncertainty))
        result.sort(key=lambda x: -x[1])
        return result
