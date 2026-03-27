"""Abstract interface for world models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from arcengine import GameAction

from remash.perception.frame import FrameDiff


@dataclass(slots=True)
class WorldModelPrediction:
    predicted_next_hash: int | None
    confidence: float  # 0.0-1.0
    predicted_frame_changes: bool | None
    source: str  # "graph" | "neural" | "unknown"


class WorldModel(ABC):
    @abstractmethod
    def predict(self, state_hash: int, action: GameAction) -> WorldModelPrediction: ...

    @abstractmethod
    def update(self, state_hash: int, action: GameAction, next_state_hash: int, diff: FrameDiff) -> None: ...

    @abstractmethod
    def get_uncertainty(self, state_hash: int, action: GameAction) -> float:
        """0.0 = known, 1.0 = unknown."""
        ...

    @abstractmethod
    def get_frontier_actions(self, state_hash: int) -> list[tuple[GameAction, float]]:
        """Actions ranked by uncertainty (highest first)."""
        ...
