"""Cross-level knowledge. Phase 1: simple dict-based storage.

Stores action priors and responsive click target colors across levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from arcengine import GameAction

if TYPE_CHECKING:
    from remash.memory.episode import EpisodeBuffer
    from remash.memory.state_graph import StateGraph


class CrossLevelMemory:
    def __init__(self) -> None:
        self._action_priors: dict[GameAction, float] = {}
        self._levels_data: list[dict] = []
        # Click target priors: colors that were responsive across levels
        self._responsive_colors: dict[int, int] = {}  # color -> number of levels it was responsive in

    def on_level_complete(
        self,
        level_num: int,
        episode: EpisodeBuffer,
        graph: StateGraph,
        responsive_click_colors: set[int] | None = None,
    ) -> None:
        """Extract and store useful facts from a completed level."""
        summary = episode.get_action_effect_summary()

        for action, s in summary.items():
            if s.times_used > 0:
                effectiveness = s.times_changed_frame / s.times_used
                old = self._action_priors.get(action, 0.5)
                self._action_priors[action] = 0.7 * effectiveness + 0.3 * old

        # Record which click target colors were responsive
        if responsive_click_colors:
            for color in responsive_click_colors:
                self._responsive_colors[color] = self._responsive_colors.get(color, 0) + 1

        self._levels_data.append({
            "level_num": level_num,
            "steps": episode.total_steps,
            "graph_nodes": len(graph.nodes),
        })

    def get_action_priors(self) -> dict[GameAction, float]:
        return dict(self._action_priors)

    def get_responsive_click_colors(self) -> set[int]:
        """Colors that were responsive click targets in previous levels.

        Returns colors that were responsive in at least 1 prior level.
        The more levels a color appeared in, the more confident we are.
        """
        return set(self._responsive_colors.keys())

    def get_context_vector(self) -> np.ndarray | None:
        """Phase 1: returns None. Phase 4: returns LinOSS hidden state."""
        return None

    def reset_game(self) -> None:
        self._action_priors.clear()
        self._levels_data.clear()
        self._responsive_colors.clear()
