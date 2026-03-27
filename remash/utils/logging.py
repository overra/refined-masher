"""Lightweight episode logging for analysis."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcengine import GameAction

    from remash.memory.state_graph import StateGraph

logger = logging.getLogger("remash")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(level)


@dataclass
class StepLog:
    step: int
    action: str
    action_value: int
    state_hash: str  # hex string for readability
    next_state_hash: str
    pixels_changed: int
    game_state: str
    levels_completed: int


class EpisodeLogger:
    def __init__(self, game_id: str, log_dir: Path | None = None) -> None:
        self.game_id = game_id
        self.log_dir = log_dir or Path("logs")
        self._step_logs: list[StepLog] = []
        self._level_results: list[dict] = []

    def log_step(
        self,
        step: int,
        action: GameAction,
        state_hash: int,
        next_state_hash: int,
        pixels_changed: int,
        game_state: str,
        levels_completed: int,
    ) -> None:
        log = StepLog(
            step=step,
            action=action.name,
            action_value=action.value,
            state_hash=f"{state_hash:#018x}",
            next_state_hash=f"{next_state_hash:#018x}",
            pixels_changed=pixels_changed,
            game_state=game_state,
            levels_completed=levels_completed,
        )
        self._step_logs.append(log)
        logger.debug(
            "step=%d action=%s pixels=%d state=%s",
            step, action.name, pixels_changed, game_state,
        )

    def log_level_complete(self, level_num: int, steps: int, graph_stats: dict) -> None:
        result = {
            "level": level_num,
            "steps": steps,
            "graph": graph_stats,
        }
        self._level_results.append(result)
        logger.info("Level %d complete in %d steps | graph: %s", level_num, steps, graph_stats)

    def log_game_over(self, levels_completed: int, total_steps: int) -> None:
        logger.info("Game over: %d levels in %d steps", levels_completed, total_steps)

    def save(self) -> Path | None:
        if not self._step_logs:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{self.game_id}.jsonl"
        with open(path, "w") as f:
            for log in self._step_logs:
                f.write(json.dumps(asdict(log)) + "\n")
        return path
