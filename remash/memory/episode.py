"""Within-episode trajectory buffer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arcengine import GameAction

if TYPE_CHECKING:
    from remash.perception.frame import Frame, FrameDiff
    from remash.perception.objects import GridObject, ObjectDelta
    from remash.perception.ui import UIState


@dataclass(slots=True)
class Step:
    step_num: int
    frame: Frame
    action: GameAction
    next_frame: Frame
    diff: FrameDiff
    objects: list[GridObject]
    object_deltas: list[ObjectDelta]
    ui_state: UIState | None
    state_hash: int
    next_state_hash: int


@dataclass(slots=True)
class ActionSummary:
    times_used: int = 0
    times_changed_frame: int = 0
    typical_num_changed_pixels: float = 0.0
    typical_object_movements: list[tuple[int, int]] = field(default_factory=list)


class EpisodeBuffer:
    def __init__(self) -> None:
        self._steps: list[Step] = []
        self._step_counter: int = 0

    def add_step(
        self,
        frame: Frame,
        action: GameAction,
        next_frame: Frame,
        diff: FrameDiff,
        objects: list[GridObject],
        object_deltas: list[ObjectDelta],
        ui_state: UIState | None,
    ) -> None:
        self._steps.append(Step(
            step_num=self._step_counter,
            frame=frame,
            action=action,
            next_frame=next_frame,
            diff=diff,
            objects=objects,
            object_deltas=object_deltas,
            ui_state=ui_state,
            state_hash=frame.hash(),
            next_state_hash=next_frame.hash(),
        ))
        self._step_counter += 1

    def get_recent(self, n: int) -> list[Step]:
        return self._steps[-n:]

    def get_action_effects(self) -> dict[GameAction, list[FrameDiff]]:
        effects: dict[GameAction, list[FrameDiff]] = defaultdict(list)
        for step in self._steps:
            effects[step.action].append(step.diff)
        return dict(effects)

    def get_action_effect_summary(self) -> dict[GameAction, ActionSummary]:
        summaries: dict[GameAction, ActionSummary] = {}
        for step in self._steps:
            if step.action not in summaries:
                summaries[step.action] = ActionSummary()
            s = summaries[step.action]
            s.times_used += 1
            if step.diff.num_changed > 0:
                s.times_changed_frame += 1
            # Running average of changed pixels
            s.typical_num_changed_pixels += (
                step.diff.num_changed - s.typical_num_changed_pixels
            ) / s.times_used
            # Collect object movements
            for delta in step.object_deltas:
                if delta.moved:
                    s.typical_object_movements.append(delta.moved)
        return summaries

    def get_trajectory(self) -> list[Step]:
        return list(self._steps)

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()
        self._step_counter = 0
