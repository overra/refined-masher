"""Directed graph of observed state transitions.

Nodes are frame hashes, edges are (action -> next_state_hash).
BFS for shortest paths and frontier discovery.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from arcengine import GameAction


@dataclass
class StateNode:
    hash: int
    transitions: dict[GameAction, int] = field(default_factory=dict)  # action -> next_state_hash
    transition_diffs: dict[GameAction, int] = field(default_factory=dict)  # action -> pixels changed
    no_change: set[GameAction] = field(default_factory=set)  # actions that didn't change frame
    is_win: bool = False
    visit_count: int = 0
    first_seen_step: int = 0
    max_diff_into: int = 0  # largest diff that produced a transition INTO this state


class StateGraph:
    def __init__(
        self,
        available_actions: list[GameAction] | None = None,
        no_change_threshold: int = 4,
    ) -> None:
        self.nodes: dict[int, StateNode] = {}
        self.available_actions: list[GameAction] = available_actions or []
        self.no_change_threshold = no_change_threshold
        self._step_counter: int = 0

    def ensure_node(self, state_hash: int) -> StateNode:
        """Create node if it doesn't exist. Returns the node."""
        if state_hash not in self.nodes:
            self.nodes[state_hash] = StateNode(
                hash=state_hash,
                first_seen_step=self._step_counter,
            )
        node = self.nodes[state_hash]
        node.visit_count += 1
        self._step_counter += 1
        return node

    def add_transition(
        self,
        state_hash: int,
        action: GameAction,
        next_state_hash: int,
        diff_pixels: int = 0,
    ) -> None:
        """Record a (state, action) -> next_state transition.

        diff_pixels below no_change_threshold are treated as no-ops
        (UI artifacts like cursor flicker, not real state changes).
        """
        self.ensure_node(state_hash)
        next_node = self.ensure_node(next_state_hash)
        node = self.nodes[state_hash]
        node.transitions[action] = next_state_hash
        node.transition_diffs[action] = diff_pixels
        if diff_pixels < self.no_change_threshold:
            node.no_change.add(action)
        # Track the largest diff that led into the destination state
        if diff_pixels > next_node.max_diff_into:
            next_node.max_diff_into = diff_pixels

    def get_transition(self, state_hash: int, action: GameAction) -> int | None:
        """Get next_state_hash for a known transition, or None."""
        node = self.nodes.get(state_hash)
        if node is None:
            return None
        return node.transitions.get(action)

    def get_untested_actions(self, state_hash: int) -> list[GameAction]:
        """Actions not yet tried from this state."""
        node = self.nodes.get(state_hash)
        if node is None:
            return list(self.available_actions)
        return [a for a in self.available_actions if a not in node.transitions]

    def get_changed_actions(self, state_hash: int) -> list[GameAction]:
        """Actions that produced a different frame from this state."""
        node = self.nodes.get(state_hash)
        if node is None:
            return []
        return [a for a in node.transitions if a not in node.no_change]

    def get_no_change_actions(self, state_hash: int) -> list[GameAction]:
        """Actions that produced no frame change from this state."""
        node = self.nodes.get(state_hash)
        if node is None:
            return []
        return list(node.no_change)

    def mark_win_state(self, state_hash: int) -> None:
        node = self.ensure_node(state_hash)
        node.is_win = True

    def shortest_path(self, from_hash: int, to_hash: int) -> list[GameAction] | None:
        """BFS shortest path from one state to another."""
        if from_hash == to_hash:
            return []
        if from_hash not in self.nodes or to_hash not in self.nodes:
            return None

        visited: set[int] = {from_hash}
        queue: deque[tuple[int, list[GameAction]]] = deque()
        queue.append((from_hash, []))

        while queue:
            current, path = queue.popleft()
            node = self.nodes[current]
            for action, next_hash in node.transitions.items():
                if next_hash in visited:
                    continue
                new_path = path + [action]
                if next_hash == to_hash:
                    return new_path
                visited.add(next_hash)
                queue.append((next_hash, new_path))

        return None

    def nearest_unexplored(self, from_hash: int) -> tuple[list[GameAction], int] | None:
        """BFS to find nearest state with untested actions.

        Returns (path_to_state, state_hash) or None if all states fully explored.
        """
        if from_hash not in self.nodes:
            return None

        # Check current state first
        if self.get_untested_actions(from_hash):
            return ([], from_hash)

        visited: set[int] = {from_hash}
        queue: deque[tuple[int, list[GameAction]]] = deque()
        queue.append((from_hash, []))

        while queue:
            current, path = queue.popleft()
            node = self.nodes[current]
            for action, next_hash in node.transitions.items():
                if next_hash in visited:
                    continue
                new_path = path + [action]
                if self.get_untested_actions(next_hash):
                    return (new_path, next_hash)
                visited.add(next_hash)
                queue.append((next_hash, new_path))

        return None

    def get_path_to_win(self, from_hash: int) -> list[GameAction] | None:
        """Find shortest path to any known win state."""
        win_states = [h for h, n in self.nodes.items() if n.is_win]
        if not win_states:
            return None

        best: list[GameAction] | None = None
        for win_hash in win_states:
            path = self.shortest_path(from_hash, win_hash)
            if path is not None and (best is None or len(path) < len(best)):
                best = path
        return best

    def get_doorway_frontiers(
        self,
        from_hash: int,
        doorway_min: int = 80,
        doorway_max: int = 500,
    ) -> list[tuple[list[GameAction], int]]:
        """Find frontier states that are reachable through doorway transitions.

        A doorway is a transition with doorway_min <= diff < doorway_max.
        (Excludes death animations which have diffs > 1000.)
        Returns (path, state_hash) pairs for frontier states behind doorways,
        sorted by number of untested actions (most untested first).
        """
        if from_hash not in self.nodes:
            return []

        visited: set[int] = {from_hash}
        queue: deque[tuple[int, list[GameAction], bool]] = deque()
        queue.append((from_hash, [], False))
        results: list[tuple[list[GameAction], int, int]] = []

        while queue:
            current, path, crossed = queue.popleft()
            node = self.nodes[current]

            for action, next_hash in node.transitions.items():
                if next_hash in visited:
                    continue
                visited.add(next_hash)
                new_path = path + [action]
                diff = node.transition_diffs.get(action, 0)
                is_doorway = doorway_min <= diff < doorway_max
                now_crossed = crossed or is_doorway

                untested = self.get_untested_actions(next_hash)
                if untested and now_crossed:
                    results.append((new_path, next_hash, len(untested)))

                queue.append((next_hash, new_path, now_crossed))

        results.sort(key=lambda r: (-r[2], len(r[0])))
        return [(r[0], r[1]) for r in results]

    def frontier_count(self) -> int:
        """Count states that have untested actions."""
        return sum(
            1 for n in self.nodes.values()
            if self.get_untested_actions(n.hash)
        )

    def get_stats(self) -> dict:
        total_nodes = len(self.nodes)
        total_transitions = sum(len(n.transitions) for n in self.nodes.values())
        total_no_change = sum(len(n.no_change) for n in self.nodes.values())
        win_states = sum(1 for n in self.nodes.values() if n.is_win)
        fully_explored = sum(
            1 for n in self.nodes.values()
            if not self.get_untested_actions(n.hash)
        )
        return {
            "nodes": total_nodes,
            "transitions": total_transitions,
            "no_change_transitions": total_no_change,
            "win_states": win_states,
            "fully_explored_nodes": fully_explored,
        }
