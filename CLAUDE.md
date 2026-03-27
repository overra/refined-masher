# ReMash Project

ARC-AGI-3 interactive reasoning agent. See `docs/initial-project-instructions.md` for full spec.

## Build & Run

```bash
uv sync                                     # install deps
uv run python scripts/play.py --game ls20   # run on a single game
uv run python scripts/benchmark.py          # run on all games
```

## Architecture

Phased architecture: Phase 1 (reactive baseline) → Phase 2 (neural world model) → Phase 3 (active inference policy).

All modules communicate through ABCs in `remash/world_model/base.py` and `remash/policy/base.py`.
Swapping `graph_model` for `neural_model` requires zero changes to `agent.py`.

## Design Principles

- **Interfaces first.** Modules communicate through defined ABCs.
  Swapping graph_model for neural_model requires zero changes to agent.py.

- **Aggressive frame diffing.** The diff between consecutive frames is the
  single most information-dense signal. Every module should use diffs.

- **Click action masking.** Never enumerate 4096 positions. Click only on
  detected object centroids + a few exploratory points.

- **Graph persists across level resets.** When the agent dies, it keeps
  all knowledge about which actions from which states are dead ends.

- **Log everything.** Every step is logged. This data feeds future neural
  training and enables offline failure analysis.

- **Phase discipline.** When the current phase hits its ceiling across multiple
  games, move to the next phase. Don't iterate within a phase trying to squeeze
  performance on a single game. Game-specific heuristics are the overfitting trap
  at the engineering level. The architecture should generalize.

## Key API Quirks (arc-agi toolkit)

- The arc-agi package loads `.env.example` as fallback. Comment out placeholder keys.
- `GameAction.from_id(int)` not `GameAction(int)` for construction.
- `GameState.NOT_FINISHED` not "PLAYING".
- `FrameDataRaw.frame` is `list[ndarray]` (layers). Access `obs.frame[0]`.
- Frame dtype is `int8`, values 0-15 color indices. NOT RGB.
- Available actions vary per game. Read from `obs.available_actions`.
- Complex action (click): `env.step(GameAction.ACTION6, data={"x": 32, "y": 32})`.

## Current Status

Phase 1 complete. Phase 2 (CfC neural world model) in progress.

Phase 1 delivered: frame perception, BFS flood-fill object detection, state graph
with UI-masked hashing, energy bar detection, spatial player/goal tracking,
explorer policy with energy awareness, blocked-approach detection, doorway
identification, toggle detection, and per-step diagnostic logging.

Phase 1 ceiling: the reactive baseline must visit every state individually.
It cannot generalize "ACTION1 moves the player up by 5px" across unseen states.
This caps exploration at ~24 unique states per life (~42 steps per energy bar).
