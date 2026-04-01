# ReMash Project

ARC-AGI-3 interactive reasoning agent. See `docs/initial-project-instructions.md` for full spec.

## Build & Run

```bash
uv sync                                                    # install deps
uv run python scripts/play.py --game ls20                  # single game
uv run python scripts/benchmark.py                         # Explorer baseline
uv run python scripts/benchmark.py --efe                   # ActorCritic + ensemble
uv run python scripts/benchmark.py --competition-mode      # multi-life (competition)
uv run python scripts/benchmark.py --efe --competition-mode # full pipeline
```

## Architecture

Model-based RL agent:
- **Perception**: CNN frame encoder → 8x8x8 spatial latent
- **World model**: Residual MLP ensemble (predicts Δz, uncertainty = disagreement)
- **Policy**: Actor-Critic with ensemble disagreement as intrinsic reward
- **State graph**: Exact-knowledge cache (known transitions have uncertainty=0)
- **Explorer**: Reactive heuristic fallback for first 8 steps + win-path execution

All modules communicate through ABCs in `remash/world_model/base.py` and `remash/policy/base.py`.

## Design Principles

- **Assume nothing about game mechanics.** The agent must discover everything
  from scratch. No hardcoded assumptions about energy bars, directional
  movement, toggle patterns, rooms, doorways, or UI layouts. The perception
  layer detects what's there. The world model learns what it does. The policy
  decides what to do about it.

- **Phase discipline.** When the current phase hits its ceiling across multiple
  games, move to the next phase. Don't iterate within a phase trying to squeeze
  performance on a single game. Game-specific heuristics are the overfitting trap
  at the engineering level. The architecture should generalize.

- **Neural model is not optional.** The reactive baseline is scaffolding. The
  neural world model + actor-critic policy is the agent. The Explorer is a
  fallback for the first 8 steps while the neural model boots.

- **Interfaces first.** Modules communicate through defined ABCs.
  Swapping world models requires zero changes to agent.py.

- **Aggressive frame diffing.** The diff between consecutive frames is the
  single most information-dense signal. Every module should use diffs.

- **Graph persists across deaths.** When the agent dies, it keeps all knowledge
  about which actions from which states are dead ends.

- **Log everything.** Every step is logged for debugging and analysis.

## Key API Quirks (arc-agi toolkit)

- The arc-agi package loads `.env.example` as fallback. Comment out placeholder keys.
- `GameAction.from_id(int)` not `GameAction(int)` for construction.
- `GameState.NOT_FINISHED` not "PLAYING".
- `FrameDataRaw.frame` is `list[ndarray]` (layers). Access `obs.frame[0]`.
- Frame dtype is `int8`, values 0-15 color indices. NOT RGB.
- Available actions vary per game. Read from `obs.available_actions`.
- Complex action (click): `env.step(GameAction.ACTION6, data={"x": 32, "y": 32})`.

## Competition Intel

- **Scoring**: RHAE = (human_actions / ai_actions)² per level, weighted by level index
  (level 5 counts 5x more than level 1). Mean across all games.
- **Submissions**: 5 per day (not 1). Can iterate faster than initially thought.
- **Compute**: RTX 5090 (H100s coming). torch confirmed available on Kaggle.
- **Action cap**: 2000 total actions per game across all lives.
- **Top approaches**: CNN+RL (12.58%), graph+ResNet (6.71%), FORGE notebook (0.39)
- **FORGE reward shaping**: +1.5 novel states, -0.1 revisits, +0.5 pixel changes

## Current Status

Ensemble world model + actor-critic policy built and integrated.
Explorer baseline: 5.1% competition-mode (9 levels, 7/25 games).
Actor-critic: 4.5% (needs speed optimization, but proves m0r0 capability).
Next: optimize handoff speed, tune reward shaping, submit to Kaggle.
