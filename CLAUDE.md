# ReMash Project

ARC-AGI-3 interactive reasoning agent. See `docs/initial-project-instructions.md` for full spec.

## Build & Run

```bash
uv sync                                                        # install deps (torch optional)
uv run python scripts/play.py --game ls20                      # single game (Explorer)
uv run python scripts/play.py --game ls20 --efe                # single game (ActorCritic+MPC)
uv run python scripts/benchmark.py                             # Explorer baseline (single-life)
uv run python scripts/benchmark.py --competition-mode          # Explorer (multi-life, 2000 cap)
uv run python scripts/benchmark.py --efe --competition-mode    # ActorCritic+MPC (competition)
uv run python scripts/benchmark.py --efe --competition-mode --pretrained  # with pretrained weights
uv run python scripts/benchmark.py --games ls20,vc33           # specific games only
uv run python scripts/pretrain.py --episodes 100               # pre-train encoder+dynamics
uv run python scripts/bench_neural.py                          # throughput benchmark
```

## Architecture

Model-based RL agent with MPC planning:

- **Perception**: 16-channel one-hot CNN encoder (64x64 grid → 8x8x8 spatial latent).
  L2-normalized latents prevent scale explosion. One-hot input means color permutation
  = channel permutation, enabling color-invariant pre-training.
- **World model**: Residual MLP ensemble (`ensemble_model.py`). 3 independent heads
  predict Δz (change in latent, not full z_{t+1}). Zero-init output = stable identity
  prior. Uncertainty = ensemble disagreement. EMA target encoder (τ=0.995) for stable
  training. Graph model runs underneath as exact cache (known transitions = 0 uncertainty).
- **Policy**: Actor-Critic with MPC lookahead (`actor_critic.py`). For each available
  action, imagines 4-step trajectory through the world model. Scores by novelty
  (ensemble disagreement) + change magnitude. Explorer fallback for first 8 steps
  (spatial calibration) and when MPC isn't clearly better.
- **State graph**: Directed graph of observed transitions. BFS shortest paths. Win-path
  execution. Persists across deaths in competition mode.
- **Explorer**: Reactive heuristic fallback (`explorer.py`). Spatial tracking, goal
  pursuit, energy-aware mode switching, stuck detection, toggle detection.
  Used during bootstrap and as MPC's default when the world model is untrained.
- **EFE dispatcher** (`efe.py`): Routes to ActorCritic when EnsembleWorldModel is
  available, Explorer otherwise. Exists for Kaggle notebook backward compatibility.

All modules communicate through ABCs in `remash/world_model/base.py` and `remash/policy/base.py`.

## Design Principles

- **Assume nothing about game mechanics.** The agent must discover everything
  from scratch. No hardcoded assumptions about energy bars, directional
  movement, toggle patterns, rooms, doorways, or UI layouts.

- **Phase discipline.** When the current approach hits its ceiling across
  multiple games, change the approach. Don't iterate within a phase trying
  to squeeze performance on a single game. Game-specific heuristics are the
  overfitting trap at the engineering level.

- **Neural model is not optional.** The reactive Explorer is scaffolding.
  The world model + MPC policy is the agent. Explorer is a fallback for
  the first 8 steps while the world model boots.

- **Pre-mortem before long runs.** Use Deep Think (Gemini, o3) to stress-test
  any training loop before running it for >10 minutes. The first pre-training
  run exploded to loss=300 due to three interacting bugs that Deep Think
  identified in minutes.

- **Interfaces first.** Modules communicate through defined ABCs.
  Swapping world models requires zero changes to agent.py.

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
- `play_game()` accepts `external_world_model` param for injecting shared models (pre-training).
- `play_game()` accepts `competition_mode=True` for multi-life play (GAME_OVER → RESET).

## Competition Intel

- **Scoring**: RHAE = (human_actions / ai_actions)² per level, weighted by level index
  (level 5 counts 5x more than level 1). Mean across all games. The squared penalty
  is brutal: 2x human actions = 25% score, 10x human = 1%.
- **Submissions**: 5 per day. Daily reset at UTC midnight.
- **Compute**: RTX 5090 (H100s coming). torch confirmed available on Kaggle.
- **Action cap**: 2000 total actions per game across all lives.
- **Top approaches**: CNN+RL (12.58% StochasticGoose), graph+ResNet (6.71% Blind Squirrel),
  FORGE notebook (0.39 public). Source: `github.com/DriesSmit/ARC3-solution`.
- **FORGE reward shaping**: +1.5 novel states, -0.1 revisits, +0.5 pixel changes.
- **Level weighting**: Later levels are worth exponentially more. Solving level 5
  efficiently is far more valuable than perfecting level 1.
- **Hidden state**: Whether games are fully observable is unresolved on the forums.
  Consider frame stacking (last 2 frames) as a hedge.

## Current Status

**World model**: Residual MLP ensemble with 16-channel one-hot input, L2-normalized
latents, EMA target encoder (τ=0.995). Predicts Δz (residual change). Trains online
from replay buffer every 4 steps. 7,804 FPS inference on CPU.

**Policy**: Actor-Critic with MPC lookahead (4-step trajectories). Combined reward:
extrinsic (level completion, large frame changes) + β * intrinsic (ensemble variance).
Explorer fallback for first 8 steps and as MPC default. Conservative override: MPC
only replaces Explorer when score differential is significant.

**Pre-training**: Pipeline built and tested (scripts/pretrain.py). 16-channel one-hot
input, L2-normalized latents, persistent optimizer + global replay buffer across
episodes. Training is stable (loss 0.0003 across 100 episodes). Results: neutral-to-
slightly-positive effect on scores (+0.4% average). Checkpoints at ep25/50/75/100.

**Benchmark scores** (competition-mode, 25 public games):
- Explorer only: ~4.5% (6-7 games, 8-9 levels)
- Actor-Critic + MPC: ~5.3% best run (avg 4.4%, range 3.0-5.3%)
- Stable floor: 3.0% (4 games always win: vc33, lp85, cd82, sp80)
- Actor-critic unique wins: m0r0, cn04 (75% win rate)

**Kaggle**: Position 88/259 with 0.17 RHAE (from old V3 submission running broken
EFE code). V11 submitted but likely similar. New submission with MPC pending.

**Next priorities**:
1. Reduce exploration variance (biggest source of score instability)
2. Solve later levels (cross-level transfer, level weighting favors depth)
3. Submit updated code to Kaggle with new dataset
4. Investigate frame stacking for hidden-state games
