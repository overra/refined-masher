# ReMash Project Instructions

## Project Context

ReMash (Refined Masher) is an agent for the ARC-AGI-3 interactive reasoning benchmark. ARC-AGI-3 is an interactive reasoning benchmark where agents must explore novel turn-based game environments, discover rules and win conditions with no instructions, build world models, and execute strategies efficiently. Key facts:

- Observations: 64x64 grids, 16 discrete colors
- Actions: 5 key actions + 1 click action with x,y coordinates + undo + reset
- Turn-based, not real-time (2000+ FPS with rendering off)
- No language, no text, no cultural symbols in environments
- Only Core Knowledge priors: objectness, basic geometry/topology, basic physics, agentness
- Scoring: squared action efficiency relative to human baseline, weighted toward later levels
- Each game has 6+ levels with progressive difficulty through compositional mechanics
- Random agents validated to win <1/10,000 on non-tutorial levels
- Evaluation: RTX 5090, 8 hours compute, no internet, open source required
- Current SOTA: 12.58% (CNN + RL from preview), frontier LLMs score <1%, humans score 100%

The agent processes visual input directly and acts without verbalization. No language model component. The architectural approach uses continuous-time neural networks (CfC for fast dynamics, LinOSS for slow cross-level memory), active inference with Expected Free Energy for exploration-exploitation, and a learned world model. These neural components are added in later phases. Phase 1 is a purely reactive baseline with zero learned parameters.

The competition is live on Kaggle. Milestones: June 30 and September 30, 2026. Final submissions November 2. All solutions must be open-sourced. The leaderboard currently shows top scores around 0.28-0.31 (28-31% RHAE) from custom code agents. Frontier LLMs without scaffolding score below 0.4%.

## Step Zero: Inspect the ARC-AGI-3 Toolkit Before Writing Any Code

**This is mandatory. Do not skip this. Do not guess the frame format.**

Before implementing any perception, memory, or agent code, install arc-agi, create a scratch script, instantiate an environment, and inspect the actual FrameDataRaw object. Determine:

1. What type is the observation? Print `type(obs)`, `dir(obs)`, and all accessible attributes.
2. How is the 64x64 grid represented? Is it a numpy array, an RGB image, a list of lists, integer color indices, or something else?
3. If it's RGB pixels, what are the exact RGB values for each of the 16 colors? You'll need a lookup table to map RGB to color index.
4. If it's already color indices, what range? 0-15? 0-16?
5. What does `obs.state` contain? What are the GameState enum values? (WIN, GAME_OVER, PLAYING, NOT_PLAYED, etc.)
6. What does `obs.levels_completed` contain?
7. What does `env.action_space` return? What are the exact GameAction enum values?
8. When a frame sequence (animation) is returned, how is it structured? Is it a list of frames?
9. What does `env.observation_space` return vs the return value of `env.step()`?
10. How does the click action work? How do you pass x,y coordinates? Check the `GameAction.ACTION6` and its `set_data()` method.

Run this discovery script first:

```python
import arc_agi
from arcengine import GameAction, GameState

arc = arc_agi.Arcade()
env = arc.make("ls20")
obs = env.reset()

print("=== Observation type ===")
print(type(obs))
print(dir(obs))
if hasattr(obs, '__dict__'):
    print(obs.__dict__)

print("\n=== Observation attributes ===")
for attr in dir(obs):
    if not attr.startswith('_'):
        try:
            val = getattr(obs, attr)
            print(f"  {attr}: {type(val)} = {repr(val)[:200]}")
        except Exception as e:
            print(f"  {attr}: ERROR {e}")

print("\n=== Action space ===")
print(env.action_space)
for a in GameAction:
    print(f"  {a.name}: {a.value}, is_simple={a.is_simple()}, is_complex={a.is_complex()}")

print("\n=== Take an action ===")
obs2 = env.step(GameAction.ACTION1)
print(type(obs2))
for attr in dir(obs2):
    if not attr.startswith('_'):
        try:
            val = getattr(obs2, attr)
            print(f"  {attr}: {type(val)} = {repr(val)[:200]}")
        except Exception as e:
            print(f"  {attr}: ERROR {e}")

print("\n=== Try complex action ===")
action = GameAction.ACTION6
action.set_data({"x": 32, "y": 32})
obs3 = env.step(action)
print(f"Complex action result state: {obs3.state if obs3 else 'None'}")
```

**Document the findings in a file called `FRAME_FORMAT.md` in the repo root before proceeding with implementation.** All perception code must be built on observed facts about the data format, not assumptions.

---

## Implementation Spec

### Repo Structure

```
remash/
├── README.md
├── FRAME_FORMAT.md               # Document toolkit format findings here
├── pyproject.toml                # uv/pip, python 3.11+
├── .env.example                  # ARC_API_KEY=your-key-here
│
├── remash/
│   ├── __init__.py
│   ├── agent.py                  # Main agent loop + ARC-AGI-3 integration
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── frame.py              # Frame representation, hashing, diffing
│   │   ├── objects.py            # Flood fill object detection
│   │   └── ui.py                 # UI element detection (energy bar, lives, level indicator)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── state_graph.py        # Directed state transition graph
│   │   ├── episode.py            # Within-episode trajectory buffer
│   │   └── cross_level.py        # Cross-level knowledge (stub for LinOSS)
│   ├── world_model/
│   │   ├── __init__.py
│   │   ├── base.py               # WorldModel ABC with predict/update/uncertainty interface
│   │   ├── graph_model.py        # Graph-based exact model (Phase 1)
│   │   └── neural_model.py       # CfC-based learned model (stub for Phase 2)
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── base.py               # Policy ABC with select_action interface
│   │   ├── explorer.py           # Phase 1: graph-guided systematic exploration
│   │   └── efe.py                # EFE active inference policy (stub for Phase 3)
│   ├── interoception/
│   │   ├── __init__.py
│   │   └── state.py              # Internal state module (stub for Phase 5)
│   └── utils/
│       ├── __init__.py
│       └── logging.py            # Lightweight episode logging for analysis
│
├── scripts/
│   ├── discover_format.py        # Step zero discovery script (above)
│   ├── play.py                   # Run agent on a single game
│   ├── benchmark.py              # Run agent on all public games, report scores
│   └── replay_viewer.py          # Simple terminal replay of recorded episodes
│
└── tests/
    ├── test_frame.py
    ├── test_objects.py
    ├── test_state_graph.py
    └── test_explorer.py
```

### Dependencies

```toml
[project]
name = "remash"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "arc-agi",           # ARC-AGI-3 toolkit
    "numpy",             # Frame operations
    "xxhash",            # Fast frame hashing (non-crypto, speed > security)
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
neural = ["torch", "ncps"]  # Future: CfC via ncps package
```

---

## Module Specs

### `remash/perception/frame.py`

Core frame representation and operations.

```python
"""
Frame: wrapper around a 64x64 numpy array with 16 discrete color values (0-15).

Key operations:
- Frame.from_raw(frame_data) -> Frame
    Construct from ARC-AGI-3 FrameDataRaw. Extract the 64x64 grid as np.uint8.
    Implementation depends on FRAME_FORMAT.md findings.

- Frame.hash() -> int
    Fast deterministic hash of the grid contents. Use xxhash.xxh64 on the
    raw bytes of the numpy array. This is the node ID in the state graph.

- Frame.diff(other: Frame) -> FrameDiff
    Element-wise comparison. Returns a FrameDiff object containing:
      - changed_mask: np.ndarray (64x64 bool) - which cells changed
      - num_changed: int - total cells changed
      - changes: list[CellChange] - (x, y, old_color, new_color) for each change

- Frame.color_masks() -> dict[int, np.ndarray]
    Returns {color_id: 64x64 bool mask} for each color present in the frame.
    Only include colors that actually appear (skip absent colors).

- Frame.grid: np.ndarray (64x64, uint8) - the raw grid data
"""
```

Implementation notes:
- The grid format depends on what the toolkit provides. If RGB, build a lookup
  table mapping RGB tuples to color indices 0-15. Determine this in step zero.
- xxhash is ~10x faster than hashlib for this size data.
  `xxhash.xxh64(grid.tobytes()).intdigest()`
- Frame objects should be lightweight. Don't precompute masks on construction.

### `remash/perception/objects.py`

Flood-fill based object detection on raw frames.

```python
"""
Detect objects as connected components of same-colored pixels.

- detect_objects(frame: Frame, ui_mask: np.ndarray | None = None) -> list[GridObject]
    Run connected-component labeling on the frame grid.
    Use scipy.ndimage.label or a custom BFS flood fill (4-connected).
    If ui_mask is provided, exclude those pixels from object detection.

    GridObject:
      - color: int (0-15)
      - pixels: set[tuple[int, int]] - (x, y) coordinates
      - bbox: tuple[int, int, int, int] - (x_min, y_min, x_max, y_max)
      - centroid: tuple[float, float]
      - area: int - number of pixels
      - shape_hash: int - hash of the pixel pattern relative to bbox origin
                          (for comparing shapes across positions/colors)

- track_objects(prev_objects: list[GridObject], curr_objects: list[GridObject],
                diff: FrameDiff) -> list[ObjectDelta]
    Match objects between frames using the diff.
    ObjectDelta:
      - obj: GridObject (current frame version)
      - prev_obj: GridObject | None (previous frame version, None if new)
      - moved: tuple[int, int] | None - (dx, dy) displacement
      - color_changed: bool
      - shape_changed: bool
      - is_new: bool
      - is_gone: bool

Notes:
- Background color: detect as most frequent color in frame borders (rows 0, 63,
  cols 0, 63) OR most frequent color overall.
- Small objects (1-4 pixels) may be cursors or indicators. Flag separately.
- shape_hash: subtract bbox origin from all pixel coordinates, then hash the
  relative coordinate set. This enables position-independent shape comparison.
- If scipy is too heavy a dep, implement BFS flood fill directly. It's trivial
  on a 64x64 grid.
"""
```

### `remash/perception/ui.py`

Detect and parse UI elements from the frame.

```python
"""
ARC-AGI-3 environments embed UI info directly in the grid frame.
Common UI elements observed in ls20:

- Energy bar: horizontal bar near the bottom that shrinks each step.
- Lives: small colored squares near the energy bar.
- Target display: boxed region showing a target shape.
- Step counter: visual element that decreases.

- detect_ui(frame: Frame, prev_frame: Frame | None) -> UIState
    UIState:
      - energy: float | None (0.0-1.0, None if not detected)
      - lives: int | None
      - target_shape: GridObject | None
      - ui_region_mask: np.ndarray (64x64 bool) - which pixels are UI

Strategy: on the first few frames, identify UI rows/columns by checking for
pixels that change predictably (energy bar) or never change (borders/chrome).
Cache the UI layout for the rest of the episode.

NOTE: UI layout varies between games. This module should be adaptive and
fail gracefully (return None for undetected elements).
"""
```

### `remash/memory/state_graph.py`

Directed graph of observed state transitions.

```python
"""
Exact record of all observed (state, action) -> next_state transitions.

- StateGraph:
    - add_transition(state_hash, action, next_state_hash, frame_changed: bool)
    - get_transition(state_hash, action) -> next_state_hash | None
    - get_untested_actions(state_hash) -> list[Action]
    - get_changed_actions(state_hash) -> list[Action]
    - get_no_change_actions(state_hash) -> list[Action]
    - shortest_path(from_hash, to_hash) -> list[Action] | None
        BFS shortest path.
    - nearest_unexplored(from_hash) -> tuple[list[Action], int] | None
        BFS to nearest state with untested actions.
    - mark_win_state(state_hash)
    - get_path_to_win(from_hash) -> list[Action] | None
    - get_stats() -> dict
    - ensure_node(state_hash)
        Create node if it doesn't exist.

Internal:
    nodes: dict[int, StateNode]
    StateNode:
      - hash: int
      - transitions: dict[Action, int]  # action -> next_state_hash
      - no_change: set[Action]
      - is_win: bool
      - visit_count: int
      - first_seen_step: int
"""
```

### `remash/memory/episode.py`

Within-episode trajectory buffer.

```python
"""
Stores the full trajectory for the current level attempt.

- EpisodeBuffer:
    - add_step(frame, action, next_frame, diff, objects, object_deltas, ui_state)
    - get_recent(n) -> list[Step]
    - get_action_effects() -> dict[Action, list[FrameDiff]]
    - get_action_effect_summary() -> dict[Action, ActionSummary]
        ActionSummary:
          - times_used: int
          - times_changed_frame: int
          - typical_num_changed_pixels: float
          - typical_object_movements: list[tuple[int, int]]
    - get_trajectory() -> list[Step]
    - clear()

    Step:
      - step_num, frame, action, next_frame, diff, objects,
        object_deltas, ui_state, state_hash, next_state_hash
"""
```

### `remash/memory/cross_level.py`

Cross-level knowledge. Phase 1 is a simple dict, later becomes LinOSS.

```python
"""
Phase 1: dict-based storage of discovered facts across levels.

- CrossLevelMemory:
    - on_level_complete(level_num, episode, graph)
        Extract and store: action_map, win_state_properties, mechanics.
    - get_action_priors() -> dict[Action, ActionPrior]
    - get_context_vector() -> np.ndarray | None
        Phase 1: returns None. Phase 4: returns LinOSS hidden state.
    - reset_game()
"""
```

### `remash/world_model/base.py`

Abstract interface for world models.

```python
"""
- WorldModel (ABC):
    - predict(state_hash, action) -> WorldModelPrediction
        WorldModelPrediction:
          - predicted_next_hash: int | None
          - confidence: float (0.0-1.0)
          - predicted_frame_changes: bool | None
          - source: str ("graph" | "neural" | "unknown")

    - update(state_hash, action, next_state_hash, diff)
    - get_uncertainty(state_hash, action) -> float
        0.0 = known, 1.0 = unknown.
    - get_frontier_actions(state_hash) -> list[tuple[Action, float]]
        Actions ranked by uncertainty (highest first).
"""
```

### `remash/world_model/graph_model.py`

Phase 1 world model backed by the state graph.

```python
"""
Exact knowledge for observed transitions, unknown for everything else.
- predict: exact if observed, else None with confidence 0.0
- update: delegates to graph.add_transition
- get_uncertainty: 0.0 if in graph, else 1.0 (binary)
- get_frontier_actions: untested first, no-change last
"""
```

### `remash/policy/base.py`

```python
"""
- Policy (ABC):
    - select_action(state_hash, frame, objects, ui_state, world_model,
                    episode, graph, cross_level, context=None) -> Action
    - on_level_start(level_num)
    - on_level_complete(level_num)
"""
```

### `remash/policy/explorer.py`

Phase 1 systematic exploration policy.

```python
"""
Graph-guided systematic exploration.

Priority order:
1. Win state known and reachable -> execute shortest path.
2. Untested actions from current state -> try highest priority one.
3. All actions tested from current state -> navigate to nearest state
   with untested actions (BFS on graph).
4. No reachable frontier -> random action (may discover hidden states).

Action priority for untested actions:
- Use cross_level priors if available from previous levels.
- Default: directional actions first, then spacebar, then click on
  detected object centroids, then click random.
- For clicks: ONLY click on object centroids and detected UI elements.
  Never enumerate 4096 positions. Keep effective action space ~10-20.

Energy awareness:
- If energy/step counter detected and running low, switch to exploitation:
  execute best known path toward goal.

Death/reset awareness:
- When level resets (ran out of energy/lives), graph knowledge persists.
  Don't re-test known transitions. Resume exploration from reset state.
"""
```

### `remash/agent.py`

Main agent loop.

```python
"""
Main ReMash agent. Integrates all components.

- ReMashAgent:
    - __init__(policy, world_model)
    - play_game(env) -> GameResult

    Core loop pseudocode:

        cross_level = CrossLevelMemory()
        obs = env.reset()
        frame = Frame.from_raw(obs)
        graph = StateGraph()
        episode = EpisodeBuffer()
        level_num = 1
        prev_frame = None

        while not done:
            # Perception
            objects = detect_objects(frame)
            ui_state = detect_ui(frame, prev_frame)
            state_hash = frame.hash()
            graph.ensure_node(state_hash)

            # Diff from previous frame
            if prev_frame:
                diff = frame.diff(prev_frame)
                deltas = track_objects(prev_objects, objects, diff)
            else:
                diff, deltas = None, []

            # Check game state
            if obs.state == GameState.WIN:
                graph.mark_win_state(prev_state_hash)
                cross_level.on_level_complete(level_num, episode, graph)
                level_num += 1
                episode.clear()

            if obs.state == GameState.GAME_OVER:
                episode.clear()
                # Graph persists - don't re-explore known transitions

            if game_is_done(obs):
                break

            # Select and execute action
            action = policy.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level
            )

            prev_frame = frame
            prev_objects = objects
            prev_state_hash = state_hash

            obs = env.step(action)
            frame = Frame.from_raw(obs)

            # Update memory
            new_diff = frame.diff(prev_frame)
            new_objects = detect_objects(frame)
            new_deltas = track_objects(prev_objects, new_objects, new_diff)
            episode.add_step(prev_frame, action, frame, new_diff,
                           prev_objects, new_deltas, ui_state)
            world_model.update(prev_state_hash, action, frame.hash(), new_diff)

    IMPORTANT: The exact obs attribute names and GameState values depend on
    FRAME_FORMAT.md findings. Adapt this pseudocode to match reality.
"""
```

### Scripts

**`scripts/discover_format.py`**: The step zero discovery script (see above).

**`scripts/play.py`**: Run agent on a single game.
```
Usage: uv run scripts/play.py --game ls20 [--render terminal] [--max-steps 500]
Output: per-level action counts, win/loss status, episode log.
```

**`scripts/benchmark.py`**: Run agent on all public games.
```
Usage: uv run scripts/benchmark.py [--games ls20,ft09,vc33]
Output: per-game scores, per-level breakdown, aggregate score.
```

---

## Interface Contracts for Future Phases

These are defined now so future components plug in without refactoring.

### Phase 2: Neural World Model (CfC)
Implements WorldModel ABC. CNN encoder (Frame -> latent z_t).
CfC predicts z_{t+1} = CfC(z_t, action, context).
Ensemble of 3-5 CfC heads for uncertainty via disagreement.
Graph model still runs underneath as cache:
  - (state, action) in graph: uncertainty = 0.0
  - Not in graph: uncertainty = ensemble disagreement (continuous)

### Phase 3: EFE Active Inference Policy
Implements Policy ABC. Computes Expected Free Energy:
  G(a) = -epistemic_value(a) - pragmatic_value(a)
  epistemic = world_model.get_uncertainty(state, a)
  pragmatic = similarity(predicted_next_state, inferred_goal)
Explorer logic kept as fallback for high-uncertainty situations.

### Phase 4: LinOSS Cross-Level Memory
cross_level.py upgraded to LinOSS module.
Produces context_vector consumed by CfC (FiLM conditioning),
EFE policy (goal priors), and interoceptive module.
Updated only at level boundaries.

### Phase 5: Interoceptive Module
Small CfC with slow time constants.
Inputs: world_model_loss, steps_remaining, learning_rate.
Outputs: modulation vector for policy precision weighting.

---

## Build Order

1. Create repo structure and pyproject.toml.
2. Run discover_format.py. Document findings in FRAME_FORMAT.md.
3. Implement Frame class based on discovered format.
4. Implement flood fill object detection.
5. Implement StateGraph.
6. Implement explorer policy.
7. Wire everything together in agent.py.
8. Run on ls20. Target: complete level 1 within ~200 actions.
9. Run on all public games. Report scores.
10. Analyze failure modes. Identify where neural components would help.

---

## Design Principles

- **No premature neural networks.** Phase 1 has zero learned parameters.
  Every component is deterministic and debuggable.

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