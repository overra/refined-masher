# Frame Format Discovery (Step Zero)

Discovered 2026-03-27 from `arc-agi==0.9.6`, `arcengine==0.9.3`, game `ls20-9607627b`.

## Observation: FrameDataRaw

`env.reset()` and `env.step()` both return `arcengine.enums.FrameDataRaw` (Pydantic BaseModel).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `list[np.ndarray]` | List of frame layers. Each is `(64, 64)` int8. Typically length 1. |
| `state` | `GameState` | Current game state enum |
| `levels_completed` | `int` | Number of levels completed so far |
| `win_levels` | `int` | Total levels needed to win the game (7 for ls20) |
| `available_actions` | `list[int]` | Action IDs available in current state |
| `game_id` | `str` | Game ID with version hash (e.g., `ls20-9607627b`) |
| `guid` | `str` | Unique run identifier |
| `full_reset` | `bool` | True on initial reset |
| `action_input` | `ActionInput` | The action that produced this observation |

### Frame Grid

- **Type**: `list[np.ndarray]` - list of layers, typically length 1
- **Access**: `obs.frame[0]` for the primary grid
- **Shape**: `(64, 64)`
- **Dtype**: `int8` (signed, but values are non-negative)
- **Values**: Color indices 0-15. Not all colors used in every game.
- **NOT RGB**. No color-to-index conversion needed.
- ls20 initial frame uses colors: `[0, 1, 3, 4, 5, 8, 9, 11, 12]` (9 of 16)

### GameState Enum

| Value | Meaning |
|-------|---------|
| `NOT_PLAYED` | Initial state before any interaction |
| `NOT_FINISHED` | Game/level in progress |
| `WIN` | Level completed |
| `GAME_OVER` | Failed (out of energy/lives/etc.) |

**Note**: The spec assumed `PLAYING` - actual value is `NOT_FINISHED`.

### Actions

| Action | Value | Type | ls20 available | Initial effect (ls20) |
|--------|-------|------|----------------|----------------------|
| `RESET` | 0 | Simple | No | Resets game |
| `ACTION1` | 1 | Simple | Yes | 52 pixels changed (movement) |
| `ACTION2` | 2 | Simple | Yes | 2 pixels changed (subtle) |
| `ACTION3` | 3 | Simple | Yes | 52 pixels changed (movement) |
| `ACTION4` | 4 | Simple | Yes | 52 pixels changed (movement) |
| `ACTION5` | 5 | Simple | No | 0 pixels changed |
| `ACTION6` | 6 | Complex (x,y) | No | 0 pixels changed (click) |
| `ACTION7` | 7 | Simple | No | 0 pixels changed |

**Critical**: Available actions vary per game. Read from `obs.available_actions`.
ls20 only uses actions `[1, 2, 3, 4]`. ACTION5, 6, 7 are not available.

### Complex Action (Click)

Pass coordinates via `data` parameter on `step()`:
```python
obs = env.step(GameAction.ACTION6, data={"x": 32, "y": 32})
```
Coordinates: 0-63 for both x and y.

### observation_space

`env.observation_space` returns the last `FrameDataRaw` (same object as last step result).

### action_space

`env.action_space` returns `list[GameAction]` based on `obs.available_actions`.

## Design Implications

1. `Frame.from_raw(obs)` should extract `obs.frame[0]` as the primary grid.
2. Grid is already color indices - no RGB lookup table needed.
3. dtype is `int8`; cast to `uint8` for consistency (values are 0-15).
4. Must check `obs.available_actions` per game, not assume all 7 actions exist.
5. GameState checks: use `NOT_FINISHED` not `PLAYING`.
6. `win_levels` tells total levels; `levels_completed` tracks progress.
7. Multi-layer frames (`len(obs.frame) > 1`) may exist in other games - handle gracefully.
