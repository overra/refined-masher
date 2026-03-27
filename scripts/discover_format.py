"""Step Zero: Inspect ARC-AGI-3 toolkit data formats before writing any code."""

import numpy as np

import arc_agi
from arcengine import GameAction, GameState

arc = arc_agi.Arcade()
print(f"Available games: {len(arc.available_environments)}")
for e in arc.available_environments:
    print(f"  {e.game_id}: {e.title}")

env = arc.make("ls20")
if env is None:
    print("ERROR: Could not create ls20 environment")
    raise SystemExit(1)

obs = env.reset()

print("\n=== Observation type ===")
print(f"type: {type(obs)}")
print(f"is FrameDataRaw: {type(obs).__name__}")

print("\n=== Observation attributes ===")
for attr in sorted(dir(obs)):
    if not attr.startswith("_"):
        try:
            val = getattr(obs, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__} = {repr(val)[:200]}")
        except Exception as e:
            print(f"  {attr}: ERROR {e}")

print("\n=== Frame data (the pixel grid) ===")
frame = obs.frame
print(f"  obs.frame type: {type(frame)}")
if isinstance(frame, list):
    print(f"  obs.frame length: {len(frame)}")
    for i, f in enumerate(frame):
        print(f"  frame[{i}]: type={type(f)}")
        if isinstance(f, np.ndarray):
            print(f"    shape={f.shape}, dtype={f.dtype}")
            print(f"    min={f.min()}, max={f.max()}")
            unique = np.unique(f)
            print(f"    unique values ({len(unique)}): {unique}")
            print(f"    sample [0:3, 0:3]:\n{f[0:3, 0:3]}")
elif isinstance(frame, np.ndarray):
    print(f"  shape={frame.shape}, dtype={frame.dtype}")
    print(f"  min={frame.min()}, max={frame.max()}")
    unique = np.unique(frame)
    print(f"  unique values ({len(unique)}): {unique}")
    print(f"  sample [0:3, 0:3]:\n{frame[0:3, 0:3]}")

print("\n=== GameState enum values ===")
for s in GameState:
    print(f"  {s.name}: {s.value}")
print(f"  Current obs.state: {obs.state}")

print("\n=== Levels info ===")
print(f"  levels_completed: {obs.levels_completed}")
print(f"  win_levels: {obs.win_levels}")

print("\n=== Action space ===")
print(f"  env.action_space: {env.action_space}")
for a in GameAction:
    print(f"  {a.name}: value={a.value}, simple={a.is_simple()}, complex={a.is_complex()}")

print("\n=== Available actions from obs ===")
print(f"  obs.available_actions: {obs.available_actions}")

print("\n=== Take ACTION1 ===")
obs2 = env.step(GameAction.ACTION1)
print(f"  state: {obs2.state}")
print(f"  levels_completed: {obs2.levels_completed}")
frame2 = obs2.frame
if isinstance(frame2, list):
    print(f"  frame count: {len(frame2)}")
    for i, f in enumerate(frame2):
        if isinstance(f, np.ndarray):
            print(f"  frame[{i}]: shape={f.shape}, dtype={f.dtype}, unique_count={len(np.unique(f))}")

# Check if frames changed
if isinstance(frame, list) and isinstance(frame2, list):
    if len(frame) > 0 and len(frame2) > 0:
        f1 = frame[0] if isinstance(frame[0], np.ndarray) else np.array(frame[0])
        f2 = frame2[0] if isinstance(frame2[0], np.ndarray) else np.array(frame2[0])
        if f1.shape == f2.shape:
            diff = f1 != f2
            print(f"  Pixels changed after ACTION1: {diff.sum()}")

print("\n=== Take all simple actions and check frame changes ===")
env2 = arc.make("ls20")
base_obs = env2.reset()
base_frame = base_obs.frame[0] if isinstance(base_obs.frame, list) and len(base_obs.frame) > 0 else None

if base_frame is not None and isinstance(base_frame, np.ndarray):
    for action in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                   GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]:
        env3 = arc.make("ls20")
        obs_reset = env3.reset()
        obs_act = env3.step(action)
        act_frame = obs_act.frame[0] if isinstance(obs_act.frame, list) and len(obs_act.frame) > 0 else None
        if act_frame is not None and isinstance(act_frame, np.ndarray):
            reset_frame = obs_reset.frame[0]
            diff_count = (reset_frame != act_frame).sum()
            print(f"  {action.name}: {diff_count} pixels changed, state={obs_act.state}")

print("\n=== Try complex action (click at 32,32) ===")
env4 = arc.make("ls20")
obs_reset = env4.reset()
obs_click = env4.step(GameAction.ACTION6, data={"x": 32, "y": 32})
if obs_click:
    click_frame = obs_click.frame[0] if isinstance(obs_click.frame, list) and len(obs_click.frame) > 0 else None
    reset_frame = obs_reset.frame[0] if isinstance(obs_reset.frame, list) and len(obs_reset.frame) > 0 else None
    if click_frame is not None and reset_frame is not None:
        diff_count = (reset_frame != click_frame).sum()
        print(f"  Click at (32,32): {diff_count} pixels changed, state={obs_click.state}")
else:
    print("  Click returned None")

print("\n=== Check if frame values are color indices or RGB ===")
if isinstance(frame, list) and len(frame) > 0:
    f = frame[0]
    if isinstance(f, np.ndarray):
        print(f"  ndim: {f.ndim}")
        if f.ndim == 2:
            print("  -> 2D array: likely color indices (not RGB)")
            print(f"  Value range: {f.min()} to {f.max()}")
        elif f.ndim == 3:
            print(f"  -> 3D array: shape {f.shape}")
            if f.shape[2] == 3:
                print("  -> RGB image")
                # Sample some unique RGB values
                reshaped = f.reshape(-1, 3)
                unique_rgb = np.unique(reshaped, axis=0)
                print(f"  Unique RGB values ({len(unique_rgb)}):")
                for rgb in unique_rgb[:20]:
                    print(f"    {tuple(rgb)}")
            elif f.shape[2] == 4:
                print("  -> RGBA image")

print("\n=== observation_space property ===")
obs_space = env.observation_space
print(f"  type: {type(obs_space)}")
if obs_space is not None:
    print(f"  is same as last step result: {obs_space is obs2}")

print("\n=== Full game reset check ===")
print(f"  obs.full_reset: {obs.full_reset}")
print(f"  obs.guid: {obs.guid}")
print(f"  obs.game_id: {obs.game_id}")
print(f"  obs.action_input: {obs.action_input}")
