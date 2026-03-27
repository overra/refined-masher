"""
ReMash ARC-AGI-3 Kaggle Submission Notebook

Install and run the ReMash agent against all competition games.
This notebook is designed to run on Kaggle's servers with no internet
access during evaluation (the package is pre-installed).
"""

# --- Install ---
import subprocess
subprocess.check_call([
    "pip", "install", "-q",
    "git+https://github.com/overra/refined-masher.git",
    "xxhash",
])

# --- Run agent ---
import logging
import time

import arc_agi
from arcengine import GameAction, GameState

# Import our agent components
from remash.perception.frame import Frame
from remash.perception.objects import detect_objects, detect_background_color, track_objects
from remash.perception.ui import UIDetector
from remash.memory.state_graph import StateGraph
from remash.memory.cross_level import CrossLevelMemory
from remash.memory.episode import EpisodeBuffer
from remash.world_model.graph_model import GraphWorldModel
from remash.policy.efe import EFEPolicy
from remash.agent import ClickTargetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("remash")


def play_game(env, game_id, max_actions=2000):
    """Play a single game using the ReMash agent."""
    obs = env.reset()
    if obs is None:
        return

    frame = Frame.from_raw(obs)
    available_actions = [GameAction.from_id(a) for a in obs.available_actions]

    # Initialize components
    graph = StateGraph(available_actions=available_actions)
    world_model = GraphWorldModel(graph)
    cross_level = CrossLevelMemory()
    episode = EpisodeBuffer()
    click_mgr = ClickTargetManager()
    ui_detector = UIDetector()
    policy = EFEPolicy()
    policy.on_level_start(0)

    level_num = 0
    prev_frame = None
    prev_objects = []
    prev_state_hash = 0
    prev_action = None
    prev_click_target = None

    for step in range(max_actions):
        # Handle reset states
        if obs.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if obs.state == GameState.GAME_OVER:
                episode.clear()
                # Reset click targets but keep color priors
                saved_priors = click_mgr._color_priors.copy()
                saved_responsive = click_mgr._responsive_colors.copy()
                click_mgr.clear()
                click_mgr._color_priors = saved_priors
                if saved_responsive:
                    click_mgr.set_color_priors(saved_priors | saved_responsive)
                prev_frame = None
            obs = env.step(GameAction.RESET)
            if obs is None or not hasattr(obs, 'frame') or not obs.frame:
                continue
            frame = Frame.from_raw(obs)
            continue

        if not hasattr(obs, 'frame') or not obs.frame:
            break

        frame = Frame.from_raw(obs)

        # Perception
        objects = detect_objects(frame)
        ui_state = ui_detector.detect(frame, prev_frame)
        state_hash = frame.game_hash(ui_state.ui_region_mask)
        graph.ensure_node(state_hash)

        # Process previous step's results BEFORE selecting next action
        if prev_frame is not None and prev_action is not None:
            diff = frame.diff(prev_frame)
            new_deltas = track_objects(prev_objects, objects, diff)
            episode.add_step(
                prev_frame, prev_action, frame, diff,
                prev_objects, new_deltas, ui_state,
            )
            world_model.update(prev_state_hash, prev_action, state_hash, diff)

            spatial = getattr(policy, "spatial", None)
            if spatial is not None:
                bg = detect_background_color(frame)
                spatial.on_step(prev_action, new_deltas, diff.num_changed, objects, bg)
            if hasattr(policy, "on_step_result"):
                policy.on_step_result(state_hash, diff.num_changed)
            if prev_click_target is not None:
                click_mgr.report_result(diff.num_changed)

        # Check level transition
        if obs.levels_completed > level_num:
            resp_colors = click_mgr.get_responsive_colors()
            cross_level.on_level_complete(
                level_num, episode, graph,
                responsive_click_colors=resp_colors,
            )
            policy.on_level_complete(level_num)
            level_num = obs.levels_completed
            episode.clear()
            click_mgr.clear()
            color_priors = cross_level.get_responsive_click_colors()
            click_mgr.set_color_priors(color_priors)
            policy.on_level_start(level_num)
            logger.info(f"Level {level_num-1} complete! Now on level {level_num}")

        # Select action
        action = policy.select_action(
            state_hash, frame, objects, ui_state,
            world_model, episode, graph, cross_level,
        )

        # Handle click coordinates
        click_target = None
        if action.is_complex():
            efe_target = getattr(policy, "selected_click_target", None)
            if efe_target is not None:
                click_target = efe_target
            else:
                click_target = click_mgr.pick_target(state_hash, objects)
            action.set_data({"x": click_target[0], "y": click_target[1]})

        # Save state for next iteration
        prev_frame = frame
        prev_objects = objects
        prev_state_hash = state_hash
        prev_action = action
        prev_click_target = click_target

        # Execute action
        if action.is_complex() and click_target:
            obs = env.step(action, data={"x": click_target[0], "y": click_target[1]})
        else:
            obs = env.step(action)

        if obs is None:
            break

    logger.info(f"{game_id}: {obs.levels_completed} levels in {step+1} actions")


# --- Main ---
print("Starting ReMash agent...")
t0 = time.time()

arcade = arc_agi.Arcade()
games = arcade.get_environments()
print(f"Found {len(games)} games")

for env_info in games:
    gid = env_info.game_id
    try:
        env = arcade.make(gid)
        if env is None:
            continue
        play_game(env, gid)
    except Exception as e:
        print(f"Error on {gid}: {e}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(arcade.get_scorecard())
