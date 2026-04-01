"""Pre-train CNN encoder + MLP ensemble on public games with augmentation.

Trains the world model on general grid-world physics so the agent starts
each new game already knowing how to see and predict.

Key augmentations:
- Color permutation: random remap of all 16 color indices per episode
- Action permutation: random directional action mapping per episode

Usage:
    uv run python scripts/pretrain.py                      # 100 episodes
    uv run python scripts/pretrain.py --episodes 200       # custom count
    uv run python scripts/pretrain.py --resume             # resume from checkpoint
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

import arc_agi
from arcengine import GameAction

from remash.agent import ReMashAgent
from remash.policy.explorer import ExplorerPolicy
from remash.memory.state_graph import StateGraph
from remash.world_model.ensemble_model import (
    EnsembleWorldModel, SpatialEncoder, EnsembleDynamics,
    LEARNING_RATE,
)
from remash.utils.logging import setup_logging, logger

WEIGHTS_DIR = Path("weights")
CHECKPOINT_PATH = WEIGHTS_DIR / "pretrained.pt"

# Hold-out games for validation (zero-score games)
HOLDOUT_GAMES = {"ft09", "dc22", "sb26", "r11l", "tn36"}

# Offline rehearsal config
REHEARSAL_EVERY = 10
REHEARSAL_PASSES = 5
REHEARSAL_BATCH = 32


def make_color_permutation() -> np.ndarray:
    """Random permutation of 16 color indices."""
    perm = np.arange(16, dtype=np.uint8)
    np.random.shuffle(perm)
    return perm


class AugmentedEnsembleWorldModel(EnsembleWorldModel):
    """World model that applies color permutation to cached frames."""

    def __init__(self, graph: StateGraph, color_perm: np.ndarray):
        super().__init__(graph)
        self._color_perm = color_perm

    def cache_frame(self, state_hash: int, grid: np.ndarray) -> None:
        augmented = self._color_perm[grid]
        super().cache_frame(state_hash, augmented)




def main():
    parser = argparse.ArgumentParser(description="Pre-train world model on public games")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_logging(level=logging.WARNING)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    arcade = arc_agi.Arcade()
    all_envs = arcade.get_environments()
    train_games = [e.game_id for e in all_envs if e.game_id.split("-")[0] not in HOLDOUT_GAMES]
    print(f"Training games: {len(train_games)}, Hold-out: {len(HOLDOUT_GAMES)}")

    device = torch.device("cpu")

    # Shared encoder + dynamics (persist across episodes, online training only)
    shared_encoder = SpatialEncoder().to(device)
    shared_dynamics = EnsembleDynamics().to(device)

    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        shared_encoder.load_state_dict(ckpt["encoder"])
        shared_dynamics.load_state_dict(ckpt["dynamics"])
        print(f"Resumed from {CHECKPOINT_PATH} (episode {ckpt.get('episode', '?')})")

    print(f"\nPre-training {args.episodes} episodes...")
    print(f"{'Ep':>4} {'Game':<15} {'Lvl':>5} {'Steps':>6} {'Loss':>8} {'Buf':>7} {'Time':>6}")
    print("-" * 58)

    t0 = time.time()

    for ep in range(args.episodes):
        game_id = random.choice(train_games)
        color_perm = make_color_permutation()

        env = arcade.make(game_id)
        if env is None:
            continue

        # Create fresh graph and augmented world model with shared weights
        graph = StateGraph()
        wm = AugmentedEnsembleWorldModel(graph, color_perm)

        # Copy shared weights into the episode's world model
        wm.encoder.load_state_dict(shared_encoder.state_dict())
        wm.target_encoder.load_state_dict(shared_target_encoder.state_dict())
        wm.dynamics.load_state_dict(shared_dynamics.state_dict())
        wm.optimizer = torch.optim.Adam(
            list(wm.encoder.parameters()) + list(wm.dynamics.parameters()),
            lr=LEARNING_RATE,
        )

        policy = ExplorerPolicy()
        agent = ReMashAgent(policy=policy, max_total_steps=args.max_steps)

        ep_t0 = time.time()
        result = agent.play_game(
            env, game_id=game_id,
            competition_mode=True,
            external_world_model=wm,
        )
        ep_time = time.time() - ep_t0

        # Copy trained weights back to shared model
        shared_encoder.load_state_dict(wm.encoder.state_dict())
        shared_target_encoder.load_state_dict(wm.target_encoder.state_dict())
        shared_dynamics.load_state_dict(wm.dynamics.state_dict())

        print(
            f"{ep+1:4d} {game_id:<15} {result.levels_completed:>2}/{result.win_levels:<3}"
            f" {result.total_steps:>6} {wm.avg_loss:>8.5f} {ep_time:>5.1f}s"
        )

        # Checkpoint periodically (save both numbered and latest)
        if (ep + 1) % 25 == 0:
            numbered_path = WEIGHTS_DIR / f"pretrained_ep{ep+1}.pt"
            ckpt_data = {
                "encoder": shared_encoder.state_dict(),
                "dynamics": shared_dynamics.state_dict(),
                "episode": ep + 1,
            }
            torch.save(ckpt_data, numbered_path)
            torch.save(ckpt_data, CHECKPOINT_PATH)
            print(f"     [checkpoint] saved to {numbered_path}")

    elapsed = time.time() - t0

    # Final save
    torch.save({
        "encoder": shared_encoder.state_dict(),
        "dynamics": shared_dynamics.state_dict(),
        "episode": args.episodes,
    }, CHECKPOINT_PATH)

    print(f"\nPre-training complete: {args.episodes} episodes in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Final buffer: {len(persistent_buffer)} transitions")
    print(f"Saved weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
