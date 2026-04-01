"""Pre-train CNN encoder + MLP ensemble on public games with color augmentation.

The encoder and dynamics learn general grid-world physics so the agent
starts each new game already knowing how to see and predict.

Key design (per Gemini Deep Think analysis):
- 16-channel one-hot input: color permutation = channel permutation
- L2-normalized latents: bounded MSE, no scale explosion
- Persistent optimizer across episodes: Adam keeps momentum
- Global replay buffer: raw integer grids, permutation applied per-batch
- No offline rehearsal: online training within episodes is sufficient

Usage:
    uv run python scripts/pretrain.py                      # 100 episodes
    uv run python scripts/pretrain.py --episodes 200
    uv run python scripts/pretrain.py --resume
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
    EnsembleWorldModel, SpatialEncoder, EnsembleDynamics, ReplayBuffer,
    LEARNING_RATE, LATENT_CHANNELS,
)
from remash.utils.logging import setup_logging

WEIGHTS_DIR = Path("weights")
CHECKPOINT_PATH = WEIGHTS_DIR / "pretrained.pt"

# Hold-out games for validation
HOLDOUT_GAMES = {"ft09", "dc22", "sb26", "r11l", "tn36"}


def make_color_permutation() -> np.ndarray:
    """Random permutation of color indices 1-15. Color 0 (background) stays fixed."""
    perm = np.arange(16, dtype=np.uint8)
    np.random.shuffle(perm[1:])  # keep 0 fixed
    return perm


class PretrainWorldModel(EnsembleWorldModel):
    """World model that applies per-episode color permutation to cached frames."""

    def __init__(self, graph: StateGraph, color_perm: np.ndarray,
                 encoder: SpatialEncoder, target_encoder: SpatialEncoder,
                 dynamics: EnsembleDynamics, optimizer: torch.optim.Adam,
                 replay: ReplayBuffer):
        # Skip EnsembleWorldModel.__init__ — we inject everything
        self.graph = graph
        self.device = torch.device("cpu")
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.dynamics = dynamics
        self.optimizer = optimizer
        self.replay = replay
        self._step_count = 0
        self._train_count = 0
        self.avg_loss = 0.0
        self._loss_history = []
        self._latent_cache = {}
        from collections import OrderedDict
        self._frame_cache = OrderedDict()
        self._max_uncertainty = 0.01
        self._color_perm = color_perm

    def cache_frame(self, state_hash: int, grid: np.ndarray) -> None:
        """Cache color-permuted frame (raw integers, not float)."""
        augmented = self._color_perm[grid]
        if state_hash not in self._frame_cache:
            self._frame_cache[state_hash] = augmented.copy()
            while len(self._frame_cache) > 500:
                self._frame_cache.popitem(last=False)


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

    # === PERSISTENT STATE (survives across all episodes) ===
    encoder = SpatialEncoder().to(device)
    target_encoder = SpatialEncoder().to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    dynamics = EnsembleDynamics().to(device)

    # Single optimizer — Adam keeps momentum and variance across episodes
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(dynamics.parameters()),
        lr=LEARNING_RATE,
    )

    # Global replay buffer — raw integer grids from all episodes
    global_replay = ReplayBuffer(max_size=100_000)

    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        target_encoder.load_state_dict(ckpt["encoder"])
        dynamics.load_state_dict(ckpt["dynamics"])
        print(f"Resumed from {CHECKPOINT_PATH} (episode {ckpt.get('episode', '?')})")

    print(f"\nPre-training {args.episodes} episodes...")
    print(f"{'Ep':>4} {'Game':<15} {'Lvl':>5} {'Steps':>6} {'Loss':>8} {'Replay':>7} {'Time':>6}")
    print("-" * 62)

    t0 = time.time()

    for ep in range(args.episodes):
        game_id = random.choice(train_games)
        color_perm = make_color_permutation()

        env = arcade.make(game_id)
        if env is None:
            continue

        # Fresh graph per episode (don't memorize game solutions)
        # but shared encoder, dynamics, optimizer, and replay buffer
        graph = StateGraph()
        wm = PretrainWorldModel(
            graph=graph,
            color_perm=color_perm,
            encoder=encoder,
            target_encoder=target_encoder,
            dynamics=dynamics,
            optimizer=optimizer,
            replay=global_replay,
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

        print(
            f"{ep+1:4d} {game_id:<15} {result.levels_completed:>2}/{result.win_levels:<3}"
            f" {result.total_steps:>6} {wm.avg_loss:>8.5f} {len(global_replay):>7} {ep_time:>5.1f}s"
        )

        # Checkpoint every 25 episodes
        if (ep + 1) % 25 == 0:
            numbered_path = WEIGHTS_DIR / f"pretrained_ep{ep+1}.pt"
            ckpt_data = {
                "encoder": encoder.state_dict(),
                "dynamics": dynamics.state_dict(),
                "episode": ep + 1,
            }
            torch.save(ckpt_data, numbered_path)
            torch.save(ckpt_data, CHECKPOINT_PATH)
            print(f"     [checkpoint] saved to {numbered_path}")

    elapsed = time.time() - t0

    torch.save({
        "encoder": encoder.state_dict(),
        "dynamics": dynamics.state_dict(),
        "episode": args.episodes,
    }, CHECKPOINT_PATH)

    print(f"\nPre-training complete: {args.episodes} episodes in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Global replay buffer: {len(global_replay)} transitions")
    print(f"Saved weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
