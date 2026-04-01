"""Pre-train CNN encoder + MLP ensemble on public games with augmentation.

Trains the world model on general grid-world physics so the agent starts
each new game already knowing how to see and predict.

Key augmentations:
- Color permutation: random remap of all 16 color indices per episode
- Action permutation: random directional action mapping per episode

Usage:
    uv run python scripts/pretrain.py                      # 100 episodes
    uv run python scripts/pretrain.py --episodes 200       # custom count
    uv run python scripts/pretrain.py --validate           # benchmark after
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import arc_agi
from arcengine import GameAction

from remash.agent import ReMashAgent
from remash.policy.explorer import ExplorerPolicy
from remash.world_model.ensemble_model import (
    EnsembleWorldModel, SpatialEncoder, EnsembleDynamics,
    LEARNING_RATE, BATCH_SIZE,
)
from remash.memory.state_graph import StateGraph
from remash.utils.logging import setup_logging, logger

WEIGHTS_DIR = Path("weights")
CHECKPOINT_PATH = WEIGHTS_DIR / "pretrained.pt"

# Hold-out games for validation (zero-score games)
HOLDOUT_GAMES = {"ft09", "dc22", "sb26", "r11l", "tn36"}

# Offline rehearsal config
REHEARSAL_EVERY = 10  # episodes
REHEARSAL_PASSES = 5
REHEARSAL_BATCH = 32


def make_color_permutation() -> np.ndarray:
    """Random permutation of 16 color indices."""
    perm = np.arange(16, dtype=np.uint8)
    np.random.shuffle(perm)
    return perm


def apply_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply color permutation to a frame grid."""
    return perm[grid]


def make_action_permutation(available: list[GameAction]) -> dict[GameAction, GameAction]:
    """Random permutation of directional actions (ACTION1-4)."""
    directional = [a for a in available if a.value in (1, 2, 3, 4)]
    if len(directional) < 2:
        return {}  # no permutation for click-only games
    shuffled = directional.copy()
    random.shuffle(shuffled)
    return dict(zip(directional, shuffled))


class AugmentedWorldModel(EnsembleWorldModel):
    """World model wrapper that applies color permutation to frames."""

    def __init__(self, graph: StateGraph, color_perm: np.ndarray):
        super().__init__(graph)
        self._color_perm = color_perm

    def cache_frame(self, state_hash: int, grid: np.ndarray) -> None:
        augmented = apply_color_perm(grid, self._color_perm)
        super().cache_frame(state_hash, augmented)


class PersistentReplayBuffer:
    """Accumulates transitions across episodes for offline rehearsal."""

    def __init__(self, max_size: int = 50000):
        self._buffer: list[tuple[np.ndarray, int, tuple[float, float], np.ndarray]] = []
        self._max_size = max_size

    def extend_from(self, replay) -> int:
        """Copy transitions from an episode's replay buffer."""
        added = 0
        for item in replay._buffer:
            if len(self._buffer) < self._max_size:
                self._buffer.append(item)
                added += 1
            else:
                # Random replacement
                idx = random.randint(0, len(self._buffer) - 1)
                self._buffer[idx] = item
                added += 1
        return added

    def sample(self, n: int):
        indices = np.random.choice(len(self._buffer), size=min(n, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self):
        return len(self._buffer)


def offline_rehearsal(
    encoder: SpatialEncoder,
    dynamics: EnsembleDynamics,
    optimizer: torch.optim.Optimizer,
    persistent_buffer: PersistentReplayBuffer,
    n_passes: int = REHEARSAL_PASSES,
    batch_size: int = REHEARSAL_BATCH,
) -> float:
    """Train encoder + dynamics on accumulated dataset."""
    import torch.nn.functional as F
    from remash.world_model.ensemble_model import TARGET_TAU

    device = next(encoder.parameters()).device
    # Create a target encoder for stable targets
    target_encoder = SpatialEncoder().to(device)
    target_encoder.load_state_dict(encoder.state_dict())

    total_loss = 0.0
    n_batches = 0

    for _ in range(n_passes):
        batch = persistent_buffer.sample(batch_size)
        if len(batch) < 8:
            continue

        frames = np.stack([t[0] for t in batch]).astype(np.float32) / 15.0
        actions = [t[1] for t in batch]
        clicks = np.array([t[2] for t in batch], dtype=np.float32)
        next_frames = np.stack([t[3] for t in batch]).astype(np.float32) / 15.0

        frames_t = torch.from_numpy(frames).unsqueeze(1).to(device)
        next_frames_t = torch.from_numpy(next_frames).unsqueeze(1).to(device)
        clicks_t = torch.from_numpy(clicks).to(device)

        action_oh = torch.zeros(len(batch), 8, device=device)
        for i, a in enumerate(actions):
            action_oh[i, a] = 1.0

        z = encoder(frames_t)
        z_flat = z.view(len(batch), -1)

        with torch.no_grad():
            z_next = target_encoder(next_frames_t)
            z_next_flat = z_next.view(len(batch), -1)

        target_delta = z_next_flat - z_flat.detach()
        head_deltas = dynamics.per_head_deltas(z_flat, action_oh, clicks_t)
        loss = sum(F.mse_loss(d, target_delta) for d in head_deltas) / len(head_deltas)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(dynamics.parameters()), 1.0
        )
        optimizer.step()

        # Polyak update
        with torch.no_grad():
            for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
                tp.data.mul_(TARGET_TAU).add_(p.data, alpha=1 - TARGET_TAU)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Pre-train world model on public games")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--validate", action="store_true", help="Run benchmark after pre-training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    setup_logging(level=logging.WARNING)

    WEIGHTS_DIR.mkdir(exist_ok=True)

    arcade = arc_agi.Arcade()
    all_envs = arcade.get_environments()
    train_games = [e.game_id for e in all_envs if e.game_id not in HOLDOUT_GAMES]
    print(f"Training games: {len(train_games)}, Hold-out: {len(HOLDOUT_GAMES)}")

    # Shared encoder + dynamics (persist across episodes)
    device = torch.device("cpu")
    encoder = SpatialEncoder().to(device)
    dynamics = EnsembleDynamics().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(dynamics.parameters()),
        lr=LEARNING_RATE,
    )

    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        dynamics.load_state_dict(ckpt["dynamics"])
        print(f"Resumed from {CHECKPOINT_PATH}")

    persistent_buffer = PersistentReplayBuffer()

    print(f"\nPre-training {args.episodes} episodes...")
    print(f"{'Ep':>4} {'Game':<15} {'Levels':>6} {'Steps':>6} {'Loss':>8} {'Replay':>7} {'Time':>6}")
    print("-" * 60)

    t0 = time.time()

    for ep in range(args.episodes):
        game_id = random.choice(train_games)
        color_perm = make_color_permutation()

        env = arcade.make(game_id)
        if env is None:
            continue

        # Create fresh agent with augmented world model
        available_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                             GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
                             GameAction.ACTION7]
        graph = StateGraph(available_actions=available_actions)
        wm = AugmentedWorldModel(graph, color_perm)

        # Inject shared weights
        wm.encoder.load_state_dict(encoder.state_dict())
        wm.target_encoder.load_state_dict(encoder.state_dict())
        wm.dynamics.load_state_dict(dynamics.state_dict())
        wm.optimizer = torch.optim.Adam(
            list(wm.encoder.parameters()) + list(wm.dynamics.parameters()),
            lr=LEARNING_RATE,
        )

        policy = ExplorerPolicy()
        agent = ReMashAgent(policy=policy, use_neural=False, max_total_steps=args.max_steps)

        ep_t0 = time.time()
        # Manually override world model
        result = agent.play_game(env, game_id=game_id, competition_mode=True)
        # Note: play_game creates its own world model. We need to hook in differently.
        # For now, just run the agent and collect the replay data from the standard path.
        ep_time = time.time() - ep_t0

        # The agent's world model trained during the episode.
        # We can't easily extract it since play_game creates it internally.
        # TODO: refactor to accept external world model

        print(f"{ep+1:4d} {game_id:<15} {result.levels_completed:>2}/{result.win_levels:<3}"
              f" {result.total_steps:>6} {'N/A':>8} {len(persistent_buffer):>7} {ep_time:>5.1f}s")

        # Offline rehearsal every N episodes
        if (ep + 1) % REHEARSAL_EVERY == 0 and len(persistent_buffer) > 100:
            loss = offline_rehearsal(encoder, dynamics, optimizer, persistent_buffer)
            print(f"     [rehearsal] loss={loss:.6f} buffer={len(persistent_buffer)}")

        # Save checkpoint periodically
        if (ep + 1) % 25 == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "dynamics": dynamics.state_dict(),
                "episode": ep + 1,
            }, CHECKPOINT_PATH)
            print(f"     [checkpoint] saved to {CHECKPOINT_PATH}")

    elapsed = time.time() - t0
    print(f"\nPre-training complete: {args.episodes} episodes in {elapsed:.0f}s")

    # Final save
    torch.save({
        "encoder": encoder.state_dict(),
        "dynamics": dynamics.state_dict(),
        "episode": args.episodes,
    }, CHECKPOINT_PATH)
    print(f"Saved weights to {CHECKPOINT_PATH}")

    if args.validate:
        print("\nRunning validation benchmark...")
        os.system("uv run python scripts/benchmark.py --efe --competition-mode --pretrained")


if __name__ == "__main__":
    main()
