"""Residual MLP ensemble world model.

Predicts Δz (change in latent) via ensemble of MLP heads.
Uncertainty = ensemble disagreement.
Graph model runs underneath as exact cache.

Key design decisions (informed by Gemini Deep Think analysis):
- 16-channel one-hot input: color permutation becomes channel permutation
- L2-normalized latents: bounds MSE loss, prevents scale explosion
- Raw integer frames in replay buffer: augmentation applied per-batch
- Residual prediction with zero-init output: stable identity prior
"""

from __future__ import annotations

from collections import deque, OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcengine import GameAction

from remash.memory.state_graph import StateGraph
from remash.perception.frame import FrameDiff
from remash.world_model.base import WorldModel, WorldModelPrediction

if TYPE_CHECKING:
    pass

# --- Hyperparameters ---
NUM_COLORS = 16  # ARC-AGI-3 uses 16 discrete colors
LATENT_CHANNELS = 8  # spatial latent: 8x8xC
LATENT_SPATIAL = 8
LATENT_DIM = LATENT_CHANNELS * LATENT_SPATIAL * LATENT_SPATIAL  # 512
NUM_ACTIONS = 8
CLICK_COORD_DIM = 2
ENSEMBLE_SIZE = 3
REPLAY_BUFFER_SIZE = 2000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TRAIN_EVERY_N_STEPS = 4
MIN_BUFFER_FOR_TRAINING = 64
MAX_FRAME_CACHE = 500
TARGET_TAU = 0.995


def grid_to_onehot(grid: np.ndarray) -> torch.Tensor:
    """Convert integer grid (H, W) with values 0-15 to one-hot (16, H, W)."""
    grid_t = torch.from_numpy(grid.astype(np.int64))
    onehot = F.one_hot(grid_t, num_classes=NUM_COLORS)  # (H, W, 16)
    return onehot.permute(2, 0, 1).float()  # (16, H, W)


def batch_grids_to_onehot(grids: np.ndarray) -> torch.Tensor:
    """Convert batch of integer grids (B, H, W) to one-hot (B, 16, H, W)."""
    grids_t = torch.from_numpy(grids.astype(np.int64))
    onehot = F.one_hot(grids_t, num_classes=NUM_COLORS)  # (B, H, W, 16)
    return onehot.permute(0, 3, 1, 2).float()  # (B, 16, H, W)


class SpatialEncoder(nn.Module):
    """CNN encoder: 64x64 one-hot (16 channels) → 8x8xC spatial features.

    16-channel input means color permutation = channel permutation,
    which is a natural augmentation for convnets. All color boundaries
    have identical mathematical magnitude regardless of permutation.

    Output is NOT normalized here — L2 normalization is applied after
    flattening in the training and inference code.
    """

    def __init__(self, channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 32, 3, stride=2, padding=1),  # -> 32x32x32
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64x16x16
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=2, padding=1),  # -> Cx8x8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (batch, C, 8, 8)


class ResidualDynamicsHead(nn.Module):
    """Single MLP head: predicts Δz from [z_flat, action, click_xy].

    Residual prediction: z_{t+1} = z_t + Δz.
    Zero-init output layer provides correct prior that z_{t+1} ≈ z_t.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        input_dim = latent_dim + num_actions + CLICK_COORD_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_flat: torch.Tensor, action_oh: torch.Tensor,
                click_xy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_flat, action_oh, click_xy], dim=-1)
        return self.net(x)


class EnsembleDynamics(nn.Module):
    """Ensemble of residual MLP heads. Uncertainty = prediction disagreement."""

    def __init__(self, n_heads: int = ENSEMBLE_SIZE, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.heads = nn.ModuleList([ResidualDynamicsHead(latent_dim) for _ in range(n_heads)])
        self.n_heads = n_heads

    def forward(self, z_flat: torch.Tensor, action_oh: torch.Tensor,
                click_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        deltas = torch.stack([h(z_flat, action_oh, click_xy) for h in self.heads])
        mean_delta = deltas.mean(dim=0)
        uncertainty = deltas.std(dim=0).mean(dim=-1)
        return mean_delta, uncertainty

    def per_head_deltas(self, z_flat: torch.Tensor, action_oh: torch.Tensor,
                        click_xy: torch.Tensor) -> list[torch.Tensor]:
        return [h(z_flat, action_oh, click_xy) for h in self.heads]


class ReplayBuffer:
    """Stores (frame_grid, action_id, click_xy, next_frame_grid) as raw integers."""

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE) -> None:
        self._buffer: deque[tuple[np.ndarray, int, tuple[float, float], np.ndarray]] = deque(maxlen=max_size)

    def add(self, frame: np.ndarray, action_id: int, next_frame: np.ndarray,
            click_xy: tuple[float, float] = (0.0, 0.0)) -> None:
        # Store raw integer grids (0-15), NOT float-encoded
        self._buffer.append((frame.copy(), action_id, click_xy, next_frame.copy()))

    def sample(self, n: int) -> list[tuple[np.ndarray, int, tuple[float, float], np.ndarray]]:
        indices = np.random.choice(len(self._buffer), size=min(n, len(self._buffer)), replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class EnsembleWorldModel(WorldModel):
    """Residual MLP ensemble world model with L2-normalized latents.

    - 16-channel one-hot input (color-permutation invariant)
    - L2-normalized latent space (bounds MSE, prevents scale explosion)
    - Raw integer frames in replay buffer (augmentation per-batch)
    - Graph model as exact cache for visited transitions
    - Polyak-averaged target encoder for stable training
    """

    def __init__(self, graph: StateGraph, pretrained_path: str | None = None) -> None:
        self.graph = graph
        self.device = torch.device("cpu")

        self.encoder = SpatialEncoder().to(self.device)
        self.target_encoder = SpatialEncoder().to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.dynamics = EnsembleDynamics().to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            lr=LEARNING_RATE,
        )
        self.replay = ReplayBuffer()
        self._step_count = 0
        self._train_count = 0
        self.avg_loss: float = 0.0
        self._loss_history: deque[float] = deque(maxlen=100)

        self._latent_cache: dict[int, torch.Tensor] = {}
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_uncertainty: float = 0.01

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
            self.target_encoder.load_state_dict(ckpt["encoder"])
        if "dynamics" in ckpt:
            self.dynamics.load_state_dict(ckpt["dynamics"])

    def cache_frame(self, state_hash: int, grid: np.ndarray) -> None:
        """Cache raw integer frame grid (0-15) for replay and encoding."""
        if state_hash not in self._frame_cache:
            self._frame_cache[state_hash] = grid.copy()
            while len(self._frame_cache) > MAX_FRAME_CACHE:
                self._frame_cache.popitem(last=False)

    def _encode(self, grid: np.ndarray) -> torch.Tensor:
        """Encode a single frame grid to L2-normalized spatial latent."""
        onehot = grid_to_onehot(grid).unsqueeze(0).to(self.device)  # (1, 16, 64, 64)
        with torch.no_grad():
            z = self.encoder(onehot)  # (1, C, 8, 8)
            z_flat = z.view(1, -1)
            z_norm = F.normalize(z_flat, p=2, dim=1)
        return z_norm  # (1, LATENT_DIM), unit norm

    def _get_latent(self, state_hash: int) -> torch.Tensor | None:
        if state_hash in self._latent_cache:
            return self._latent_cache[state_hash]
        if state_hash in self._frame_cache:
            z = self._encode(self._frame_cache[state_hash])
            self._latent_cache[state_hash] = z
            return z
        return None

    def _make_action_input(self, action: GameAction,
                           click_xy: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        action_oh = torch.zeros(1, NUM_ACTIONS, device=self.device)
        action_oh[0, action.value] = 1.0
        if click_xy is not None:
            click_t = torch.tensor([[click_xy[0] / 63.0, click_xy[1] / 63.0]], device=self.device)
        else:
            click_t = torch.zeros(1, CLICK_COORD_DIM, device=self.device)
        return action_oh, click_t

    # --- WorldModel interface ---

    def predict(self, state_hash: int, action: GameAction) -> WorldModelPrediction:
        node = self.graph.nodes.get(state_hash)
        if node and action in node.transitions:
            next_hash = node.transitions[action]
            changes = action not in node.no_change
            return WorldModelPrediction(next_hash, 1.0, changes, "graph")

        z = self._get_latent(state_hash)
        if z is None:
            return WorldModelPrediction(None, 0.0, None, "unknown")

        action_oh, click_t = self._make_action_input(action)
        with torch.no_grad():
            _, uncertainty = self.dynamics(z, action_oh, click_t)

        unc_val = uncertainty.item()
        self._max_uncertainty = max(self._max_uncertainty, unc_val)
        confidence = max(0.0, 1.0 - unc_val / self._max_uncertainty)
        return WorldModelPrediction(None, confidence, None, "neural")

    def update(self, state_hash: int, action: GameAction,
               next_state_hash: int, diff: FrameDiff,
               click_xy: tuple[int, int] | None = None) -> None:
        self.graph.add_transition(state_hash, action, next_state_hash, diff.num_changed)

        if state_hash in self._frame_cache and next_state_hash in self._frame_cache:
            click = (click_xy[0] / 63.0, click_xy[1] / 63.0) if click_xy else (0.0, 0.0)
            self.replay.add(
                self._frame_cache[state_hash],
                action.value,
                self._frame_cache[next_state_hash],
                click,
            )

        self._step_count += 1
        if (self._step_count % TRAIN_EVERY_N_STEPS == 0
                and len(self.replay) >= MIN_BUFFER_FOR_TRAINING):
            self._train_step()

    def get_uncertainty(self, state_hash: int, action: GameAction) -> float:
        node = self.graph.nodes.get(state_hash)
        if node and action in node.transitions:
            return 0.0

        z = self._get_latent(state_hash)
        if z is None:
            return 1.0

        action_oh, click_t = self._make_action_input(action)
        with torch.no_grad():
            _, uncertainty = self.dynamics(z, action_oh, click_t)

        unc_val = uncertainty.item()
        self._max_uncertainty = max(self._max_uncertainty, unc_val)
        return min(1.0, unc_val / self._max_uncertainty)

    def get_frontier_actions(self, state_hash: int) -> list[tuple[GameAction, float]]:
        result = [(a, self.get_uncertainty(state_hash, a)) for a in self.graph.available_actions]
        result.sort(key=lambda x: -x[1])
        return result

    def get_click_uncertainties(self, state_hash: int,
                                candidates: list[tuple[int, int]]) -> list[float]:
        z = self._get_latent(state_hash)
        if z is None:
            return [1.0] * len(candidates)

        z_expanded = z.expand(len(candidates), -1)
        action_oh = torch.zeros(len(candidates), NUM_ACTIONS, device=self.device)
        action_oh[:, GameAction.ACTION6.value] = 1.0
        click_t = torch.tensor(
            [[cx / 63.0, cy / 63.0] for cx, cy in candidates], device=self.device,
        )

        with torch.no_grad():
            _, uncertainties = self.dynamics(z_expanded, action_oh, click_t)

        raw = uncertainties.cpu().numpy().tolist()
        self._max_uncertainty = max(self._max_uncertainty, max(raw) if raw else 0.01)
        return [min(1.0, u / self._max_uncertainty) for u in raw]

    # --- Training ---

    def _train_step(self) -> None:
        batch = self.replay.sample(BATCH_SIZE)

        # Load raw integer grids and convert to one-hot
        frames = np.stack([t[0] for t in batch])
        actions = [t[1] for t in batch]
        clicks = np.array([t[2] for t in batch], dtype=np.float32)
        next_frames = np.stack([t[3] for t in batch])

        frames_t = batch_grids_to_onehot(frames).to(self.device)
        next_frames_t = batch_grids_to_onehot(next_frames).to(self.device)
        clicks_t = torch.from_numpy(clicks).to(self.device)

        action_oh = torch.zeros(len(batch), NUM_ACTIONS, device=self.device)
        for i, a in enumerate(actions):
            action_oh[i, a] = 1.0

        # Encode current frames (with gradients) and L2-normalize
        z = self.encoder(frames_t)
        z_flat = z.view(len(batch), -1)
        z_norm = F.normalize(z_flat, p=2, dim=1)

        # Encode target frames with Polyak encoder (no gradients), L2-normalize
        with torch.no_grad():
            z_next = self.target_encoder(next_frames_t)
            z_next_flat = z_next.view(len(batch), -1)
            z_next_norm = F.normalize(z_next_flat, p=2, dim=1)

        # Target delta on unit sphere — max MSE bounded to 4.0
        target_delta = z_next_norm - z_norm.detach()

        # Per-head loss on residual prediction
        head_deltas = self.dynamics.per_head_deltas(z_norm, action_oh, clicks_t)
        loss = sum(F.mse_loss(d, target_delta) for d in head_deltas) / len(head_deltas)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        # Polyak update target encoder
        with torch.no_grad():
            for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                tp.data.mul_(TARGET_TAU).add_(p.data, alpha=1 - TARGET_TAU)

        loss_val = loss.item()
        self._loss_history.append(loss_val)
        self.avg_loss = sum(self._loss_history) / len(self._loss_history)
        self._train_count += 1

        self._latent_cache.clear()
