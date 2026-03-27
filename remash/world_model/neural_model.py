"""Phase 2: CfC-based neural world model.

CNN encoder maps frames to latent vectors.
CfC ensemble predicts next-latent from (current-latent, action).
Graph model runs underneath as exact cache for visited transitions.
Uncertainty = ensemble disagreement (continuous, 0-1).

Trains online from observed transitions during gameplay.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from arcengine import GameAction

from remash.memory.state_graph import StateGraph
from remash.perception.frame import FrameDiff
from remash.world_model.base import WorldModel, WorldModelPrediction

if TYPE_CHECKING:
    from remash.perception.frame import Frame

# --- Hyperparameters ---
LATENT_DIM = 64
NUM_ACTIONS = 8  # ACTION0-7 (one-hot encoding)
ENSEMBLE_SIZE = 3
REPLAY_BUFFER_SIZE = 2000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TRAIN_EVERY_N_STEPS = 4  # train every N steps to amortize cost
MIN_BUFFER_FOR_TRAINING = 64


class FrameEncoder(nn.Module):
    """Small CNN: 64x64 color-index grid → LATENT_DIM vector.

    Input: (batch, 1, 64, 64) float tensor with values 0-15 normalized to 0-1.
    Output: (batch, LATENT_DIM) latent vector.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        # 3 conv layers with stride-2 downsampling: 64→32→16→8
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16x32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> 32x8x8
            nn.ReLU(),
        )
        # 32*8*8 = 2048 → latent_dim
        self.fc = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


CLICK_COORD_DIM = 2  # normalized (x/63, y/63) for click actions


class DynamicsHead(nn.Module):
    """Single CfC head: predicts next latent from (current_latent, action_onehot, click_xy).

    Uses CfC for its ability to learn continuous-time dynamics with fast adaptation.
    Click coordinates are appended for ACTION6; zero for other actions.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_actions: int = NUM_ACTIONS,
        cfc_units: int = 128,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + num_actions + CLICK_COORD_DIM
        wiring = AutoNCP(units=cfc_units, output_size=latent_dim)
        self.cfc = CfC(input_size=input_dim, units=wiring, batch_first=True)

    def forward(
        self,
        z: torch.Tensor,
        action_onehot: torch.Tensor,
        click_xy: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim)
            action_onehot: (batch, num_actions)
            click_xy: (batch, 2) normalized click coords, or None (zeros)
            hx: optional hidden state from previous step
        Returns:
            z_next: (batch, latent_dim) predicted next latent
            hx_new: updated hidden state
        """
        if click_xy is None:
            click_xy = torch.zeros(z.size(0), CLICK_COORD_DIM, device=z.device)
        x = torch.cat([z, action_onehot, click_xy], dim=-1)
        x = x.unsqueeze(1)  # (batch, 1, input_dim) — single timestep
        out, hx_new = self.cfc(x, hx)
        z_next = out.squeeze(1)  # (batch, latent_dim)
        return z_next, hx_new


class EnsembleDynamics(nn.Module):
    """Ensemble of CfC heads. Uncertainty = disagreement between heads."""

    def __init__(
        self,
        n_heads: int = ENSEMBLE_SIZE,
        latent_dim: int = LATENT_DIM,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            DynamicsHead(latent_dim, num_actions) for _ in range(n_heads)
        ])
        self.n_heads = n_heads
        self.latent_dim = latent_dim

    def forward(
        self,
        z: torch.Tensor,
        action_onehot: torch.Tensor,
        click_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred: (batch, latent_dim) mean prediction across heads
            uncertainty: (batch,) disagreement score (std of predictions)
        """
        preds = []
        for head in self.heads:
            z_next, _ = head(z, action_onehot, click_xy)
            preds.append(z_next)
        stacked = torch.stack(preds, dim=0)  # (n_heads, batch, latent_dim)
        mean_pred = stacked.mean(dim=0)
        # Uncertainty = mean std across latent dimensions
        uncertainty = stacked.std(dim=0).mean(dim=-1)
        return mean_pred, uncertainty

    def per_head_predictions(
        self,
        z: torch.Tensor,
        action_onehot: torch.Tensor,
        click_xy: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return individual head predictions for per-head loss computation."""
        return [head(z, action_onehot, click_xy)[0] for head in self.heads]


# --- Replay buffer ---

class ReplayBuffer:
    """Stores (frame, action, click_xy, next_frame) transitions for training."""

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE) -> None:
        self._buffer: deque[tuple[np.ndarray, int, tuple[float, float], np.ndarray]] = deque(maxlen=max_size)

    def add(
        self,
        frame_grid: np.ndarray,
        action_id: int,
        next_frame_grid: np.ndarray,
        click_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._buffer.append((frame_grid.copy(), action_id, click_xy, next_frame_grid.copy()))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch. Returns (frames, action_onehots, click_xys, next_frames)."""
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        frames, actions, clicks, next_frames = [], [], [], []
        for i in indices:
            f, a, xy, nf = self._buffer[i]
            frames.append(f)
            actions.append(a)
            clicks.append(xy)
            next_frames.append(nf)

        frame_tensor = _grids_to_tensor(np.array(frames))
        next_tensor = _grids_to_tensor(np.array(next_frames))
        action_tensor = _actions_to_onehot(actions)
        click_tensor = torch.tensor(clicks, dtype=torch.float32)
        return frame_tensor, action_tensor, click_tensor, next_tensor

    def __len__(self) -> int:
        return len(self._buffer)


def _grids_to_tensor(grids: np.ndarray) -> torch.Tensor:
    """Convert (batch, 64, 64) uint8 grids to (batch, 1, 64, 64) float tensors."""
    t = torch.from_numpy(grids.astype(np.float32)) / 15.0  # normalize 0-15 to 0-1
    return t.unsqueeze(1)  # add channel dim


def _actions_to_onehot(action_ids: list[int]) -> torch.Tensor:
    t = torch.zeros(len(action_ids), NUM_ACTIONS)
    for i, a in enumerate(action_ids):
        if 0 <= a < NUM_ACTIONS:
            t[i, a] = 1.0
    return t


# --- Neural World Model ---

class NeuralWorldModel(WorldModel):
    """Phase 2 world model: CfC ensemble with graph cache.

    - Known transitions (in graph): uncertainty = 0.0, prediction is exact.
    - Unknown transitions: uncertainty = ensemble disagreement.
    - Trains online from a replay buffer of observed transitions.
    """

    def __init__(
        self,
        graph: StateGraph,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.graph = graph

        # Neural components
        self.encoder = FrameEncoder().to(self.device)
        self.dynamics = EnsembleDynamics().to(self.device)

        # Optimizer for joint encoder + dynamics training
        params = list(self.encoder.parameters()) + list(self.dynamics.parameters())
        self.optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

        # Replay buffer and training state
        self.replay = ReplayBuffer()
        self._step_count: int = 0
        self._train_losses: deque[float] = deque(maxlen=100)

        # Latent cache: state_hash → latent vector (detached)
        self._latent_cache: dict[int, torch.Tensor] = {}

        # Frame cache: state_hash → grid (for encoding on demand)
        self._frame_cache: dict[int, np.ndarray] = {}

    def cache_frame(self, state_hash: int, grid: np.ndarray) -> None:
        """Call each step so we can encode frames for prediction."""
        self._frame_cache[state_hash] = grid

    def _encode(self, grid: np.ndarray) -> torch.Tensor:
        """Encode a single frame grid to latent vector."""
        t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 15.0
        t = t.to(self.device)
        with torch.no_grad():
            z = self.encoder(t)
        return z.squeeze(0)  # (LATENT_DIM,)

    def _get_latent(self, state_hash: int) -> torch.Tensor | None:
        """Get or compute latent for a state hash."""
        if state_hash in self._latent_cache:
            return self._latent_cache[state_hash]
        if state_hash in self._frame_cache:
            z = self._encode(self._frame_cache[state_hash])
            self._latent_cache[state_hash] = z
            return z
        return None

    # --- WorldModel ABC ---

    def predict(self, state_hash: int, action: GameAction) -> WorldModelPrediction:
        # Graph cache first (exact)
        next_hash = self.graph.get_transition(state_hash, action)
        if next_hash is not None:
            frame_changes = action not in self.graph.nodes[state_hash].no_change
            return WorldModelPrediction(
                predicted_next_hash=next_hash,
                confidence=1.0,
                predicted_frame_changes=frame_changes,
                source="graph",
            )

        # Neural prediction
        z = self._get_latent(state_hash)
        if z is None:
            return WorldModelPrediction(None, 0.0, None, "unknown")

        action_oh = torch.zeros(1, NUM_ACTIONS, device=self.device)
        action_oh[0, action.value] = 1.0

        with torch.no_grad():
            _, uncertainty = self.dynamics(z.unsqueeze(0), action_oh)

        confidence = max(0.0, 1.0 - float(uncertainty.item()))
        return WorldModelPrediction(
            predicted_next_hash=None,  # can't predict hash, only latent
            confidence=confidence,
            predicted_frame_changes=True,  # assume change for unknown
            source="neural",
        )

    def update(
        self,
        state_hash: int,
        action: GameAction,
        next_state_hash: int,
        diff: FrameDiff,
        click_xy: tuple[int, int] | None = None,
    ) -> None:
        # Always update graph (exact cache)
        self.graph.add_transition(state_hash, action, next_state_hash, diff_pixels=diff.num_changed)

        # Normalize click coordinates for the replay buffer
        norm_xy = (0.0, 0.0)
        if click_xy is not None:
            norm_xy = (click_xy[0] / 63.0, click_xy[1] / 63.0)

        # Add to replay buffer if we have both frames
        if state_hash in self._frame_cache and next_state_hash in self._frame_cache:
            self.replay.add(
                self._frame_cache[state_hash],
                action.value,
                self._frame_cache[next_state_hash],
                click_xy=norm_xy,
            )

        # Invalidate latent cache for changed state (encoder updates during training)
        self._latent_cache.pop(state_hash, None)
        self._latent_cache.pop(next_state_hash, None)

        # Online training
        self._step_count += 1
        if (self._step_count % TRAIN_EVERY_N_STEPS == 0
                and len(self.replay) >= MIN_BUFFER_FOR_TRAINING):
            loss = self._train_step()
            self._train_losses.append(loss)

    def get_uncertainty(self, state_hash: int, action: GameAction) -> float:
        # Graph = certain
        if self.graph.get_transition(state_hash, action) is not None:
            return 0.0

        # Neural uncertainty
        z = self._get_latent(state_hash)
        if z is None:
            return 1.0

        action_oh = torch.zeros(1, NUM_ACTIONS, device=self.device)
        action_oh[0, action.value] = 1.0

        with torch.no_grad():
            _, uncertainty = self.dynamics(z.unsqueeze(0), action_oh)

        return min(1.0, float(uncertainty.item()))

    def get_click_uncertainties(
        self,
        state_hash: int,
        candidates: list[tuple[int, int]],
    ) -> list[float]:
        """Get uncertainty for ACTION6 at each (x,y) candidate in a single batch.

        Returns a list of uncertainty values, one per candidate.
        Much more efficient than calling get_uncertainty per candidate.
        """
        z = self._get_latent(state_hash)
        if z is None:
            return [1.0] * len(candidates)

        n = len(candidates)
        z_batch = z.unsqueeze(0).expand(n, -1)
        action_oh = torch.zeros(n, NUM_ACTIONS, device=self.device)
        action_oh[:, 6] = 1.0  # ACTION6
        click_xy = torch.tensor(
            [(x / 63.0, y / 63.0) for x, y in candidates],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            _, uncertainties = self.dynamics(z_batch, action_oh, click_xy)

        return [min(1.0, float(u)) for u in uncertainties.tolist()]

    def get_frontier_actions(self, state_hash: int) -> list[tuple[GameAction, float]]:
        result: list[tuple[GameAction, float]] = []
        for action in self.graph.available_actions:
            u = self.get_uncertainty(state_hash, action)
            result.append((action, u))
        result.sort(key=lambda x: -x[1])
        return result

    # --- Training ---

    def _train_step(self) -> float:
        """One training step on a mini-batch from the replay buffer."""
        self.encoder.train()
        self.dynamics.train()

        frames, action_oh, click_xy, next_frames = self.replay.sample(BATCH_SIZE)
        frames = frames.to(self.device)
        action_oh = action_oh.to(self.device)
        click_xy = click_xy.to(self.device)
        next_frames = next_frames.to(self.device)

        # Encode current and target frames
        z_current = self.encoder(frames)
        with torch.no_grad():
            z_target = self.encoder(next_frames)

        # Per-head loss (each head trained on the same data)
        head_preds = self.dynamics.per_head_predictions(z_current, action_oh, click_xy)
        loss = torch.tensor(0.0, device=self.device)
        for pred in head_preds:
            loss = loss + F.mse_loss(pred, z_target)
        loss = loss / len(head_preds)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        self.encoder.eval()
        self.dynamics.eval()

        return float(loss.item())

    @property
    def avg_loss(self) -> float:
        if not self._train_losses:
            return 0.0
        return sum(self._train_losses) / len(self._train_losses)
