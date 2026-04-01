"""Actor-Critic policy with ensemble disagreement as intrinsic reward.

Replaces EFE with model-based RL:
- Actor MLP outputs action probabilities from latent state
- Critic MLP estimates value of (state, action) pairs
- Intrinsic reward = ensemble disagreement (novel states are rewarding)
- Extrinsic reward = large frame changes + level completion
- Training via imagined rollouts through the world model

Falls back to Explorer for the first ~15 steps while the world model
bootstraps, and for BFS win-path execution.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcengine import GameAction

from remash.policy.base import Policy
from remash.policy.explorer import ExplorerPolicy
from remash.utils.logging import logger

if TYPE_CHECKING:
    import numpy as np

    from remash.memory.cross_level import CrossLevelMemory
    from remash.memory.episode import EpisodeBuffer
    from remash.memory.state_graph import StateGraph
    from remash.perception.frame import Frame
    from remash.perception.objects import GridObject
    from remash.perception.ui import UIState
    from remash.world_model.base import WorldModel

# --- Hyperparameters ---
_BOOTSTRAP_STEPS = 15  # use Explorer for this many steps per level
_MIN_REPLAY_FOR_AC = 30  # need this many transitions before AC training
_IMAGINATION_HORIZON = 5  # rollout steps for imagined training
_ACTOR_LR = 1e-3
_CRITIC_LR = 1e-3
_GAMMA = 0.95  # discount factor
_BETA_INIT = 1.0  # initial intrinsic reward weight
_BETA_DECAY = 0.995  # per-step decay of intrinsic reward weight
_BETA_MIN = 0.1  # minimum intrinsic weight (always some curiosity)
_ENTROPY_COEF = 0.01  # entropy regularization for actor
_TRAIN_ACTOR_EVERY = 4  # train actor-critic every N steps
_LARGE_DIFF_THRESHOLD = 20  # extrinsic reward for diffs above this


class ActorMLP(nn.Module):
    """Maps latent state to action probabilities."""

    def __init__(self, latent_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Returns logits (not probabilities) for each action."""
        return self.net(z_flat)


class CriticMLP(nn.Module):
    """Maps (latent state, action) to scalar value estimate."""

    def __init__(self, latent_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, z_flat: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate."""
        x = torch.cat([z_flat, action_oh], dim=-1)
        return self.net(x).squeeze(-1)


class ActorCriticPolicy(Policy):
    """Actor-critic with ensemble disagreement intrinsic reward.

    Delegates to Explorer for:
    - First _BOOTSTRAP_STEPS steps (spatial calibration + initial data)
    - BFS win-path execution (Explorer is better at this)

    After bootstrap, uses the actor network to select actions.
    Trains via imagined rollouts through the world model.
    """

    def __init__(self) -> None:
        self._explorer = ExplorerPolicy()
        self.spatial = self._explorer.spatial
        self.last_reason: str = ""
        self._mode: str = "bootstrap"
        self._level_steps: int = 0
        self._total_steps: int = 0
        self._using_explorer: bool = True
        self.selected_click_target: tuple[int, int] | None = None

        # β decays over time: high early (explore), low late (exploit)
        self._beta: float = _BETA_INIT

        # Deferred initialization — need to know action space first
        self._actor: ActorMLP | None = None
        self._critic: CriticMLP | None = None
        self._actor_opt: torch.optim.Optimizer | None = None
        self._critic_opt: torch.optim.Optimizer | None = None
        self._num_actions: int = 0
        self._latent_dim: int = 0
        self._available_actions: list[GameAction] = []
        self._initialized: bool = False
        self._device = torch.device("cpu")

        # Track last diff for extrinsic reward
        self._last_diff: int = 0

    def _ensure_init(self, graph: StateGraph, world_model: WorldModel) -> bool:
        """Initialize actor-critic networks once we know the action space.

        Returns True if we have a neural world model to work with.
        """
        if self._initialized:
            return self._actor is not None

        try:
            from remash.world_model.ensemble_model import EnsembleWorldModel
            if not isinstance(world_model, EnsembleWorldModel):
                return False
        except ImportError:
            return False

        self._available_actions = graph.available_actions
        self._num_actions = 8  # fixed: ACTION0-ACTION7, consistent with world model

        # Get latent dim from world model
        from remash.world_model.ensemble_model import LATENT_DIM
        self._latent_dim = LATENT_DIM

        self._actor = ActorMLP(self._latent_dim, self._num_actions).to(self._device)
        self._critic = CriticMLP(self._latent_dim, self._num_actions).to(self._device)
        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=_ACTOR_LR)
        self._critic_opt = torch.optim.Adam(self._critic.parameters(), lr=_CRITIC_LR)
        self._initialized = True
        logger.info("Actor-critic initialized: %d actions, %d latent dim", self._num_actions, self._latent_dim)
        return True

    def on_level_start(self, level_num: int) -> None:
        self._explorer.on_level_start(level_num)
        self._level_steps = 0
        self._beta = _BETA_INIT
        self._mode = "bootstrap"

    def on_level_complete(self, level_num: int) -> None:
        self._explorer.on_level_complete(level_num)

    def on_step_result(self, state_hash: int, diff_pixels: int) -> None:
        self._explorer.on_step_result(state_hash, diff_pixels)
        self._last_diff = diff_pixels

    def select_action(
        self,
        state_hash: int,
        frame: Frame,
        objects: list[GridObject],
        ui_state: UIState | None,
        world_model: WorldModel,
        episode: EpisodeBuffer,
        graph: StateGraph,
        cross_level: CrossLevelMemory,
        context: np.ndarray | None = None,
    ) -> GameAction:
        self._level_steps += 1
        self._total_steps += 1

        has_neural = self._ensure_init(graph, world_model)

        # No neural model → pure Explorer
        if not has_neural:
            self._using_explorer = True
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self._mode = self._explorer._mode
            self.last_reason = self._explorer.last_reason
            return action

        # Bootstrap phase: let Explorer calibrate and gather initial data
        if self._level_steps <= _BOOTSTRAP_STEPS:
            self._using_explorer = True
            self._mode = "bootstrap"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"boot:{self._explorer.last_reason}"
            return action

        # Always defer to Explorer for known win paths
        win_path = graph.get_path_to_win(state_hash)
        if win_path:
            self._using_explorer = True
            self._mode = "ac-win"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"win:{self._explorer.last_reason}"
            return action

        # Not enough data yet — keep exploring
        if hasattr(world_model, 'replay') and len(world_model.replay) < _MIN_REPLAY_FOR_AC:
            self._using_explorer = True
            self._mode = "gathering"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"gather:{self._explorer.last_reason}"
            return action

        # --- Actor-Critic action selection ---
        self._using_explorer = False
        self.selected_click_target = None

        # Get latent for current state
        z = world_model._get_latent(state_hash)
        if z is None:
            # Can't encode — fall back to Explorer
            self._using_explorer = True
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"no-latent:{self._explorer.last_reason}"
            return action

        z_flat = z.view(1, -1)

        # Compute action probabilities from actor
        with torch.no_grad():
            logits = self._actor(z_flat)
            # Mask unavailable actions
            mask = torch.full((1, self._num_actions), float('-inf'), device=self._device)
            for a in self._available_actions:
                mask[0, a.value] = 0.0
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)

            # Add exploration bonus: boost actions with high ensemble uncertainty
            uncertainties = torch.zeros(1, self._num_actions, device=self._device)
            for a in self._available_actions:
                unc = world_model.get_uncertainty(state_hash, a)
                uncertainties[0, a.value] = unc

            # Blend: (1-β) * actor_probs + β * uncertainty_probs
            unc_sum = uncertainties.sum()
            if unc_sum > 1e-8:
                unc_probs = uncertainties / unc_sum
            else:
                # All uncertainties zero (all known) — uniform over available
                unc_probs = torch.zeros_like(uncertainties)
                for a in self._available_actions:
                    unc_probs[0, a.value] = 1.0 / len(self._available_actions)
            blended = (1 - self._beta) * probs + self._beta * unc_probs
            # Ensure valid distribution
            blended = blended.clamp(min=0.0)
            blended_sum = blended.sum()
            if blended_sum < 1e-8:
                # Fallback: uniform over available actions
                for a in self._available_actions:
                    blended[0, a.value] = 1.0 / len(self._available_actions)
            else:
                blended = blended / blended_sum

        # Sample action
        action_idx = torch.multinomial(blended, 1).item()
        action = GameAction.from_id(action_idx)

        # For click actions, delegate target selection to ClickTargetManager
        # (it already tracks responsive positions and cycles targets)

        # Decay β
        self._beta = max(_BETA_MIN, self._beta * _BETA_DECAY)

        # Energy-aware mode label
        energy = ui_state.energy if ui_state else None
        if energy is not None and energy < 0.3:
            self._mode = "ac-low"
        elif self._beta > 0.5:
            self._mode = "ac-expl"
        else:
            self._mode = "ac-expl" if self._beta > 0.3 else "ac"

        # Train periodically via imagined rollouts
        if self._total_steps % _TRAIN_ACTOR_EVERY == 0:
            self._train_imagination(world_model, graph)

        self.last_reason = (
            f"ac(β={self._beta:.2f}"
            f" p={blended[0, action_idx]:.2f}"
            f" unc={uncertainties[0, action_idx]:.2f})"
        )
        return action

    def _train_imagination(self, world_model: WorldModel, graph: StateGraph) -> None:
        """Train actor-critic via imagined rollouts through the world model."""
        if self._actor is None or self._critic is None:
            return

        try:
            from remash.world_model.ensemble_model import EnsembleWorldModel
            if not isinstance(world_model, EnsembleWorldModel):
                return
        except ImportError:
            return

        # Sample a batch of starting states from the replay buffer
        if len(world_model.replay) < _MIN_REPLAY_FOR_AC:
            return

        batch = world_model.replay.sample(min(16, len(world_model.replay)))
        if not batch:
            return

        # Encode starting frames
        import numpy as np
        frames = np.stack([t[0] for t in batch]).astype(np.float32) / 15.0
        frames_t = torch.from_numpy(frames).unsqueeze(1).to(self._device)
        with torch.no_grad():
            z = world_model.encoder(frames_t)  # (batch, C, 8, 8)
        z_flat = z.view(len(batch), -1)

        # Imagined rollout
        total_actor_loss = torch.tensor(0.0, device=self._device)
        total_critic_loss = torch.tensor(0.0, device=self._device)
        gamma_t = 1.0

        current_z = z_flat

        for t in range(_IMAGINATION_HORIZON):
            # Actor produces action distribution
            logits = self._actor(current_z)
            mask = torch.full((len(batch), self._num_actions), float('-inf'), device=self._device)
            for a in self._available_actions:
                mask[:, a.value] = 0.0
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs + 1e-8)
            action_indices = dist.sample()
            log_probs = dist.log_prob(action_indices)
            entropy = dist.entropy()

            # One-hot actions
            action_oh = torch.zeros(len(batch), self._num_actions, device=self._device)
            action_oh.scatter_(1, action_indices.unsqueeze(1), 1.0)

            # Click coordinates (zeros for non-click actions)
            click_t = torch.zeros(len(batch), 2, device=self._device)

            # Predict next state via world model dynamics
            with torch.no_grad():
                mean_delta, uncertainties = world_model.dynamics(current_z, action_oh, click_t)
                next_z = current_z + mean_delta

            # Compute reward
            # Intrinsic: ensemble disagreement (curiosity)
            intrinsic_reward = uncertainties  # (batch,)
            # Extrinsic: predicted magnitude of change (proxy for meaningful interaction)
            delta_magnitude = mean_delta.abs().mean(dim=-1)  # (batch,)
            extrinsic_reward = (delta_magnitude > 0.01).float() * 0.1

            reward = extrinsic_reward + self._beta * intrinsic_reward

            # Critic value estimates
            v_current = self._critic(current_z.detach(), action_oh)
            with torch.no_grad():
                # Bootstrap value of next state (max over actions)
                best_v_next = torch.full((len(batch),), 0.0, device=self._device)
                for a in self._available_actions:
                    a_oh = torch.zeros(len(batch), self._num_actions, device=self._device)
                    a_oh[:, a.value] = 1.0
                    v_a = self._critic(next_z, a_oh)
                    best_v_next = torch.max(best_v_next, v_a)
                target = reward + _GAMMA * best_v_next

            # Critic loss: MSE to TD target
            critic_loss = F.mse_loss(v_current, target)

            # Actor loss: policy gradient with entropy bonus
            advantage = (target - v_current).detach()
            actor_loss = -(log_probs * advantage).mean() - _ENTROPY_COEF * entropy.mean()

            total_actor_loss = total_actor_loss + gamma_t * actor_loss
            total_critic_loss = total_critic_loss + gamma_t * critic_loss
            gamma_t *= _GAMMA

            current_z = next_z

        # Update networks
        self._actor_opt.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)
        self._actor_opt.step()

        self._critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_opt.step()
