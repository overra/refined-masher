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
_CALIBRATION_STEPS = 8
_MIN_REPLAY_FOR_MPC = 40  # need this many transitions before MPC planning
_MPC_HORIZON = 4  # lookahead steps for planning
_ACTOR_LR = 3e-3
_CRITIC_LR = 3e-3
_GAMMA = 0.95
_BETA_INIT = 0.8
_BETA_DECAY = 0.99
_BETA_MIN = 0.1
_ENTROPY_COEF = 0.02
_TRAIN_ACTOR_EVERY = 8
_TRAIN_BATCH_SIZE = 8


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

        # Phase 1: Explorer drives for spatial calibration (first 8 steps)
        # The world model is already training from Explorer's actions
        if self._level_steps <= _CALIBRATION_STEPS:
            self._using_explorer = True
            self._mode = "calibrate"
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"cal:{self._explorer.last_reason}"
            return action

        # --- Actor-Critic action selection ---
        self._using_explorer = False
        self.selected_click_target = None

        # Get latent for current state
        z = world_model._get_latent(state_hash)
        if z is None:
            self._using_explorer = True
            action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            self.last_reason = f"no-latent:{self._explorer.last_reason}"
            return action

        z_flat = z.view(1, -1)

        # Before world model is ready: Explorer + uncertainty override
        replay_size = len(world_model.replay) if hasattr(world_model, 'replay') else 0
        if replay_size < _MIN_REPLAY_FOR_MPC:
            explorer_action = self._explorer.select_action(
                state_hash, frame, objects, ui_state,
                world_model, episode, graph, cross_level, context,
            )
            # With probability β, override with highest-uncertainty action
            if random.random() < self._beta:
                unc_vals = [(a, world_model.get_uncertainty(state_hash, a))
                            for a in self._available_actions]
                unc_vals.sort(key=lambda x: -x[1])
                if unc_vals and unc_vals[0][1] > 0.01:
                    action = unc_vals[0][0]
                    self._mode = "ac-unc"
                    self.last_reason = f"unc-pick({action.name} u={unc_vals[0][1]:.2f})"
                    self._beta = max(_BETA_MIN, self._beta * _BETA_DECAY)
                    return action
            self._mode = "ac-expl"
            self.last_reason = f"expl+unc:{self._explorer.last_reason}"
            self._beta = max(_BETA_MIN, self._beta * _BETA_DECAY)
            return explorer_action

        # --- MPC: Model-Predictive Control ---
        # Get Explorer's recommendation as the default
        explorer_action = self._explorer.select_action(
            state_hash, frame, objects, ui_state,
            world_model, episode, graph, cross_level, context,
        )

        # Run MPC to see if any action is clearly better
        mpc_action, mpc_score, mpc_reason = self._mpc_plan(
            z_flat, world_model, graph,
        )

        # Also score the Explorer's preferred action via MPC
        explorer_idx = None
        for i, a in enumerate(self._available_actions):
            if a == explorer_action:
                explorer_idx = i
                break

        # Use MPC action only if it's significantly better than Explorer's choice
        # This prevents MPC from overriding good heuristics with noise
        action = explorer_action
        reason = f"expl+mpc:{self._explorer.last_reason}"
        mode = "mpc-expl"

        if mpc_action != explorer_action:
            # MPC wants something different — use it if score is meaningfully positive
            if mpc_score > 0.1:
                action = mpc_action
                reason = mpc_reason
                mode = "mpc"

        self._beta = max(_BETA_MIN, self._beta * _BETA_DECAY)

        energy = ui_state.energy if ui_state else None
        if energy is not None and energy < 0.3:
            mode = mode + "-low"
        self._mode = mode

        # Train actor-critic periodically
        if self._total_steps % _TRAIN_ACTOR_EVERY == 0:
            self._train_imagination(world_model, graph)

        self.last_reason = reason
        return action

    def _mpc_plan(
        self,
        z_flat: torch.Tensor,
        world_model: WorldModel,
        graph: StateGraph,
    ) -> tuple[GameAction, float, str]:
        """Model-Predictive Control: score each opening action by imagined trajectory.

        For each available action:
        1. Predict next latent via world model dynamics
        2. Continue for _MPC_HORIZON steps using actor policy
        3. Score trajectory by: novelty (uncertainty) + change magnitude + win bonus
        4. Return the opening action of the best trajectory.

        All forward passes are batched for efficiency.
        """
        n_actions = len(self._available_actions)
        if n_actions == 0:
            return GameAction.ACTION1, 0.0, "mpc-no-actions"

        horizon = _MPC_HORIZON

        # Build batched first-step: one copy of z per available action
        # Shape: (n_actions, latent_dim)
        z_batch = z_flat.expand(n_actions, -1).clone()

        # First actions: one-hot for each available action
        first_action_oh = torch.zeros(n_actions, self._num_actions, device=self._device)
        for i, a in enumerate(self._available_actions):
            first_action_oh[i, a.value] = 1.0
        click_batch = torch.zeros(n_actions, 2, device=self._device)

        # Track scores per trajectory
        scores = torch.zeros(n_actions, device=self._device)

        with torch.no_grad():
            # Step 1: apply each opening action
            delta, unc = world_model.dynamics(z_batch, first_action_oh, click_batch)
            z_batch = F.normalize(z_batch + delta, p=2, dim=1)
            change_mag = delta.abs().mean(dim=-1)

            # Score step 1
            scores += unc * self._beta  # novelty (weighted by exploration drive)
            scores += change_mag * 0.5  # something happened

            # Check if any trajectory reaches a known win state in the graph
            # (we can't directly check latent→hash, but graph wins are handled by Explorer)

            # Steps 2..horizon: use actor policy for subsequent actions
            for t in range(1, horizon):
                # Actor picks next action for each trajectory
                logits = self._actor(z_batch)
                mask = torch.full((n_actions, self._num_actions), float('-inf'), device=self._device)
                for a in self._available_actions:
                    mask[:, a.value] = 0.0
                probs = F.softmax(logits + mask, dim=-1)
                next_actions = probs.argmax(dim=-1)  # greedy for planning

                next_action_oh = torch.zeros(n_actions, self._num_actions, device=self._device)
                next_action_oh.scatter_(1, next_actions.unsqueeze(1), 1.0)
                click_batch = torch.zeros(n_actions, 2, device=self._device)

                delta, unc = world_model.dynamics(z_batch, next_action_oh, click_batch)
                z_batch = F.normalize(z_batch + delta, p=2, dim=1)
                change_mag = delta.abs().mean(dim=-1)

                discount = _GAMMA ** t
                scores += discount * unc * self._beta
                scores += discount * change_mag * 0.5

            # Final state value from critic (if available)
            if self._critic is not None:
                # Use the actor's preferred action for value estimation
                final_logits = self._actor(z_batch)
                final_probs = F.softmax(final_logits + mask, dim=-1)
                final_action_oh = torch.zeros(n_actions, self._num_actions, device=self._device)
                final_action_oh.scatter_(1, final_probs.argmax(dim=-1).unsqueeze(1), 1.0)
                terminal_value = self._critic(z_batch, final_action_oh)
                scores += (_GAMMA ** horizon) * terminal_value

        # Pick best opening action
        best_idx = scores.argmax().item()
        best_action = self._available_actions[best_idx]
        best_score = scores[best_idx].item()

        # Format reason string
        action_scores = " ".join(
            f"{a.name}={scores[i]:.2f}" for i, a in enumerate(self._available_actions)
        )
        reason = f"mpc(best={best_action.name} s={best_score:.2f} h={horizon} | {action_scores})"

        return best_action, best_score, reason

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

        # Only do imagination rollouts when model has enough data
        if len(world_model.replay) < _MIN_REPLAY_FOR_MPC:
            return

        batch = world_model.replay.sample(min(_TRAIN_BATCH_SIZE, len(world_model.replay)))
        if not batch:
            return

        # Encode starting frames (one-hot, L2-normalized)
        import numpy as np
        from remash.world_model.ensemble_model import batch_grids_to_onehot
        frames = np.stack([t[0] for t in batch])
        frames_t = batch_grids_to_onehot(frames).to(self._device)
        with torch.no_grad():
            z = world_model.encoder(frames_t)
            z_flat = F.normalize(z.view(len(batch), -1), p=2, dim=1)

        # Imagined rollout
        total_actor_loss = torch.tensor(0.0, device=self._device)
        total_critic_loss = torch.tensor(0.0, device=self._device)
        gamma_t = 1.0

        current_z = z_flat

        for t in range(_MPC_HORIZON):
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
                # Bootstrap: use the actor's own preferred action for V(s')
                next_logits = self._actor(next_z)
                next_probs = F.softmax(next_logits, dim=-1)
                next_action_oh = torch.zeros(len(batch), self._num_actions, device=self._device)
                next_action_idx = next_probs.argmax(dim=-1)
                next_action_oh.scatter_(1, next_action_idx.unsqueeze(1), 1.0)
                v_next = self._critic(next_z, next_action_oh)
                target = reward + _GAMMA * v_next

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
