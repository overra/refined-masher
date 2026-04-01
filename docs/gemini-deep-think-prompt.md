# Pre-Training Stability Analysis — Gemini Deep Think

## Context

I'm building a model-based RL agent for ARC-AGI-3, an interactive reasoning benchmark with 64x64 grid worlds (16 discrete colors, turn-based). The agent has a CNN encoder that maps frames to latent vectors and an MLP ensemble that predicts dynamics in latent space.

I'm pre-training the encoder and dynamics across many games with color augmentation to learn general grid-world physics. **The online training loss explodes from 0.001 to 300+ over 100 episodes.** I need to understand why and how to fix it.

---

## 1. Pre-Training Loop Code

Each episode: pick a random game, generate a random color permutation, play the game for up to 2000 steps. The world model trains online during the episode. Between episodes, encoder+dynamics weights are copied back to shared parameters.

```python
for ep in range(100):
    game_id = random.choice(train_games)
    color_perm = make_color_permutation()  # random permutation of [0..15]

    graph = StateGraph()
    wm = AugmentedEnsembleWorldModel(graph, color_perm)

    # Copy shared weights into this episode's world model
    wm.encoder.load_state_dict(shared_encoder.state_dict())
    wm.target_encoder.load_state_dict(shared_target_encoder.state_dict())
    wm.dynamics.load_state_dict(shared_dynamics.state_dict())
    wm.optimizer = torch.optim.Adam(
        list(wm.encoder.parameters()) + list(wm.dynamics.parameters()),
        lr=3e-4,
    )

    # Play game — world model trains online via _train_step() every 4 steps
    result = agent.play_game(env, external_world_model=wm, competition_mode=True)

    # Copy trained weights back to shared model
    shared_encoder.load_state_dict(wm.encoder.state_dict())
    shared_target_encoder.load_state_dict(wm.target_encoder.state_dict())
    shared_dynamics.load_state_dict(wm.dynamics.state_dict())
```

Color permutation is applied when frames are cached:

```python
def make_color_permutation() -> np.ndarray:
    perm = np.arange(16, dtype=np.uint8)
    np.random.shuffle(perm)
    return perm

class AugmentedEnsembleWorldModel(EnsembleWorldModel):
    def __init__(self, graph, color_perm):
        super().__init__(graph)
        self._color_perm = color_perm

    def cache_frame(self, state_hash, grid):
        augmented = self._color_perm[grid]  # apply permutation to all pixels
        super().cache_frame(state_hash, augmented)
```

---

## 2. Encoder Architecture

```python
class SpatialEncoder(nn.Module):
    """CNN: 64x64 single-channel → 8x8x8 spatial features"""
    def __init__(self, channels=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> 16x32x32
            nn.LayerNorm([16, 32, 32]),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32x16x16
            nn.LayerNorm([32, 16, 16]),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, stride=2, padding=1),  # -> 8x8x8
        )

    def forward(self, x):
        return self.conv(x)
```

**Input preprocessing:** Raw grid has integer values 0-15. Converted to float and divided by 15:
```python
frame_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 15.0
```

So the CNN sees a single-channel 64x64 image with continuous values in [0, 1]. Color index 0 → 0.0, color index 15 → 1.0.

---

## 3. Training Step (runs every 4 game steps)

```python
def _train_step(self):
    batch = self.replay.sample(32)

    frames = np.stack([t[0] for t in batch]).astype(np.float32) / 15.0
    next_frames = np.stack([t[3] for t in batch]).astype(np.float32) / 15.0

    frames_t = torch.from_numpy(frames).unsqueeze(1)      # (32, 1, 64, 64)
    next_frames_t = torch.from_numpy(next_frames).unsqueeze(1)

    # Encode current frames WITH gradients
    z = self.encoder(frames_t)             # (32, 8, 8, 8)
    z_flat = z.view(32, -1)               # (32, 512)

    # Encode next frames with Polyak-averaged target encoder, NO gradients
    with torch.no_grad():
        z_next_target = self.target_encoder(next_frames_t)
        z_next_target_flat = z_next_target.view(32, -1)

    # Target delta: what changed in latent space
    target_delta = z_next_target_flat - z_flat.detach()

    # Each MLP head predicts Δz independently
    head_deltas = [head(z_flat, action_oh, clicks) for head in dynamics.heads]
    loss = mean(MSE(delta, target_delta) for delta in head_deltas)

    optimizer.zero_grad()
    loss.backward()          # gradients flow into encoder AND dynamics
    clip_grad_norm(1.0)
    optimizer.step()

    # Polyak update: target_encoder ← 0.995 * target_encoder + 0.005 * encoder
    for p, tp in zip(encoder.parameters(), target_encoder.parameters()):
        tp.data.mul_(0.995).add_(p.data, alpha=0.005)
```

Key: `target_delta = z_next_target_flat - z_flat.detach()`. The online encoder produces z_flat (with gradients). The target encoder (Polyak average) produces z_next_target. The target is the difference. The dynamics MLP heads learn to predict this delta.

---

## 4. Loss Trajectory from Failed Run (100 episodes with offline rehearsal)

Episodes 1-10: Online loss drops nicely
```
Ep  Game             Loss      Notes
 1  ka59             0.18456
 2  m0r0             0.08286
 3  wa30             0.24948
 4  sc25             0.13879
 5  s5i5             0.02407
 6  ka59             0.00068   ← converged
 7  cn04             0.00226
 8  lp85             0.00069
 9  sp80             0.00078
10  wa30             0.00135
```

Episodes 11-30: Loss starts climbing
```
11  re86             1.90252   ← first spike
12  ka59             0.06396
13  cd82             3.43115
16  ls20             5.85544
18  sp80             9.79298
22  s5i5            14.10871
27  cd82            23.89717
28  ls20            25.81765
```

Episodes 30-60: Loss grows to 100+
```
40  wa30           116.06964
44  sp80           196.52834
54  ar25            92.70181
56  cd82           122.73584
58  g50t           104.45998
60  cd82           268.84077   ← peak
```

Episodes 60-100: Oscillates wildly (1 to 300)
```
73  vc33             1.83721   ← occasional low
76  cn04           141.58297
80  re86           102.48719
82  ar25           308.82448   ← worst
88  vc33             1.94551
100 sp80            29.48975
```

**Rehearsal loss** (offline training on accumulated buffer from all episodes):
```
After ep 10: 0.005  (low, working)
After ep 20: 3.326
After ep 25: 14.801
After ep 30: 74.985
After ep 40: 104.110
After ep 50: 123.153
After ep 75: 199.214
After ep 100: 145.238
```

---

## 5. What Changed Between Run 1 (Failed) and Run 2 (Currently Running)

**Removed:** All offline rehearsal. The `PersistentReplayBuffer` and `offline_rehearsal()` function were deleted entirely. No cross-episode replay.

**Kept:** Everything else identical. Online training within each episode still runs (every 4 steps from the episode's replay buffer). Encoder + dynamics weights still persist across episodes. Color permutation still changes every episode. New optimizer created each episode.

**Hypothesis:** The offline rehearsal was the primary cause because it mixed color-permuted frames from different episodes. But the online training loss ALSO climbed (0.001 → 30-55 by episode 100), even before rehearsal ran. This suggests the problem isn't just rehearsal.

---

## Questions

1. **Will the encoder converge to a color-invariant representation** when the color mapping changes every episode, given that encoder weights persist across episodes? The input is `pixel_value / 15.0`, so color index 5 in episode 1 might map to the same float value as color index 12 in episode 2 (after permutation). The encoder sees the same float but it means a different structural element. Can a CNN learn to be invariant to this when training across many permutations?

2. **Is MSE loss on Δz stable** when the encoder that defines the latent space is shifting under the dynamics model? The target is `target_delta = target_encoder(next_frame) - encoder(current_frame).detach()`. As the encoder changes episode to episode (from different color distributions), the meaning of "latent distance" shifts. Does Polyak averaging (τ=0.995) provide sufficient stability, or does the 2000-step episode with aggressive online training (every 4 steps = 500 updates per episode) cause the encoder to drift too far from the target encoder?

3. **What's the correct way to pre-train** a visual encoder across multiple episodes with color augmentation?
   - Should the encoder be frozen after N episodes?
   - Should there be a separate color-invariant training objective (e.g., contrastive loss: same structure different colors → same latent)?
   - Should color permutation be applied per-batch (within episode) rather than per-episode?
   - Should the input be a 16-channel one-hot encoding instead of a single float channel? (This makes color permutation a channel permutation, which is a more natural augmentation for a CNN.)

4. **Are there other failure modes in this code?**
   - The optimizer is re-created each episode (`torch.optim.Adam(...)`). This resets Adam's momentum buffers. Is that harmful?
   - The replay buffer is per-episode (max 2000 transitions, cleared between episodes). With 500 training steps sampling from 2000 transitions, late training steps are heavily overfitting to the same data. Is this causing the encoder to specialize to the current episode's color mapping?
   - The residual dynamics heads are initialized with zeros. After many episodes of training, are they drifting away from the residual initialization in a harmful way?

5. **Concrete recommendation:** Given this architecture, what specific changes would you make to achieve stable pre-training loss across 100+ episodes with color augmentation? Provide code-level changes, not just principles.
