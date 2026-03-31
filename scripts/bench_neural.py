"""Benchmark neural model inference throughput.

Tests CfC ensemble at multiple sizes + MLP baseline.
Reports FPS for encode, predict, uncertainty on CPU and MPS.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from ncps.torch import CfC
from ncps.wirings import AutoNCP

LATENT_DIM = 64
ACTION_DIM = 8
CLICK_DIM = 2
INPUT_DIM = LATENT_DIM + ACTION_DIM + CLICK_DIM  # 74
WARMUP = 50
ITERS = 500


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CfCHead(nn.Module):
    def __init__(self, units, latent_dim=LATENT_DIM):
        super().__init__()
        wiring = AutoNCP(units, latent_dim)
        self.cfc = CfC(INPUT_DIM, wiring, batch_first=True)

    def forward(self, x):
        # x: (batch, input_dim) -> add time dim -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        out, _ = self.cfc(x)
        return out[:, -1, :]  # (batch, latent_dim)


class MLPHead(nn.Module):
    def __init__(self, hidden, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def bench_fps(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    return iters / elapsed


def run_config(name, num_heads, head_factory, device):
    encoder = FrameEncoder().to(device).eval()
    heads = [head_factory().to(device).eval() for _ in range(num_heads)]

    frame = torch.randn(1, 1, 64, 64, device=device)
    action_oh = torch.zeros(1, ACTION_DIM, device=device)
    action_oh[0, 2] = 1.0
    click = torch.tensor([[32.0, 32.0]], device=device) / 64.0

    with torch.no_grad():
        # Encode
        def encode():
            return encoder(frame)

        # Single head predict
        z = encoder(frame)
        inp = torch.cat([z, action_oh, click], dim=-1)

        def predict_one():
            return heads[0](inp)

        # Full uncertainty (all heads + disagreement)
        def uncertainty():
            preds = torch.stack([h(inp) for h in heads])
            return preds.std(dim=0).mean()

        # Full pipeline: encode + predict all heads + uncertainty
        def full_pipeline():
            z = encoder(frame)
            inp = torch.cat([z, action_oh, click], dim=-1)
            preds = torch.stack([h(inp) for h in heads])
            return preds.mean(dim=0), preds.std(dim=0).mean()

        enc_fps = bench_fps(encode)
        pred_fps = bench_fps(predict_one)
        unc_fps = bench_fps(uncertainty)
        full_fps = bench_fps(full_pipeline)

    print(f"  {name:40s} | enc:{enc_fps:6.0f} | pred:{pred_fps:6.0f} | unc:{unc_fps:6.0f} | full:{full_fps:6.0f} FPS")
    return full_fps


def main():
    configs = [
        ("Current: 3×128-unit CfC",    3, lambda: CfCHead(128)),
        ("Medium:  2×64-unit CfC",      2, lambda: CfCHead(64)),
        ("Small:   2×32-unit CfC",      2, lambda: CfCHead(32)),
        ("Baseline: 2×128-unit MLP",    2, lambda: MLPHead(128)),
        ("Baseline: 2×64-unit MLP",     2, lambda: MLPHead(64)),
    ]

    for device_name in ["cpu", "mps"]:
        if device_name == "mps" and not torch.backends.mps.is_available():
            print(f"\n{'='*90}")
            print(f"  MPS not available, skipping")
            continue

        device = torch.device(device_name)
        print(f"\n{'='*90}")
        print(f"  Device: {device_name.upper()}")
        print(f"  {'Config':<40s} | {'Encode':>6s} | {'Predict':>6s} | {'Uncert':>6s} | {'Full':>6s}")
        print(f"  {'-'*40}-+--------+--------+--------+--------")

        for name, num_heads, factory in configs:
            try:
                run_config(name, num_heads, factory, device)
            except Exception as e:
                print(f"  {name:40s} | ERROR: {e}")

    print(f"\n{'='*90}")
    print("  Threshold: 100+ FPS = viable, 30-100 = tight, <30 = too slow")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
