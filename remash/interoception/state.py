"""Internal state module. Stub for Phase 5."""

from __future__ import annotations

import numpy as np


class InteroceptiveState:
    """Phase 5: Small CfC with slow time constants.

    Inputs: world_model_loss, steps_remaining, learning_rate.
    Outputs: modulation vector for policy precision weighting.

    Phase 1: no-op.
    """

    def get_modulation(self) -> np.ndarray | None:
        return None
