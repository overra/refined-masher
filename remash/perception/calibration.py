"""Per-game adaptive calibration from observed frame diffs.

Instead of hardcoded thresholds, the agent observes the distribution of
diff sizes during the first N steps and sets thresholds at natural gaps.

Typical diff distributions have two clusters:
  - UI noise: 0-3px (cursor blink, counter tick, energy bar)
  - Real changes: 30-200px (player movement, object interaction)

The no-change threshold belongs in the gap between these clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_CALIBRATION_STEPS = 20  # observe this many transitions before committing


@dataclass
class GameCalibration:
    """Per-game learned parameters, set during the first ~20 steps."""

    no_change_threshold: int = 4  # default until calibrated
    responsive_click_threshold: int = 4  # same as no_change for clicks
    energy_budget: int = 50  # fallback energy budget (steps per life)
    calibrated: bool = False


class DiffCalibrator:
    """Observes diff magnitudes and finds the adaptive no-change threshold.

    Call `observe(diff_pixels)` each step. After `_CALIBRATION_STEPS`
    observations, `calibrate()` returns a `GameCalibration` with learned
    thresholds.
    """

    def __init__(self) -> None:
        self._diffs: list[int] = []
        self._calibration: GameCalibration | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def calibration(self) -> GameCalibration:
        if self._calibration is not None:
            return self._calibration
        return GameCalibration()  # defaults

    def observe(self, diff_pixels: int) -> GameCalibration:
        """Record a diff observation. Returns current calibration.

        Automatically calibrates after enough observations.
        """
        self._diffs.append(diff_pixels)

        if not self.is_calibrated and len(self._diffs) >= _CALIBRATION_STEPS:
            self._calibrate()

        return self.calibration

    def observe_game_over(self, steps_in_life: int) -> None:
        """Record how many steps a life lasted. Updates energy budget."""
        if self._calibration is None:
            self._calibrate()
        # Use observed life length as energy budget (with margin)
        self._calibration.energy_budget = max(20, steps_in_life)

    def _calibrate(self) -> None:
        """Analyze collected diffs and set thresholds at the natural gap."""
        if not self._diffs:
            self._calibration = GameCalibration()
            return

        sorted_diffs = sorted(self._diffs)

        # Find the largest gap between consecutive sorted diff values.
        # The threshold belongs in this gap.
        threshold = self._find_gap_threshold(sorted_diffs)

        self._calibration = GameCalibration(
            no_change_threshold=threshold,
            responsive_click_threshold=threshold,
            calibrated=True,
        )

    def _find_gap_threshold(self, sorted_diffs: list[int]) -> int:
        """Find the natural gap between UI noise and real changes.

        Strategy: find the largest multiplicative gap between consecutive
        sorted diff values. The threshold is the midpoint of that gap.

        Example: [0, 1, 1, 2, 2, 3, 50, 52, 55, 100]
        Gaps: 1, 0, 1, 0, 1, 47, 2, 3, 45
        Largest gap: 3→50 (gap=47)
        Threshold: (3+50)//2 = 26
        """
        if len(sorted_diffs) < 2:
            return 4  # default

        # Filter to unique values for gap analysis
        unique = sorted(set(sorted_diffs))
        if len(unique) < 2:
            # All diffs are the same value
            return max(unique[0] + 1, 4)

        # Find the largest absolute gap
        best_gap = 0
        best_low = 0
        best_high = 0
        for i in range(len(unique) - 1):
            gap = unique[i + 1] - unique[i]
            if gap > best_gap:
                best_gap = gap
                best_low = unique[i]
                best_high = unique[i + 1]

        if best_gap < 4:
            # No clear gap — all diffs are similar. Use a conservative default.
            # If all diffs are small (<10), everything is noise → low threshold
            # If all diffs are large (>20), everything is real → low threshold
            median = unique[len(unique) // 2]
            if median < 10:
                return max(median + 1, 4)
            return 4

        # Threshold is just above the lower cluster
        return best_low + max(best_gap // 4, 1)
