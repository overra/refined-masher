"""Run ReMash agent on multiple games and report scores.

Usage:
    uv run python scripts/benchmark.py                          # single-life baseline
    uv run python scripts/benchmark.py --efe                    # EFE policy (Kaggle code path)
    uv run python scripts/benchmark.py --competition-mode       # multi-life, 2000 action cap
    uv run python scripts/benchmark.py --games ls20,ft09        # specific games
"""

import argparse
import logging
import time

import arc_agi

from remash.agent import ReMashAgent, GameResult
from remash.policy.explorer import ExplorerPolicy
from remash.utils.logging import setup_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ReMash agent")
    parser.add_argument("--games", default="", help="Comma-separated game IDs (empty = all)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max total steps per game")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--efe", action="store_true", help="Use EFE policy (matches Kaggle code path)")
    parser.add_argument("--competition-mode", action="store_true",
                        help="Multi-life mode: GAME_OVER triggers RESET, not exit. "
                             "Matches actual competition behavior.")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.WARNING)

    arcade = arc_agi.Arcade()
    available = arcade.get_environments()

    if args.games:
        game_ids = [g.strip() for g in args.games.split(",")]
    else:
        game_ids = [e.game_id for e in available]

    mode_label = "competition" if args.competition_mode else "single-life"
    if args.efe:
        mode_label += "+EFE"
    print(f"Running {mode_label} on {len(game_ids)} games (max {args.max_steps} steps)...")
    results: list[GameResult] = []
    errors: list[str] = []
    t0 = time.time()

    for i, game_id in enumerate(game_ids):
        env = arcade.make(game_id)
        if env is None:
            errors.append(game_id)
            continue

        if args.efe:
            from remash.policy.efe import EFEPolicy
            policy = EFEPolicy()
            try:
                from remash.world_model.neural_model import NeuralWorldModel
                use_neural = True
            except ImportError:
                use_neural = False
        else:
            policy = ExplorerPolicy(max_steps_per_level=args.max_steps)
            use_neural = False

        try:
            agent = ReMashAgent(
                policy=policy,
                use_neural=use_neural,
                max_total_steps=args.max_steps,
            )
            result = agent.play_game(
                env, game_id=game_id,
                competition_mode=args.competition_mode,
            )
            results.append(result)
            marker = "*" if result.levels_completed > 0 else " "
            print(
                f"  [{i+1:2d}/{len(game_ids)}] {marker} {game_id:<20s}"
                f" {result.levels_completed}/{result.win_levels} levels"
                f" {result.total_steps:4d} steps"
                f" {result.score:6.1%}",
            )
        except Exception as e:
            errors.append(f"{game_id}: {e}")
            print(f"  [{i+1:2d}/{len(game_ids)}]   {game_id:<20s} ERROR: {e}")

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Game':<20} {'Levels':>10} {'Steps':>8} {'Score':>8}")
    print(f"{'-'*65}")

    scoring = []
    zero = []
    for r in sorted(results, key=lambda r: -r.score):
        marker = " *" if r.score > 0 else ""
        print(f"{r.game_id:<20} {r.levels_completed}/{r.win_levels:>7} {r.total_steps:>8} {r.score:>7.1%}{marker}")
        if r.score > 0:
            scoring.append(r)
        else:
            zero.append(r)

    print(f"{'-'*65}")
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        total_levels = sum(r.levels_completed for r in results)
        total_possible = sum(r.win_levels for r in results)
        total_steps = sum(r.total_steps for r in results)
        print(f"{'TOTAL':<20} {total_levels}/{total_possible:>7} {total_steps:>8} {avg_score:>7.1%}")
    print(f"{'='*65}")
    print(f"\nScoring games: {len(scoring)}/{len(results)}")
    print(f"Zero-score games: {len(zero)}/{len(results)}")
    if errors:
        print(f"Errors: {len(errors)} ({', '.join(errors)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/game)")


if __name__ == "__main__":
    main()
