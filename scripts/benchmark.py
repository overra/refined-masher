"""Run ReMash agent on multiple games and report scores.

Usage:
    uv run python scripts/benchmark.py                    # baseline on all games
    uv run python scripts/benchmark.py --efe              # EFE+neural on all games
    uv run python scripts/benchmark.py --games ls20,ft09  # specific games
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
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per level")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--efe", action="store_true", help="Use EFE + neural world model")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.WARNING)

    arcade = arc_agi.Arcade()
    available = arcade.get_environments()

    if args.games:
        game_ids = [g.strip() for g in args.games.split(",")]
    else:
        game_ids = [e.game_id for e in available]

    print(f"Running {'EFE+neural' if args.efe else 'baseline'} on {len(game_ids)} games...")
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
            # Try neural, gracefully fall back to graph (matches Kaggle code path)
            try:
                from remash.world_model.neural_model import NeuralWorldModel
                use_neural = True
            except ImportError:
                use_neural = False
        else:
            policy = ExplorerPolicy(max_steps_per_level=args.max_steps)
            use_neural = False

        try:
            agent = ReMashAgent(policy=policy, use_neural=use_neural)
            result = agent.play_game(env, game_id=game_id)
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
        print(f"{'TOTAL':<20} {total_levels}/{total_possible:>7} {'':>8} {avg_score:>7.1%}")
    print(f"{'='*65}")
    print(f"\nScoring games: {len(scoring)}/{len(results)}")
    print(f"Zero-score games: {len(zero)}/{len(results)}")
    if errors:
        print(f"Errors: {len(errors)} ({', '.join(errors)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/game)")


if __name__ == "__main__":
    main()
