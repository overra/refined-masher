"""Run ReMash agent on a single game.

Usage:
    uv run python scripts/play.py --game ls20                  # Phase 1 reactive baseline
    uv run python scripts/play.py --game ls20 --neural         # Phase 2 neural world model
    uv run python scripts/play.py --game ls20 --efe            # Phase 3 EFE active inference
"""

import argparse
import logging

import arc_agi

from remash.agent import ReMashAgent
from remash.policy.explorer import ExplorerPolicy
from remash.utils.logging import setup_logging, logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReMash agent on a single game")
    parser.add_argument("--game", required=True, help="Game ID (e.g., ls20)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per level")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--neural", action="store_true", help="Use CfC neural world model (Phase 2)")
    parser.add_argument("--efe", action="store_true", help="Use EFE active inference policy (Phase 3, implies --neural)")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    arcade = arc_agi.Arcade()
    logger.info("Available games: %d", len(arcade.available_environments))

    env = arcade.make(args.game)
    if env is None:
        logger.error("Could not create environment for game: %s", args.game)
        return

    if args.efe:
        from remash.policy.efe import EFEPolicy
        policy = EFEPolicy()
        use_neural = True
        logger.info("Using EFE active inference policy + neural world model")
    else:
        policy = ExplorerPolicy(max_steps_per_level=args.max_steps)
        use_neural = args.neural

    agent = ReMashAgent(policy=policy, use_neural=use_neural)
    result = agent.play_game(env, game_id=args.game)

    print(f"\n{'='*50}")
    print(f"Game: {result.game_id}")
    print(f"Levels: {result.levels_completed}/{result.win_levels}")
    print(f"Total steps: {result.total_steps}")
    print(f"Score: {result.score:.2%}")
    for i, steps in enumerate(result.level_steps):
        print(f"  Level {i}: {steps} steps")
    print(f"Graph: {result.graph_stats}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
