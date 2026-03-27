# ReMash — Refined Masher

ARC-AGI-3 interactive reasoning agent. Achieves **8.43% RHAE** across 25 public games, scoring on 4 games (sp80, lp85, vc33, ls20).

## Architecture

Phased architecture with no game-specific heuristics:

- **Perception**: 64x64 color-index frame parsing, BFS flood-fill object detection, adaptive UI/energy bar detection, player tracking via movement diffs
- **Memory**: Directed state graph with UI-masked hashing, episode buffer, cross-level knowledge transfer (responsive click colors)
- **World Model**: CfC neural network (continuous-time) with CNN frame encoder, ensemble of 3 heads for uncertainty via disagreement, graph model as exact cache
- **Policy**: Expected Free Energy (EFE) active inference — epistemic (uncertainty) + pragmatic (diff magnitude) value computation, softmax action selection with adaptive precision, explorer fallback for bootstrap
- **Click Targeting**: Object centroid priority with stale demotion, grid sampling fallback, cross-level color priors

## Quick Start

```bash
uv sync --extra neural          # install all deps including torch/ncps
uv run python scripts/play.py --game vc33 --efe    # run on a single game
uv run python scripts/benchmark.py --efe            # run on all public games
```

## Competition Agent

The `agents/templates/remash_agent.py` in the [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) repo wraps this package for the official competition format:

```bash
cd ../ARC-AGI-3-Agents
uv run main.py --agent=remashagent --game=vc33    # single game
uv run main.py --agent=remashagent                 # all games (scorecard)
```

## Results

| Game | RHAE Score | Levels |
|------|-----------|--------|
| sp80 | 130.0% | 1/6 |
| lp85 | 47.3% | 1/8 |
| vc33 | 17.4% | 1/7 |
| ls20 | 16.0% | 1/7 |
| **Average** | **8.43%** | **4/25 games** |

## License

MIT-0 (MIT No Attribution)
