"""
run_benchmark.py
----------------
CLI entry point for running Connect Four LLM-MCTS benchmarks.

Quick start examples:

  # DeepSeek cloud vs random (Ollama disabled)
  python run_benchmark.py --agent1-backend deepseek --agent1-model deepseek-chat \
      --agent2-backend random --games 5

  # Ollama local deepseek vs ollama local llama
  python run_benchmark.py \
      --agent1-backend ollama --agent1-model deepseek-r1:7b --agent1-mode hybrid \
      --agent2-backend ollama --agent2-model llama3.2:3b --agent2-mode direct \
      --games 10 --iterations 100

  # Anthropic vs DeepSeek
  python run_benchmark.py \
      --agent1-backend anthropic --agent1-model claude-haiku-4-5-20251001 \
      --agent2-backend deepseek  --agent2-model deepseek-chat \
      --games 3 --iterations 50

  # Single agent self-play
  python run_benchmark.py --agent1-backend ollama --agent1-model deepseek-r1:7b \
      --agent2-backend ollama --agent2-model deepseek-r1:7b --games 5

Evaluation modes:
  direct   - LLM estimates win probability (replaces rollout entirely)
  prior    - LLM biases move selection weights
  guided   - LLM picks every move during rollout
  hybrid   - LLM for first k plies, random after (default, k=2)
  random   - Pure random rollout (no LLM, for baseline comparison)
"""

import argparse
import sys
import os

# Make sure scripts/ is in path when run from repo root
sys.path.insert(0, os.path.dirname(__file__))

from llm_provider import LLMConfig
from game_adapter import MCTSAgent, run_benchmark, PLAYER1, PLAYER2


class RandomAgent:
    """Baseline agent using pure random rollout (no LLM)."""
    def __init__(self, name, player_id, iterations=500):
        from game_adapter import Connect4Game
        from llm_provider import LLMConfig
        import math, random

        self.name = name or f"Random-P{player_id}"
        self.player_id = player_id
        self.iterations = iterations
        self.eval_mode = "random"
        self.hybrid_k = 0
        self.llm_config = LLMConfig(backend="none", model="random")
        self.move_records = []
        self._game = Connect4Game()
        self._provider = None

    def get_move(self, board):
        import random, math
        from game_adapter import Connect4Game, Connect4Game as G

        game = self._game
        legal = game.get_legal_moves(board)

        # Simple UCT with random rollout
        scores = {c: 0.0 for c in legal}
        visits = {c: 0 for c in legal}

        for _ in range(self.iterations):
            col = random.choice(legal)
            state = game.make_move(board, col, self.player_id)
            result = self._random_rollout(state, 3 - self.player_id)
            scores[col] += result
            visits[col] += 1

        best = max(legal, key=lambda c: scores[c] / max(visits[c], 1))
        return best

    def _random_rollout(self, board, player):
        import random
        game = self._game
        state = [row[:] for row in board]
        current = player
        for _ in range(42):
            legal = game.get_legal_moves(state)
            if not legal:
                return 0.5
            col = random.choice(legal)
            state = game.make_move(state, col, current)
            if game.check_win(state, current):
                return 1.0 if current == self.player_id else 0.0
            current = 3 - current
        return 0.5

    def _ensure_provider(self):
        pass  # no-op for random agent

    @property
    def provider_stats(self):
        return {"backend": "random", "model": "random", "api_calls": 0, "cache_hits": 0}


def make_agent(player_id: int, args, prefix: str) -> MCTSAgent | RandomAgent:
    backend  = getattr(args, f"{prefix}_backend")
    model    = getattr(args, f"{prefix}_model")
    mode     = getattr(args, f"{prefix}_mode")
    name     = getattr(args, f"{prefix}_name")
    api_key  = getattr(args, f"{prefix}_api_key", None)
    base_url = getattr(args, f"{prefix}_base_url", None)

    if backend == "random":
        return RandomAgent(name or f"Random-P{player_id}", player_id, args.iterations)

    cfg = LLMConfig(
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        cache=not args.no_cache,
    )
    return MCTSAgent(
        name=name or f"{backend.title()}-{model}-{mode}",
        player_id=player_id,
        llm_config=cfg,
        eval_mode=mode,
        hybrid_k=args.hybrid_k,
        iterations=args.iterations,
        time_limit=args.time_limit,
        use_prior=(mode == "prior"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Connect Four LLM-MCTS Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Agent 1
    g1 = parser.add_argument_group("Agent 1 (Player X)")
    g1.add_argument("--agent1-backend",  default="ollama",
                    choices=["anthropic","deepseek","ollama","random"],
                    help="LLM backend for agent 1")
    g1.add_argument("--agent1-model",    default="deepseek-r1:7b",
                    help="Model name/tag for agent 1")
    g1.add_argument("--agent1-mode",     default="hybrid",
                    choices=["direct","prior","guided","hybrid","random"],
                    help="Evaluation mode for agent 1")
    g1.add_argument("--agent1-name",     default=None, help="Display name for agent 1")
    g1.add_argument("--agent1-api-key",  default=None, help="API key (or use env var)")
    g1.add_argument("--agent1-base-url", default=None, help="Custom base URL")

    # Agent 2
    g2 = parser.add_argument_group("Agent 2 (Player O)")
    g2.add_argument("--agent2-backend",  default="random",
                    choices=["anthropic","deepseek","ollama","random"],
                    help="LLM backend for agent 2")
    g2.add_argument("--agent2-model",    default="",
                    help="Model name/tag for agent 2")
    g2.add_argument("--agent2-mode",     default="hybrid",
                    choices=["direct","prior","guided","hybrid","random"],
                    help="Evaluation mode for agent 2")
    g2.add_argument("--agent2-name",     default=None, help="Display name for agent 2")
    g2.add_argument("--agent2-api-key",  default=None, help="API key (or use env var)")
    g2.add_argument("--agent2-base-url", default=None, help="Custom base URL")

    # Run settings
    rg = parser.add_argument_group("Run settings")
    rg.add_argument("--games",       type=int,   default=5,   help="Number of games to play")
    rg.add_argument("--iterations",  type=int,   default=150, help="MCTS iterations per move")
    rg.add_argument("--time-limit",  type=float, default=0,   help="Seconds per move (0=use iterations)")
    rg.add_argument("--hybrid-k",    type=int,   default=2,   help="LLM plies for hybrid mode")
    rg.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    rg.add_argument("--timeout",     type=int,   default=30,  help="HTTP timeout in seconds")
    rg.add_argument("--no-cache",    action="store_true",     help="Disable position caching")
    rg.add_argument("--output-dir",  default="reports",       help="Output directory for reports")
    rg.add_argument("--no-boards",   action="store_true",     help="Don't log board states (smaller logs)")

    args = parser.parse_args()

    print("=" * 60)
    print("  Connect Four LLM-MCTS Benchmark")
    print("=" * 60)

    agent1 = make_agent(PLAYER1, args, "agent1")
    agent2 = make_agent(PLAYER2, args, "agent2")

    print(f"\n  Agent 1 (X): {agent1.name}")
    print(f"    Backend: {args.agent1_backend}  Model: {args.agent1_model}  Mode: {args.agent1_mode}")
    print(f"\n  Agent 2 (O): {agent2.name}")
    print(f"    Backend: {args.agent2_backend}  Model: {args.agent2_model}  Mode: {args.agent2_mode}")
    print(f"\n  Games: {args.games}  Iterations: {args.iterations}")
    print()

    run_dir = run_benchmark(
        agent1=agent1,
        agent2=agent2,
        num_games=args.games,
        output_dir=args.output_dir,
        log_boards=not args.no_boards,
    )

    print(f"\n📊 Full reports in: {run_dir}/")
    print("   summary.txt          — overview and win rates")
    print("   game_log.txt         — move-by-move game log")
    print("   move_stats.csv       — raw CSV data")
    print("   win_probability.txt  — LLM win estimate per move")
    print("   agent_<name>.txt     — detailed per-agent stats")
    print("   config.json          — full run configuration")


if __name__ == "__main__":
    main()