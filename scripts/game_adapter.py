"""
game_adapter.py
---------------
Adapter that bridges the Gualor/connect4-montecarlo codebase to the
LLM evaluation system. Provides:

1. `Connect4Game` – lightweight game logic for rollout simulations
   (no Pygame dependency; works standalone for headless runs/tests)

2. `LLMMCTSNode` – drop-in MCTSNode replacement that uses LLMEvaluator
   instead of random rollout

3. `MCTSAgent` – wraps the tree search with agent identity + stat tracking

How to integrate with existing code
-------------------------------------
Option A (minimal change): In your existing MCTSNode.simulate(), call:
    return self.evaluator.rollout(board, current_player, game)
instead of the random loop.

Option B (full replacement): Use LLMMCTSNode directly in place of your
existing node class — it mirrors the same interface.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Optional

from llm_mcts import LLMEvaluator, MoveRecord
from llm_provider import CachedProvider, LLMConfig, get_provider
from reporter import AgentStats, GameReporter


# ---------------------------------------------------------------------------
# Connect4 game logic (standalone, no Pygame)
# ---------------------------------------------------------------------------

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


class Connect4Game:
    """
    Lightweight Connect Four logic for rollout and testing.
    All board operations return new boards (immutable style).
    """

    def get_legal_moves(self, board: list[list[int]]) -> list[int]:
        return [c for c in range(COLS) if board[0][c] == EMPTY]

    def make_move(
        self, board: list[list[int]], col: int, player: int
    ) -> list[list[int]]:
        new_board = [row[:] for row in board]
        for row in range(ROWS - 1, -1, -1):
            if new_board[row][col] == EMPTY:
                new_board[row][col] = player
                return new_board
        return new_board  # shouldn't happen if col is legal

    def check_win(self, board: list[list[int]], player: int) -> bool:
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(board[r][c + i] == player for i in range(4)):
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(board[r + i][c] == player for i in range(4)):
                    return True
        # Diagonal /
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(board[r - i][c + i] == player for i in range(4)):
                    return True
        # Diagonal \
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self, board: list[list[int]]) -> bool:
        return not self.get_legal_moves(board)

    def is_terminal(self, board: list[list[int]]) -> bool:
        return (
            self.check_win(board, PLAYER1)
            or self.check_win(board, PLAYER2)
            or self.is_draw(board)
        )

    def board_to_text(self, board: list[list[int]]) -> str:
        sym = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        lines = [" ".join(sym[c] for c in row) for row in board]
        lines.append(" ".join(str(i) for i in range(COLS)))
        return "\n".join(lines)

    @staticmethod
    def empty_board() -> list[list[int]]:
        return [[EMPTY] * COLS for _ in range(ROWS)]


# ---------------------------------------------------------------------------
# LLM-enhanced MCTS Node
# ---------------------------------------------------------------------------

class LLMMCTSNode:
    """
    MCTS node that uses an LLMEvaluator for simulation.
    Mirrors the structure of the original Gualor MCTSNode.

    Parameters
    ----------
    board       : current board state (2D list)
    player      : player whose turn it is (1 or 2)
    parent      : parent node (None for root)
    move        : column that led to this state
    evaluator   : LLMEvaluator instance
    game        : Connect4Game instance
    c_param     : UCT exploration constant
    use_prior   : if True, use LLM action prior to bias selection
    """

    def __init__(
        self,
        board: list[list[int]],
        player: int,
        parent: Optional["LLMMCTSNode"],
        move: Optional[int],
        evaluator: LLMEvaluator,
        game: Connect4Game,
        c_param: float = math.sqrt(2),
        use_prior: bool = False,
    ):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.evaluator = evaluator
        self.game = game
        self.c_param = c_param
        self.use_prior = use_prior

        self.wins: float = 0.0
        self.visits: int = 0
        self.children: list["LLMMCTSNode"] = []
        self._untried_moves = game.get_legal_moves(board)

        # Action priors (populated lazily)
        self._priors: dict[int, float] = {}

    def is_fully_expanded(self) -> bool:
        return len(self._untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game.is_terminal(self.board)

    def uct_value(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = self.c_param * math.sqrt(math.log(parent_visits) / self.visits)
        prior = self._priors.get(self.move, 1.0) if self.use_prior else 0.0
        return exploitation + exploration + 0.1 * prior

    def best_child(self) -> "LLMMCTSNode":
        return max(self.children, key=lambda c: c.uct_value(self.visits))

    def expand(self) -> "LLMMCTSNode":
        move = self._untried_moves.pop()
        new_board = self.game.make_move(self.board, move, self.player)
        next_player = 3 - self.player
        child = LLMMCTSNode(
            board=new_board,
            player=next_player,
            parent=self,
            move=move,
            evaluator=self.evaluator,
            game=self.game,
            c_param=self.c_param,
            use_prior=self.use_prior,
        )
        self.children.append(child)
        return child

    def simulate(self) -> float:
        """Run one simulation from this node using the LLM evaluator."""
        return self.evaluator.rollout(self.board, self.player, self.game)

    def backpropagate(self, result: float):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1.0 - result)


# ---------------------------------------------------------------------------
# MCTS Agent
# ---------------------------------------------------------------------------

class MCTSAgent:
    """
    A full MCTS agent with LLM evaluation.

    Parameters
    ----------
    name        : agent display name
    player_id   : 1 or 2
    llm_config  : LLMConfig (backend, model, mode, etc.)
    eval_mode   : "direct" | "prior" | "guided" | "hybrid"
    hybrid_k    : plies for hybrid mode
    iterations  : MCTS simulation budget per move
    time_limit  : max seconds per move (0 = use iterations only)
    c_param     : UCT exploration constant
    use_prior   : apply LLM action prior in UCT selection
    """

    def __init__(
        self,
        name: str,
        player_id: int,
        llm_config: LLMConfig,
        eval_mode: str = "hybrid",
        hybrid_k: int = 2,
        iterations: int = 200,
        time_limit: float = 0,
        c_param: float = math.sqrt(2),
        use_prior: bool = False,
    ):
        self.name = name
        self.player_id = player_id
        self.llm_config = llm_config
        self.eval_mode = eval_mode
        self.hybrid_k = hybrid_k
        self.iterations = iterations
        self.time_limit = time_limit
        self.c_param = c_param
        self.use_prior = use_prior

        self.provider: Optional[CachedProvider] = None
        self.evaluator: Optional[LLMEvaluator] = None
        self.move_records: list[MoveRecord] = []
        self.game = Connect4Game()

    def _ensure_provider(self):
        if self.provider is None:
            self.provider = get_provider(self.llm_config)
        if self.evaluator is None:
            self.evaluator = LLMEvaluator(
                provider=self.provider,
                mode=self.eval_mode,
                hybrid_k=self.hybrid_k,
                move_records=self.move_records,
            )

    def get_move(self, board: list[list[int]]) -> int:
        """Run MCTS and return the best column to play."""
        self._ensure_provider()
        root = LLMMCTSNode(
            board=board,
            player=self.player_id,
            parent=None,
            move=None,
            evaluator=self.evaluator,
            game=self.game,
            c_param=self.c_param,
            use_prior=self.use_prior,
        )

        deadline = time.time() + self.time_limit if self.time_limit > 0 else None
        i = 0
        while True:
            if deadline and time.time() >= deadline:
                break
            if not deadline and i >= self.iterations:
                break
            i += 1

            # Selection
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child()

            # Expansion
            if not node.is_terminal():
                node = node.expand()

            # Simulation
            result = node.simulate()

            # Backpropagation
            node.backpropagate(result)

        # Return most-visited child
        if root.children:
            best = max(root.children, key=lambda c: c.visits)
            return best.move
        legal = self.game.get_legal_moves(board)
        return random.choice(legal)

    @property
    def provider_stats(self) -> dict:
        return self.provider.stats if self.provider else {}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    agent1: MCTSAgent,
    agent2: MCTSAgent,
    num_games: int = 5,
    output_dir: str = "reports",
    log_boards: bool = True,
) -> str:
    """
    Run `num_games` games between agent1 and agent2.
    Returns the path to the report directory.
    """
    config = {
        "num_games": num_games,
        "agent1": {
            "name": agent1.name,
            "backend": agent1.llm_config.backend,
            "model": agent1.llm_config.model,
            "mode": agent1.eval_mode,
            "hybrid_k": agent1.hybrid_k,
            "iterations": agent1.iterations,
        },
        "agent2": {
            "name": agent2.name,
            "backend": agent2.llm_config.backend,
            "model": agent2.llm_config.model,
            "mode": agent2.eval_mode,
            "hybrid_k": agent2.hybrid_k,
            "iterations": agent2.iterations,
        },
    }

    reporter = GameReporter(config=config, output_dir=output_dir)
    reporter.register_agent(
        agent1.name, agent1.player_id,
        agent1.eval_mode, agent1.llm_config.backend, agent1.llm_config.model
    )
    reporter.register_agent(
        agent2.name, agent2.player_id,
        agent2.eval_mode, agent2.llm_config.backend, agent2.llm_config.model
    )

    game = Connect4Game()

    for g in range(num_games):
        print(f"\n▶ Game {g+1}/{num_games}")
        reporter.start_game()

        board = Connect4Game.empty_board()
        current = PLAYER1
        move_num = 0
        winner = None

        agents = {PLAYER1: agent1, PLAYER2: agent2}

        while True:
            legal = game.get_legal_moves(board)
            if not legal:
                break

            agent = agents[current]
            t0 = time.time()

            # Check for terminal before asking agent
            if game.check_win(board, 3 - current):
                winner = 3 - current
                break

            col = agent.get_move(board)
            latency = time.time() - t0
            move_num += 1

            # Evaluate position for reporting
            agent._ensure_provider()
            try:
                win_prob = agent.provider.evaluate_position(board, current)
            except Exception:
                win_prob = None

            rec = MoveRecord(
                move_number=move_num,
                player=current,
                column=col,
                legal_moves=legal,
                llm_win_prob=win_prob,
                eval_mode=agent.eval_mode,
                latency_s=latency,
            )
            board_text = game.board_to_text(board) if log_boards else None
            reporter.record_move(rec, board_text)

            board = game.make_move(board, col, current)
            prob_str = f"{win_prob:.2f}" if win_prob is not None else "N/A"
            print(f"  P{current} ({agent.name}) → col {col}  win%={prob_str}")

            if game.check_win(board, current):
                winner = current
                break
            if game.is_draw(board):
                winner = None
                break

            current = 3 - current

        reporter.finish_game(winner=winner, total_moves=move_num)
        if winner:
            print(f"  → Winner: {agents[winner].name}")
        else:
            print("  → Draw")

    # Merge move records from agents into reporter stats
    for pid, agent in {PLAYER1: agent1, PLAYER2: agent2}.items():
        a_stats = reporter.agents[pid]
        p_stats = agent.provider_stats
        a_stats.total_cache_hits = p_stats.get("cache_hits", 0)

    run_dir = reporter.save()
    print(f"\n✅ Done! Reports: {run_dir}")
    return run_dir