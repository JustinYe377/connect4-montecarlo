"""
llm_mcts.py
-----------
Drop-in LLM evaluation strategies for connect4-montecarlo.

Four modes (mirroring the assignment spec):
  1. "direct"   – LLM estimates win probability directly (replaces rollout)
  2. "prior"    – LLM ranks legal moves → biases UCB selection
  3. "guided"   – LLM picks each move during rollout instead of random
  4. "hybrid"   – LLM for first k plies, then random to terminal

Plug-in pattern
---------------
In your MCTSNode (or wherever simulate() is called), replace:

    result = self.random_rollout()

with:

    evaluator = LLMEvaluator(provider, mode="hybrid", hybrid_k=2)
    result = evaluator.rollout(board_state, current_player, game_obj)

The evaluator also records per-move statistics used by the reporter.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from llm_provider import CachedProvider


# ---------------------------------------------------------------------------
# Move statistics record
# ---------------------------------------------------------------------------

class MoveRecord:
    """Stores data for one move decision during a game."""

    def __init__(
        self,
        move_number: int,
        player: int,
        column: int,
        legal_moves: list[int],
        llm_win_prob: Optional[float],
        eval_mode: str,
        latency_s: float,
    ):
        self.move_number = move_number
        self.player = player
        self.column = column
        self.legal_moves = legal_moves
        self.llm_win_prob = llm_win_prob
        self.eval_mode = eval_mode
        self.latency_s = latency_s

    def to_dict(self) -> dict:
        return {
            "move_number": self.move_number,
            "player": self.player,
            "column": self.column,
            "legal_moves": self.legal_moves,
            "llm_win_prob": round(self.llm_win_prob, 4) if self.llm_win_prob is not None else None,
            "eval_mode": self.eval_mode,
            "latency_s": round(self.latency_s, 4),
        }


# ---------------------------------------------------------------------------
# LLM Evaluator
# ---------------------------------------------------------------------------

class LLMEvaluator:
    """
    Wraps an LLM provider and implements the four evaluation strategies.

    Parameters
    ----------
    provider    : CachedProvider from llm_provider.py
    mode        : "direct" | "prior" | "guided" | "hybrid"
    hybrid_k    : number of LLM-guided plies before switching to random (hybrid mode)
    move_records: list to append MoveRecord objects to (for reporting)
    """

    def __init__(
        self,
        provider: CachedProvider,
        mode: str = "hybrid",
        hybrid_k: int = 2,
        move_records: Optional[list] = None,
    ):
        self.provider = provider
        self.mode = mode.lower()
        self.hybrid_k = hybrid_k
        self.move_records = move_records if move_records is not None else []
        self._move_counter = 0

        valid_modes = {"direct", "prior", "guided", "hybrid"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")

    # ------------------------------------------------------------------
    # Public API — call this instead of random_rollout()
    # ------------------------------------------------------------------

    def rollout(
        self,
        board: list[list[int]],
        current_player: int,
        game,
    ) -> float:
        """
        Run one evaluation / rollout from the given board state.

        Returns a float in [0,1] representing win probability for `current_player`
        (the player who just triggered this rollout from their MCTS node).

        `game` must expose:
            game.get_legal_moves(board)          → list[int]
            game.make_move(board, col, player)   → new_board
            game.check_win(board, player)        → bool
            game.is_draw(board)                  → bool
        """
        if self.mode == "direct":
            return self._direct(board, current_player)
        elif self.mode == "prior":
            return self._prior_rollout(board, current_player, game)
        elif self.mode == "guided":
            return self._guided_rollout(board, current_player, game, max_llm=9999)
        elif self.mode == "hybrid":
            return self._guided_rollout(board, current_player, game, max_llm=self.hybrid_k)

    def get_move_prior(
        self,
        board: list[list[int]],
        current_player: int,
    ) -> dict[int, float]:
        """
        Return a probability distribution over legal moves for the Selection phase.
        Uses the LLM to suggest the best move and amplifies that column's weight.
        """
        legal = [c for c in range(len(board[0])) if board[0][c] == 0]
        if not legal:
            return {}

        t0 = time.time()
        best_col = self.provider.suggest_move(board, current_player)
        latency = time.time() - t0

        # Distribute: 60% to LLM pick, remainder spread uniformly
        prior = {c: 0.4 / max(len(legal) - 1, 1) for c in legal}
        if best_col in legal:
            prior[best_col] = 0.6
        else:
            # fallback: uniform
            prior = {c: 1.0 / len(legal) for c in legal}

        # Normalize
        total = sum(prior.values())
        prior = {c: v / total for c, v in prior.items()}

        self._log_move(current_player, best_col or -1, legal, None, "prior", latency)
        return prior

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _direct(self, board, current_player: int) -> float:
        """Ask LLM to directly estimate win probability (no rollout)."""
        t0 = time.time()
        prob = self.provider.evaluate_position(board, current_player)
        latency = time.time() - t0
        self._log_move(current_player, -1, [], prob, "direct", latency)
        return prob

    def _prior_rollout(self, board, current_player: int, game) -> float:
        """
        Use LLM priors to pick moves during rollout
        (weighted-random rather than uniform-random).
        """
        state = [row[:] for row in board]
        player = current_player
        depth = 0

        while True:
            legal = game.get_legal_moves(state)
            if not legal:
                return 0.5

            # LLM prior-weighted selection
            t0 = time.time()
            best_col = self.provider.suggest_move(state, player)
            latency = time.time() - t0

            weights = [3.0 if c == best_col else 1.0 for c in legal]
            col = random.choices(legal, weights=weights, k=1)[0]

            t_eval = time.time()
            eval_prob = self.provider.evaluate_position(state, player)
            eval_latency = time.time() - t_eval

            self._log_move(player, col, legal, eval_prob, "prior", latency + eval_latency)

            state = game.make_move(state, col, player)
            if game.check_win(state, player):
                return 1.0 if player == current_player else 0.0
            if game.is_draw(state):
                return 0.5

            player = 3 - player
            depth += 1
            if depth > 42:  # safety
                return 0.5

    def _guided_rollout(
        self, board, current_player: int, game, max_llm: int
    ) -> float:
        """
        LLM-guided for first `max_llm` plies, then random to terminal.
        max_llm=9999 → fully guided (mode='guided').
        """
        state = [row[:] for row in board]
        player = current_player
        depth = 0

        while True:
            legal = game.get_legal_moves(state)
            if not legal:
                return 0.5

            if depth < max_llm:
                # LLM picks move
                t0 = time.time()
                col = self.provider.suggest_move(state, player)
                latency = time.time() - t0

                if col not in legal:
                    col = random.choice(legal)

                # evaluate position before moving
                t_eval = time.time()
                prob = self.provider.evaluate_position(state, player)
                eval_latency = time.time() - t_eval
                self._log_move(player, col, legal, prob, self.mode, latency + eval_latency)
            else:
                # Random play
                col = random.choice(legal)
                self._log_move(player, col, legal, None, "random", 0.0)

            state = game.make_move(state, col, player)
            if game.check_win(state, player):
                return 1.0 if player == current_player else 0.0
            if game.is_draw(state):
                return 0.5

            player = 3 - player
            depth += 1
            if depth > 42:
                return 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_move(self, player, col, legal, prob, mode, latency):
        self._move_counter += 1
        self.move_records.append(MoveRecord(
            move_number=self._move_counter,
            player=player,
            column=col,
            legal_moves=list(legal),
            llm_win_prob=prob,
            eval_mode=mode,
            latency_s=latency,
        ))
