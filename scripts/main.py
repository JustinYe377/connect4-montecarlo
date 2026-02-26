"""Connect4 Monte Carlo main."""

from typing import List
from threading import Thread
from queue import Queue
import sys
import os

import pygame

from game_graphics import GameGraphics
from connect4_mcts import GameBoard, MCTS, Node

# Screen resolution
WIN_SIZE = (W_WIDTH, W_HEIGHT) = (800, 600)

# MCTS move computation time
PROCESS_TIME = 5

# Frame rate
FPS = 30

# ── LLM Configuration ─────────────────────────────────────────────────────────
# Set USE_LLM = True to enable LLM-based rollout evaluation.
# Set USE_LLM = False for original pure-random MCTS (no API calls).
USE_LLM = True

# Backend: "ollama" (local, free) | "deepseek" (cloud) | "anthropic" (cloud)
LLM_BACKEND = "ollama"

# Model name — examples:
#   ollama:     "deepseek-r1:7b", "llama3.2:3b", "mistral"
#   deepseek:   "deepseek-chat", "deepseek-reasoner"
#   anthropic:  "claude-haiku-4-5-20251001"
LLM_MODEL = "deepseek-r1:7b"

# Evaluation mode: "direct" | "prior" | "guided" | "hybrid"
LLM_MODE = "hybrid"

# For hybrid mode: how many LLM-guided plies before switching to random play
LLM_HYBRID_K = 2

# API key — leave None to read from env var (DEEPSEEK_API_KEY / ANTHROPIC_API_KEY)
LLM_API_KEY = None

# Custom Ollama URL — None uses default http://localhost:11434
LLM_BASE_URL = None

# Set True to generate a reports/ folder with stats after each game
ENABLE_REPORTING = True
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    # Initialize stuff
    os.system("cls")
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Connect 4 Montecarlo")
    window = pygame.display.set_mode(WIN_SIZE)
    clock = pygame.time.Clock()
    move_queue: "Queue[int]" = Queue()
    graphics = GameGraphics(win_size=WIN_SIZE, surface=window)

    # ── Reporter setup (optional) ──────────────────────────────────────────────
    reporter = None
    if ENABLE_REPORTING:
        try:
            from reporter import GameReporter
            reporter = GameReporter(
                config={
                    "backend": LLM_BACKEND if USE_LLM else "random",
                    "model":   LLM_MODEL   if USE_LLM else "random",
                    "mode":    LLM_MODE    if USE_LLM else "random",
                    "process_time_s": PROCESS_TIME,
                },
                output_dir="reports",
            )
            reporter.register_agent("MCTS-LLM" if USE_LLM else "MCTS-Random",
                                    1, LLM_MODE if USE_LLM else "random",
                                    LLM_BACKEND if USE_LLM else "none",
                                    LLM_MODEL   if USE_LLM else "none")
            reporter.register_agent("Human", 2, "human", "none", "none")
        except ImportError:
            print("[main] reporter.py not found — reporting disabled.")
    # ──────────────────────────────────────────────────────────────────────────

    # Begin new game
    while True:

        gameboard = GameBoard(cpu=1)
        montecarlo = MCTS(symbol=1, t=PROCESS_TIME)

        # ── Enable LLM evaluation ──────────────────────────────────────────────
        if USE_LLM:
            montecarlo.setup_llm(
                backend=LLM_BACKEND,
                model=LLM_MODEL,
                mode=LLM_MODE,
                hybrid_k=LLM_HYBRID_K,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                reporter=reporter,
            )
        # ──────────────────────────────────────────────────────────────────────

        if reporter:
            reporter.start_game()

        game_over = False
        winner_id = None
        select_move = 1
        move_number = 0
        threads: List[Thread] = []

        # Game loop
        while True:

            # Check for game over
            game_over, winner_id = gameboard.check_win()
            if game_over is True:
                if reporter:
                    reporter.finish_game(winner=winner_id, total_moves=move_number)
                    reporter.save()
                pygame.time.wait(1000)
                break

            # Human turn
            if gameboard.turn != gameboard.cpu:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            sys.exit()
                        elif event.key == pygame.K_RIGHT:
                            if select_move < 7:
                                select_move += 1
                        elif event.key == pygame.K_LEFT:
                            if select_move > 1:
                                select_move -= 1
                        elif event.key == pygame.K_RETURN:
                            if gameboard.board[5, select_move - 1] == 0:
                                # Record human move
                                if reporter:
                                    from reporter import GameReporter
                                    from llm_mcts import MoveRecord
                                    move_number += 1
                                    legal = [c+1 for c in range(7) if gameboard.board[5, c] == 0]
                                    reporter.record_move(MoveRecord(
                                        move_number=move_number,
                                        player=gameboard.turn,
                                        column=select_move,
                                        legal_moves=legal,
                                        llm_win_prob=None,
                                        eval_mode="human",
                                        latency_s=0.0,
                                    ))
                                gameboard.apply_move(select_move)

            # Monte Carlo turn
            else:

                # Start thinking
                if len(threads) == 0:
                    root = Node(
                        parent=None,
                        board=gameboard.board,
                        turn=montecarlo.symbol,
                    )
                    t = Thread(
                        target=lambda q, x: q.put(montecarlo.compute_move(x)),
                        args=(move_queue, root),
                    )
                    t.start()
                    threads.append(t)

                # Ready to play
                if move_queue.empty() is False:
                    threads.pop()
                    move = move_queue.get()
                    # Record MCTS move
                    if reporter:
                        from llm_mcts import MoveRecord
                        move_number += 1
                        legal = [c+1 for c in range(7) if gameboard.board[5, c] == 0]
                        # Evaluate position for reporting
                        win_prob = None
                        if montecarlo.llm_evaluator is not None:
                            try:
                                win_prob = montecarlo.llm_evaluator.provider.evaluate_position(
                                    gameboard.board.tolist(), montecarlo.symbol
                                )
                            except Exception:
                                pass
                        reporter.record_move(MoveRecord(
                            move_number=move_number,
                            player=montecarlo.symbol,
                            column=move[1] + 1,  # convert 0-indexed col to 1-indexed
                            legal_moves=legal,
                            llm_win_prob=win_prob,
                            eval_mode=LLM_MODE if USE_LLM else "random",
                            latency_s=0.0,
                        ))
                    gameboard.board[move] = montecarlo.symbol
                    gameboard.switch_turn()

            # Draw game graphics
            graphics.draw_background(speed=100)
            graphics.draw_board(board=gameboard.board)
            if gameboard.turn != gameboard.cpu:
                graphics.draw_select(column=select_move, turn=gameboard.turn)

            # Update stuff
            clock.tick(FPS)
            pygame.event.pump()
            pygame.display.flip()

        # Game over / continue
        select_option = 1
        new_game = False
        while new_game is False:

            # Menu controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()
                    elif event.key == pygame.K_RIGHT:
                        if select_option < 2:
                            select_option += 1
                    elif event.key == pygame.K_LEFT:
                        if select_option > 1:
                            select_option -= 1
                    elif event.key == pygame.K_RETURN:
                        if select_option == 1:
                            new_game = True
                        elif select_option == 2:
                            sys.exit()

            # Draw game over screen
            graphics.draw_background(speed=100)
            graphics.gameover_screen(winner_id, select_option)

            # Update stuff
            clock.tick(FPS)
            pygame.event.pump()
            pygame.display.flip()