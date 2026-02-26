"""
integration_patch.py
--------------------
Shows EXACTLY what to change in the original Gualor connect4-montecarlo
codebase to plug in LLM evaluation. These are surgical edits — the rest
of the original code stays identical.

The original repo structure:
    scripts/
        montecarlo.py   ← main MCTS + game logic
        ...

PATCH INSTRUCTIONS
==================

STEP 1 — Copy new files into scripts/
    cp llm_provider.py   scripts/
    cp llm_mcts.py       scripts/
    cp reporter.py       scripts/
    cp game_adapter.py   scripts/
    cp run_benchmark.py  scripts/

STEP 2 — Edit scripts/montecarlo.py (or your main game file)

Find the class that contains the simulation / rollout.
In the Gualor version it looks roughly like this:

    def simulate(self):
        current_board = copy.deepcopy(self.board)
        current_player = self.player
        while True:
            legal = self.get_legal_moves(current_board)
            if not legal:
                return 0
            col = random.choice(legal)
            current_board = self.make_move(current_board, col, current_player)
            if self.check_win(current_board, current_player):
                return 1 if current_player == self.player else -1
            current_player = 3 - current_player

PATCH: Replace that method body with:

    def simulate(self):
        # --- LLM evaluation (added) ---
        if hasattr(self, '_llm_evaluator') and self._llm_evaluator is not None:
            from game_adapter import Connect4Game
            game = Connect4Game()
            result_01 = self._llm_evaluator.rollout(self.board, self.player, game)
            # Convert from [0,1] to [-1,+1] scale used by original code
            return result_01 * 2 - 1
        # --- Original random rollout (unchanged below) ---
        current_board = copy.deepcopy(self.board)
        current_player = self.player
        while True:
            legal = self.get_legal_moves(current_board)
            if not legal:
                return 0
            col = random.choice(legal)
            current_board = self.make_move(current_board, col, current_player)
            if self.check_win(current_board, current_player):
                return 1 if current_player == self.player else -1
            current_player = 3 - current_player

STEP 3 — In the function that creates the root node / starts MCTS,
add the evaluator:

    from llm_provider import LLMConfig, get_provider
    from llm_mcts import LLMEvaluator

    llm_cfg = LLMConfig(
        backend="ollama",        # or "deepseek" or "anthropic"
        model="deepseek-r1:7b",  # your chosen model
    )
    provider  = get_provider(llm_cfg)
    evaluator = LLMEvaluator(provider, mode="hybrid", hybrid_k=2)

    root = MCTSNode(board, player, ...)
    root._llm_evaluator = evaluator   # inject

STEP 4 — Run standalone (no Pygame, headless):

    python run_benchmark.py \\
        --agent1-backend ollama --agent1-model deepseek-r1:7b \\
        --agent2-backend random \\
        --games 5 --iterations 100

OR to test DeepSeek cloud:
    export DEEPSEEK_API_KEY=your_key_here
    python run_benchmark.py \\
        --agent1-backend deepseek --agent1-model deepseek-chat \\
        --agent2-backend random \\
        --games 3 --iterations 50
"""

# This file is documentation only — no code to run.
# See the docstring above.
print(__doc__)
