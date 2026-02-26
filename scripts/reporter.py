"""
reporter.py
-----------
Records game statistics and generates a human-readable report folder.

Each call to run_benchmark() or finish_game() creates a dated folder:
    reports/run_2025-03-01_14-30-00/
        summary.txt          – overview of all agents and results
        game_log.txt         – move-by-move log with board states
        move_stats.csv       – raw CSV of every move record
        agent_<name>.txt     – per-agent detailed stats
        win_probability.txt  – LLM win-chance estimates per move
        config.json          – full run configuration

Usage
-----
    reporter = GameReporter(config={"backend": "ollama", "model": "deepseek-r1:7b"})
    reporter.start_game(agent1_name="MCTS-LLM", agent2_name="MCTS-Random")
    reporter.record_move(move_record)        # after each move
    reporter.finish_game(winner=1, moves=42)
    reporter.save()                          # writes all files
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from llm_mcts import MoveRecord


# ---------------------------------------------------------------------------
# Agent stats
# ---------------------------------------------------------------------------

@dataclass
class AgentStats:
    name: str
    player_id: int         # 1 or 2
    mode: str              # evaluation mode
    backend: str
    model: str

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    total_moves: int = 0
    total_llm_calls: int = 0
    total_cache_hits: int = 0
    total_latency_s: float = 0.0
    win_probs: list[float] = field(default_factory=list)
    legal_move_counts: list[int] = field(default_factory=list)
    chosen_columns: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games_played if self.games_played else 0.0

    @property
    def avg_win_prob(self) -> float:
        return sum(self.win_probs) / len(self.win_probs) if self.win_probs else 0.5

    @property
    def avg_legal_moves(self) -> float:
        return (
            sum(self.legal_move_counts) / len(self.legal_move_counts)
            if self.legal_move_counts else 0.0
        )

    @property
    def avg_latency_ms(self) -> float:
        n = self.total_llm_calls
        return (self.total_latency_s / n * 1000) if n else 0.0

    @property
    def column_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for c in self.chosen_columns:
            dist[c] = dist.get(c, 0) + 1
        return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class GameReporter:
    """
    Tracks games and generates a folder of human-readable reports.
    """

    def __init__(self, config: dict, output_dir: str = "reports"):
        self.config = config
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.agents: dict[int, AgentStats] = {}
        self.all_move_records: list[MoveRecord] = []
        self.game_log: list[str] = []

        self._game_start: float = 0.0
        self._current_game: int = 0
        self._game_results: list[dict] = []
        self._current_moves: list[MoveRecord] = []
        self._current_boards: list[str] = []

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def register_agent(
        self,
        name: str,
        player_id: int,
        mode: str,
        backend: str,
        model: str,
    ):
        """Call once per agent before starting games."""
        self.agents[player_id] = AgentStats(
            name=name,
            player_id=player_id,
            mode=mode,
            backend=backend,
            model=model,
        )

    def start_game(self):
        """Call at the beginning of each game."""
        self._game_start = time.time()
        self._current_game += 1
        self._current_moves = []
        self._current_boards = []
        self.game_log.append(
            f"\n{'='*60}\n"
            f"  GAME {self._current_game}\n"
            f"{'='*60}"
        )

    def record_move(self, record: MoveRecord, board_text: Optional[str] = None):
        """Call after each move in the game."""
        self._current_moves.append(record)
        self.all_move_records.append(record)

        agent = self.agents.get(record.player)
        if agent:
            agent.total_moves += 1
            if record.llm_win_prob is not None:
                agent.win_probs.append(record.llm_win_prob)
                agent.total_llm_calls += 1
                agent.total_latency_s += record.latency_s
            if record.legal_moves:
                agent.legal_move_counts.append(len(record.legal_moves))
            if record.column >= 0:
                agent.chosen_columns.append(record.column)

        agent_name = agent.name if agent else f"Player {record.player}"
        prob_str = f"{record.llm_win_prob:.2f}" if record.llm_win_prob is not None else "N/A"
        legal_str = str(record.legal_moves)

        self.game_log.append(
            f"  Move {record.move_number:>3} | {agent_name:<20} | "
            f"Col: {record.column} | Legal: {legal_str} | "
            f"Win%: {prob_str} | Mode: {record.eval_mode} | "
            f"Latency: {record.latency_s*1000:.0f}ms"
        )
        if board_text:
            self._current_boards.append(board_text)
            self.game_log.append(board_text)

    def finish_game(self, winner: Optional[int], total_moves: int):
        """
        Call at end of game.
        winner = 1 or 2 (player id), or None for draw.
        """
        duration = time.time() - self._game_start
        result = {
            "game": self._current_game,
            "winner": winner,
            "total_moves": total_moves,
            "duration_s": round(duration, 2),
        }
        self._game_results.append(result)

        for pid, agent in self.agents.items():
            agent.games_played += 1
            if winner is None:
                agent.draws += 1
            elif winner == pid:
                agent.wins += 1
            else:
                agent.losses += 1

        if winner is None:
            winner_str = "Draw"
        else:
            agent = self.agents.get(winner)
            winner_str = agent.name if agent else f"Player {winner}"

        self.game_log.append(
            f"\n  Result: {winner_str} wins! ({total_moves} moves, {duration:.1f}s)\n"
        )

    # ------------------------------------------------------------------
    # Save all reports
    # ------------------------------------------------------------------

    def save(self):
        """Write all report files to the run directory."""
        self._write_config()
        self._write_summary()
        self._write_game_log()
        self._write_move_csv()
        self._write_win_probability()
        for agent in self.agents.values():
            self._write_agent_report(agent)
        print(f"\n📁 Reports saved to: {self.run_dir}/")
        return self.run_dir

    # ------------------------------------------------------------------
    # Individual file writers
    # ------------------------------------------------------------------

    def _write_config(self):
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

    def _write_summary(self):
        path = os.path.join(self.run_dir, "summary.txt")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 70,
            "  CONNECT FOUR LLM-MCTS — RUN SUMMARY",
            f"  Generated: {now}",
            "=" * 70,
            "",
            f"  Total games played: {self._current_game}",
            "",
            "  AGENT OVERVIEW",
            "  " + "-" * 66,
        ]

        for agent in self.agents.values():
            lines += [
                f"  Agent: {agent.name}",
                f"    Player ID   : {agent.player_id}",
                f"    Backend     : {agent.backend}",
                f"    Model       : {agent.model}",
                f"    Eval Mode   : {agent.mode}",
                f"    Games       : {agent.games_played}",
                f"    Wins        : {agent.wins}",
                f"    Losses      : {agent.losses}",
                f"    Draws       : {agent.draws}",
                f"    Win Rate    : {agent.win_rate:.1%}",
                f"    Draw Rate   : {agent.draw_rate:.1%}",
                f"    Avg Win Prob: {agent.avg_win_prob:.2f}  (LLM estimate)",
                f"    Avg Legal Moves per turn: {agent.avg_legal_moves:.1f}",
                f"    LLM Calls   : {agent.total_llm_calls}",
                f"    Avg Latency : {agent.avg_latency_ms:.0f} ms/call",
                f"    Col Dist    : {agent.column_distribution}",
                "",
            ]

        lines += [
            "  GAME-BY-GAME RESULTS",
            "  " + "-" * 66,
        ]
        for r in self._game_results:
            if r["winner"] is None:
                w = "Draw"
            else:
                a = self.agents.get(r["winner"])
                w = a.name if a else f"Player {r['winner']}"
            lines.append(
                f"  Game {r['game']:>3}: {w:<25} {r['total_moves']} moves  {r['duration_s']:.1f}s"
            )

        lines += ["", "=" * 70]
        _write_lines(path, lines)

    def _write_game_log(self):
        path = os.path.join(self.run_dir, "game_log.txt")
        header = [
            "=" * 70,
            "  CONNECT FOUR — DETAILED GAME LOG",
            "=" * 70,
        ]
        _write_lines(path, header + self.game_log)

    def _write_move_csv(self):
        path = os.path.join(self.run_dir, "move_stats.csv")
        fieldnames = [
            "move_number", "player", "agent_name", "column",
            "num_legal_moves", "legal_moves",
            "llm_win_prob", "eval_mode", "latency_ms",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.all_move_records:
                agent = self.agents.get(rec.player)
                writer.writerow({
                    "move_number": rec.move_number,
                    "player": rec.player,
                    "agent_name": agent.name if agent else f"Player {rec.player}",
                    "column": rec.column,
                    "num_legal_moves": len(rec.legal_moves),
                    "legal_moves": str(rec.legal_moves),
                    "llm_win_prob": round(rec.llm_win_prob, 4) if rec.llm_win_prob is not None else "",
                    "eval_mode": rec.eval_mode,
                    "latency_ms": round(rec.latency_s * 1000, 1),
                })

    def _write_win_probability(self):
        path = os.path.join(self.run_dir, "win_probability.txt")
        lines = [
            "=" * 70,
            "  WIN PROBABILITY TRACE (LLM estimates per move)",
            "  Values: 0.0 = certain loss, 0.5 = even, 1.0 = certain win",
            "=" * 70,
            "",
        ]
        lines.append(f"  {'Move':>5}  {'Agent':<22}  {'Column':>6}  {'Win%':>6}  {'Mode':<10}")
        lines.append("  " + "-" * 56)

        for rec in self.all_move_records:
            if rec.llm_win_prob is None:
                continue
            agent = self.agents.get(rec.player)
            name = agent.name if agent else f"Player {rec.player}"
            bar_len = int(rec.llm_win_prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(
                f"  {rec.move_number:>5}  {name:<22}  col {rec.column}  "
                f"  {rec.llm_win_prob:.2f}  {rec.eval_mode:<10}  |{bar}|"
            )

        _write_lines(path, lines)

    def _write_agent_report(self, agent: AgentStats):
        fname = f"agent_{agent.name.replace(' ', '_')}.txt"
        path = os.path.join(self.run_dir, fname)
        lines = [
            "=" * 70,
            f"  AGENT REPORT: {agent.name}",
            "=" * 70,
            "",
            f"  Configuration",
            f"    Backend     : {agent.backend}",
            f"    Model       : {agent.model}",
            f"    Eval Mode   : {agent.mode}",
            f"    Player ID   : {agent.player_id}",
            "",
            f"  Performance",
            f"    Games Played: {agent.games_played}",
            f"    Wins        : {agent.wins}  ({agent.win_rate:.1%})",
            f"    Losses      : {agent.losses}",
            f"    Draws       : {agent.draws}  ({agent.draw_rate:.1%})",
            "",
            f"  Move Statistics",
            f"    Total moves : {agent.total_moves}",
            f"    LLM calls   : {agent.total_llm_calls}",
            f"    Cache hits  : {agent.total_cache_hits}",
            f"    Avg latency : {agent.avg_latency_ms:.0f} ms/call",
            f"    Total lat.  : {agent.total_latency_s:.2f}s",
            "",
            f"  Board Evaluation",
            f"    Avg Win Prob: {agent.avg_win_prob:.4f}",
        ]
        if agent.win_probs:
            mn = min(agent.win_probs)
            mx = max(agent.win_probs)
            lines.append(f"    Min/Max Prob: {mn:.4f} / {mx:.4f}")
        if agent.legal_move_counts:
            lines.append(f"    Avg Legal Moves: {agent.avg_legal_moves:.2f}")

        lines += [
            "",
            f"  Column Selection Distribution",
            "    Col | Count | Bar",
            "    " + "-" * 40,
        ]
        dist = agent.column_distribution
        max_count = max(dist.values()) if dist else 1
        for col, count in dist.items():
            bar = "█" * int(count / max_count * 25)
            lines.append(f"    {col:>3} | {count:>5} | {bar}")

        if agent.win_probs:
            lines += ["", "  Win Probability History (sampled)"]
            sample = agent.win_probs[::max(1, len(agent.win_probs) // 50)]
            for i, p in enumerate(sample):
                bar_len = int(p * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lines.append(f"    {i*max(1,len(agent.win_probs)//50):>5}: {p:.2f} |{bar}|")

        _write_lines(path, lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write_lines(path: str, lines: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
