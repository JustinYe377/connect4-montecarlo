"""
llm_provider.py
---------------
Unified LLM provider interface supporting:
  - Anthropic Claude (cloud)
  - DeepSeek (cloud, OpenAI-compatible)
  - Ollama (local, any model)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import urllib.error


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "qwen3:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512      # high enough for thinking model <think> blocks
    timeout: int = 60          # thinking models can be slow
    cache: bool = True
    debug: bool = False        # set True to print raw LLM responses

    _call_count: int = field(default=0, init=False, repr=False)
    _total_latency: float = field(default=0.0, init=False, repr=False)
    _cache_hits: int = field(default=0, init=False, repr=False)


PLAYER_SYMBOLS = {1: "X", 2: "O"}


def board_to_text(board: list[list[int]], origin: str = "top") -> str:
    """
    Convert board to readable text. Output always shows bottom row at the bottom.

    origin="top"    -> board[0] is the TOP   (game_adapter list-of-lists format)
    origin="bottom" -> board[0] is the BOTTOM (original numpy GameBoard format)
    """
    sym = {0: ".", 1: "X", 2: "O"}
    if origin == "bottom":
        display_rows = list(reversed(board))
    else:
        display_rows = list(board)
    lines = [" ".join(sym[int(c)] for c in row) for row in display_rows]
    lines.append(" ".join(str(i) for i in range(len(board[0]))))
    return "\n".join(lines)


def _get_legal_cols(board: list[list[int]], origin: str = "top") -> list[int]:
    cols = len(board[0])
    rows = len(board)
    if origin == "bottom":
        return [c for c in range(cols) if board[rows - 1][c] == 0]
    else:
        return [c for c in range(cols) if board[0][c] == 0]


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_float(text: str) -> float:
    """Extract final answer float, ignoring thinking blocks."""
    text = _strip_thinking(text).strip().strip("`").strip()
    match = re.search(r"0?\.\d+|\d+\.\d*|\d+", text)
    if match:
        val = float(match.group())
        return max(0.0, min(1.0, val))
    return 0.5


def _build_prompt(board: list[list[int]], current_player: int, origin: str = "top") -> str:
    sym = PLAYER_SYMBOLS[current_player]
    board_str = board_to_text(board, origin=origin)
    legal_cols = _get_legal_cols(board, origin=origin)
    legal_str = ", ".join(str(c) for c in legal_cols)
    return (
        f"You are a Connect Four expert evaluator.\n\n"
        f"Current board (bottom row = gravity, pieces fall downward):\n"
        f"{board_str}\n\n"
        f"Player {sym} is about to move. Legal columns: {legal_str}\n\n"
        f"Analyze the position carefully:\n"
        f"- Does either player have 3-in-a-row with an open end? (immediate threat)\n"
        f"- Who controls the center columns (2, 3, 4)?\n"
        f"- Who has more connected pieces?\n\n"
        f"Estimate the win probability for player {sym}.\n"
        f"Output ONLY a single decimal number between 0.0 and 1.0.\n"
        f"Examples: 0.3 (losing), 0.5 (even), 0.7 (winning), 0.9 (near certain win)\n"
        f"Your answer (just the number):"
    )


def _build_move_prompt(board: list[list[int]], current_player: int, origin: str = "top") -> str:
    sym = PLAYER_SYMBOLS[current_player]
    board_str = board_to_text(board, origin=origin)
    legal_cols = _get_legal_cols(board, origin=origin)
    legal_str = ", ".join(str(c) for c in legal_cols)
    return (
        f"You are a Connect Four expert.\n\n"
        f"Current board (bottom row = gravity):\n"
        f"{board_str}\n\n"
        f"Player {sym} must play. Legal columns: {legal_str}\n\n"
        f"Choose the best column. Consider winning moves first, then blocking.\n"
        f"Output ONLY a single integer column number from: {legal_str}\n"
        f"Your answer (just the number):"
    )


def _http_post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class AnthropicProvider:
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found.")

    def _call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": self.API_VERSION}
        body = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "messages": [{"role": "user", "content": prompt}]}
        return _http_post(self.BASE_URL, headers, body, self.cfg.timeout)["content"][0]["text"]

    def evaluate_position(self, board, current_player: int) -> float:
        raw = self._call(_build_prompt(board, current_player, origin="top"))
        val = _parse_float(raw)
        if self.cfg.debug:
            print(f"  [LLM eval] answer={repr(_strip_thinking(raw).strip()[:60])}  parsed={val:.2f}")
        return val

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = _get_legal_cols(board, origin="top")
        if not legal:
            return None
        raw = self._call(_build_move_prompt(board, current_player, origin="top"))
        text = _strip_thinking(raw).strip()
        if self.cfg.debug:
            print(f"  [LLM move] answer={repr(text[:60])}")
        try:
            col = int(re.search(r"\d+", text).group())
            return col if col in legal else None
        except Exception:
            return None


class DeepSeekProvider:
    BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found.")
        self.base_url = cfg.base_url or self.BASE_URL

    def _call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        body = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "messages": [{"role": "user", "content": prompt}]}
        return _http_post(self.base_url, headers, body, self.cfg.timeout)["choices"][0]["message"]["content"]

    def evaluate_position(self, board, current_player: int) -> float:
        raw = self._call(_build_prompt(board, current_player, origin="top"))
        val = _parse_float(raw)
        if self.cfg.debug:
            print(f"  [LLM eval] answer={repr(_strip_thinking(raw).strip()[:60])}  parsed={val:.2f}")
        return val

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = _get_legal_cols(board, origin="top")
        if not legal:
            return None
        raw = self._call(_build_move_prompt(board, current_player, origin="top"))
        text = _strip_thinking(raw).strip()
        if self.cfg.debug:
            print(f"  [LLM move] answer={repr(text[:60])}")
        try:
            col = int(re.search(r"\d+", text).group())
            return col if col in legal else None
        except Exception:
            return None


class OllamaProvider:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")

    def _call(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        body = {"model": self.cfg.model, "prompt": prompt, "stream": False,
                "options": {"temperature": self.cfg.temperature,
                            "num_predict": self.cfg.max_tokens}}
        return _http_post(url, headers, body, self.cfg.timeout).get("response", "")

    def evaluate_position(self, board, current_player: int) -> float:
        raw = self._call(_build_prompt(board, current_player, origin="top"))
        val = _parse_float(raw)
        if self.cfg.debug:
            print(f"  [LLM eval] answer={repr(_strip_thinking(raw).strip()[:60])}  parsed={val:.2f}")
        return val

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = _get_legal_cols(board, origin="top")
        if not legal:
            return None
        raw = self._call(_build_move_prompt(board, current_player, origin="top"))
        text = _strip_thinking(raw).strip()
        if self.cfg.debug:
            print(f"  [LLM move] answer={repr(text[:60])}")
        try:
            col = int(re.search(r"\d+", text).group())
            return col if col in legal else None
        except Exception:
            return None


class CachedProvider:
    def __init__(self, inner, cfg: LLMConfig):
        self._inner = inner
        self._cfg = cfg
        self._eval_cache: dict[tuple, float] = {}
        self._move_cache: dict[tuple, int | None] = {}

    def _board_key(self, board, player):
        return (tuple(tuple(r) for r in board), player)

    def evaluate_position(self, board, current_player: int) -> float:
        key = self._board_key(board, current_player)
        if self._cfg.cache and key in self._eval_cache:
            self._cfg._cache_hits += 1
            return self._eval_cache[key]
        t0 = time.time()
        val = self._inner.evaluate_position(board, current_player)
        self._cfg._call_count += 1
        self._cfg._total_latency += time.time() - t0
        if self._cfg.cache:
            self._eval_cache[key] = val
        return val

    def suggest_move(self, board, current_player: int) -> int | None:
        key = self._board_key(board, current_player)
        if self._cfg.cache and key in self._move_cache:
            self._cfg._cache_hits += 1
            return self._move_cache[key]
        t0 = time.time()
        col = self._inner.suggest_move(board, current_player)
        self._cfg._call_count += 1
        self._cfg._total_latency += time.time() - t0
        if self._cfg.cache:
            self._move_cache[key] = col
        return col

    @property
    def stats(self) -> dict:
        cfg = self._cfg
        avg = (cfg._total_latency / cfg._call_count) if cfg._call_count else 0
        return {"backend": cfg.backend, "model": cfg.model,
                "api_calls": cfg._call_count, "cache_hits": cfg._cache_hits,
                "total_latency_s": round(cfg._total_latency, 3),
                "avg_latency_s": round(avg, 3)}


def get_provider(cfg: LLMConfig) -> CachedProvider:
    backend = cfg.backend.lower()
    if backend == "anthropic":
        inner = AnthropicProvider(cfg)
    elif backend == "deepseek":
        inner = DeepSeekProvider(cfg)
    elif backend == "ollama":
        inner = OllamaProvider(cfg)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: anthropic, deepseek, ollama")
    return CachedProvider(inner, cfg)