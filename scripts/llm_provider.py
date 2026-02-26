"""
llm_provider.py
---------------
Unified LLM provider interface supporting:
  - Anthropic Claude (cloud)
  - DeepSeek (cloud, OpenAI-compatible)
  - Ollama (local, any model)

Usage:
    from llm_provider import get_provider, LLMConfig

    cfg = LLMConfig(backend="ollama", model="deepseek-r1:7b")
    provider = get_provider(cfg)
    score = provider.evaluate_position(board, current_player=1)
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Configuration for the LLM provider.

    Args:
        backend:    "anthropic" | "deepseek" | "ollama"
        model:      model name/tag  (e.g. "claude-haiku-4-5-20251001",
                    "deepseek-chat", "deepseek-r1:7b", "llama3.2:3b")
        api_key:    API key (reads env var automatically if not supplied)
        base_url:   Override base URL (useful for custom Ollama hosts)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Max tokens in response
        timeout:    HTTP timeout in seconds
        cache:      Whether to cache repeated identical positions
    """
    backend: str = "ollama"
    model: str = "deepseek-r1:7b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512      # enough for <think> block + final answer
    timeout: int = 60          # thinking models can be slow, bumped from 30
    cache: bool = True

    # runtime stats (populated automatically)
    _call_count: int = field(default=0, init=False, repr=False)
    _total_latency: float = field(default=0.0, init=False, repr=False)
    _cache_hits: int = field(default=0, init=False, repr=False)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PLAYER_SYMBOLS = {1: "X", 2: "O"}


def board_to_text(board: list[list[int]]) -> str:
    """
    Convert a 2-D board array (rows × cols, 0=empty, 1=X, 2=O)
    to a human-readable text grid with column indices.

        . . . . . . .
        . . . . . . .
        . X . . . . .
        . X O . . . .
        . X O O . . .
        X X O O X . .
        0 1 2 3 4 5 6
    """
    sym = {0: ".", 1: "X", 2: "O"}
    lines = [" ".join(sym[c] for c in row) for row in board]
    lines.append(" ".join(str(i) for i in range(len(board[0]))))
    return "\n".join(lines)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models emit before their answer."""
    import re as _re
    # Remove everything between <think> and </think> (including the tags)
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    # Also strip common variants
    text = _re.sub(r"<thinking>.*?</thinking>", "", text, flags=_re.DOTALL)
    return text.strip()


def _parse_float(text: str) -> float:
    """Extract the final answer float from LLM response, ignoring thinking blocks."""
    text = _strip_thinking(text)
    text = text.strip().strip("`").strip()
    match = re.search(r"0?\.\d+|\d+\.\d*|\d+", text)
    if match:
        val = float(match.group())
        return max(0.0, min(1.0, val))
    return 0.5  # neutral fallback


def _build_prompt(board: list[list[int]], current_player: int) -> str:
    sym = PLAYER_SYMBOLS[current_player]
    board_str = board_to_text(board)
    legal_cols = [
        c for c in range(len(board[0]))
        if board[0][c] == 0
    ]
    legal_str = ", ".join(str(c) for c in legal_cols)

    return (
        f"You are a Connect Four expert evaluator.\n\n"
        f"Board (rows: top=row 0, bottom=row {len(board)-1}; columns 0-{len(board[0])-1}):\n"
        f"{board_str}\n\n"
        f"It is player {sym}'s turn. Legal moves (columns): {legal_str}\n\n"
        f"Estimate the win probability for player {sym} given this position.\n"
        f"Consider: immediate threats (win or must-block), center control, "
        f"connected pieces, stacking potential.\n\n"
        f"Reply with ONLY a single decimal number between 0.0 (certain loss) "
        f"and 1.0 (certain win). No explanation, no text — just the number."
    )


def _build_move_prompt(board: list[list[int]], current_player: int) -> str:
    sym = PLAYER_SYMBOLS[current_player]
    board_str = board_to_text(board)
    legal_cols = [
        c for c in range(len(board[0]))
        if board[0][c] == 0
    ]
    legal_str = ", ".join(str(c) for c in legal_cols)

    return (
        f"You are a Connect Four expert.\n\n"
        f"Board:\n{board_str}\n\n"
        f"It is player {sym}'s turn. Legal columns: {legal_str}\n\n"
        f"Which single column should player {sym} play to maximize their "
        f"winning chances?\n"
        f"Reply with ONLY a single integer column number from: {legal_str}. "
        f"No explanation, no text — just the number."
    )


# ---------------------------------------------------------------------------
# HTTP helper (no external deps beyond stdlib)
# ---------------------------------------------------------------------------

def _http_post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class AnthropicProvider:
    """Calls Anthropic Claude API."""

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set env var or pass api_key in LLMConfig."
            )

    def _call(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = _http_post(self.BASE_URL, headers, body, self.cfg.timeout)
        return resp["content"][0]["text"]

    def evaluate_position(self, board, current_player: int) -> float:
        return _parse_float(self._call(_build_prompt(board, current_player)))

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = [c for c in range(len(board[0])) if board[0][c] == 0]
        if not legal:
            return None
        text = self._call(_build_move_prompt(board, current_player)).strip()
        try:
            col = int(re.search(r"\d+", text).group())
            if col in legal:
                return col
        except Exception:
            pass
        return None


class DeepSeekProvider:
    """
    Calls DeepSeek's OpenAI-compatible cloud API.
    Models: deepseek-chat, deepseek-reasoner
    """

    BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set env var or pass api_key in LLMConfig."
            )
        if cfg.base_url:
            self.base_url = cfg.base_url
        else:
            self.base_url = self.BASE_URL

    def _call(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = _http_post(self.base_url, headers, body, self.cfg.timeout)
        return resp["choices"][0]["message"]["content"]

    def evaluate_position(self, board, current_player: int) -> float:
        return _parse_float(self._call(_build_prompt(board, current_player)))

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = [c for c in range(len(board[0])) if board[0][c] == 0]
        if not legal:
            return None
        text = self._call(_build_move_prompt(board, current_player)).strip()
        try:
            col = int(re.search(r"\d+", text).group())
            if col in legal:
                return col
        except Exception:
            pass
        return None


class OllamaProvider:
    """
    Calls a local Ollama server (http://localhost:11434 by default).
    Works with any model pulled via `ollama pull <model>`.
    e.g. deepseek-r1:7b, llama3.2:3b, mistral, phi3, etc.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base_url = (cfg.base_url or "http://localhost:11434").rstrip("/")

    def _call(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        resp = _http_post(url, headers, body, self.cfg.timeout)
        return resp.get("response", "")

    def evaluate_position(self, board, current_player: int) -> float:
        return _parse_float(self._call(_build_prompt(board, current_player)))

    def suggest_move(self, board, current_player: int) -> int | None:
        legal = [c for c in range(len(board[0])) if board[0][c] == 0]
        if not legal:
            return None
        raw = self._call(_build_move_prompt(board, current_player))
        text = _strip_thinking(raw).strip()
        try:
            col = int(re.search(r"\d+", text).group())
            if col in legal:
                return col
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------

class CachedProvider:
    """
    Transparent cache wrapper around any provider.
    Identical board states + player → reuse last result.
    Tracks call count, latency, cache stats for reporting.
    """

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
        latency = time.time() - t0

        self._cfg._call_count += 1
        self._cfg._total_latency += latency
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
        latency = time.time() - t0

        self._cfg._call_count += 1
        self._cfg._total_latency += latency
        if self._cfg.cache:
            self._move_cache[key] = col
        return col

    @property
    def stats(self) -> dict:
        cfg = self._cfg
        avg = (cfg._total_latency / cfg._call_count) if cfg._call_count else 0
        return {
            "backend": cfg.backend,
            "model": cfg.model,
            "api_calls": cfg._call_count,
            "cache_hits": cfg._cache_hits,
            "total_latency_s": round(cfg._total_latency, 3),
            "avg_latency_s": round(avg, 3),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(cfg: LLMConfig) -> CachedProvider:
    """Return a ready-to-use CachedProvider for the given config."""
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