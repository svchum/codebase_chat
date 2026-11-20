"""Central configuration from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError: 
    pass


def _optional_int(name: str) -> Optional[int]:
    v = os.getenv(name, "").strip()
    return int(v) if v else None


def _optional_float(name: str) -> Optional[float]:
    v = os.getenv(name, "").strip()
    return float(v) if v else None


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # LLM 
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"))

    llm_timeout: Optional[float] = field(default_factory=lambda: _optional_float("LLM_TIMEOUT_SECONDS")) 
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "1")))
    llm_wait_seconds: float = field(default_factory=lambda: float(os.getenv("LLM_WAIT_SECONDS", "45")))  
    llm_fallback_model: str = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL", "").strip())  
    gemini_thinking_budget: Optional[int] = field(default_factory=lambda: _optional_int("GEMINI_THINKING_BUDGET"))

    #  Embeddings 
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # Storage 
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", ".index")))
    repos_dir: Path = field(default_factory=lambda: Path(os.getenv("REPOS_DIR", ".repos")))

    # Chunking / retrieval
    max_chunk_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CHUNK_CHARS", "2500")))
    overlap_lines: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_LINES", "3")))
    max_commits: int = field(default_factory=lambda: int(os.getenv("MAX_COMMITS", "300")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "8")))
    max_file_bytes: int = 5_000_000
    max_projects: int = field(default_factory=lambda: int(os.getenv("MAX_PROJECTS", "3")))  # indexes kept

    # Demo mode
    demo_mode: bool = field(default_factory=lambda: _bool("DEMO_MODE", True))
    max_questions_per_session: int = field(default_factory=lambda: int(os.getenv("MAX_QUESTIONS_PER_SESSION", "15")))
    daily_question_cap: int = field(default_factory=lambda: int(os.getenv("DAILY_QUESTION_CAP", "150")))
    max_question_chars: int = field(default_factory=lambda: int(os.getenv("MAX_QUESTION_CHARS", "1000")))
    cooldown_seconds: float = field(default_factory=lambda: float(os.getenv("COOLDOWN_SECONDS", "2")))


def load_config() -> Config:
    return Config()