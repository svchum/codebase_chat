"""LLM abstraction built on LangChain chat models."""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from .config import Config


def get_llm(cfg: Config, model: Optional[str] = None):
    """The chat model for the configured provider. `model` overrides the model name (used for the backup)."""
    p = cfg.llm_provider
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file (see .env.example).")

        settings = dict(model=model or cfg.gemini_model, google_api_key=key, temperature=0.1, max_retries=cfg.llm_max_retries)
        if cfg.llm_timeout is not None:  
            settings["timeout"] = cfg.llm_timeout
        if cfg.gemini_thinking_budget is not None:  
            try:
                return ChatGoogleGenerativeAI(**settings, thinking_budget=cfg.gemini_thinking_budget)
            except Exception: 
                pass
        return ChatGoogleGenerativeAI(**settings)
    if p == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return ChatOpenAI(model=model or cfg.openai_model, temperature=0.1)
    if p in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return ChatAnthropic(model=model or cfg.anthropic_model, temperature=0.1, max_tokens=2048)
    raise ValueError(f"Unknown LLM_PROVIDER '{p}'. Use gemini, openai or anthropic.")


def _text(content) -> str:
    """Chat chunks may be a str or a list of content parts depending on provider/version."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content
                       if isinstance(p, (str, dict)))
    return ""


BUSY_MARKERS = ("503", "unavailable", "overloaded", "high demand")
LIMIT_MARKERS = ("429", "resource_exhausted", "quota", "rate limit")


class LlmTimeout(Exception):
    """The AI went silent for too long (usually a stalled network connection)."""


def is_busy_error(error: Exception) -> bool:
    """True for temporary 'the model is overloaded' errors. These usually clear up if you try again."""
    text = str(error).lower()
    return any(marker in text for marker in BUSY_MARKERS)


def friendly_error(error: Exception) -> Optional[str]:
    """A short, readable message for common provider errors (None if the error isn't a known kind)."""
    if isinstance(error, LlmTimeout):
        return "The AI took too long to respond. Please try again in a moment."
    text = str(error).lower()
    if is_busy_error(error):
        return "The AI model is busy right now (high demand on the provider's side). Please try again in a moment."
    if any(marker in text for marker in LIMIT_MARKERS):
        return "The AI usage limit has been reached for now. Please try again later."
    return None


def _chunks_with_deadline(llm, messages: list[dict], wait_seconds: float) -> Iterator[str]:
    """Yield the AI's text pieces, but give up if nothing arrives for `wait_seconds`.

    The request runs in a background thread. A stalled connection can otherwise block forever, and the
    timeout settings of the AI libraries differ between versions, so this one is our own.
    """
    pieces: queue.Queue = queue.Queue()
    finished = object()

    def worker():
        try:
            for chunk in llm.stream([(m["role"], m["content"]) for m in messages]):
                text = _text(chunk.content)
                if text:
                    pieces.put(text)
            pieces.put(finished)
        except BaseException as error:  
            pieces.put(error)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            item = pieces.get(timeout=wait_seconds)
        except queue.Empty:
            raise LlmTimeout(f"The AI did not respond within {wait_seconds:.0f} seconds.") from None
        if item is finished:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def stream_answer(llm, messages: list[dict], retries: int = 3, wait: float = 2.0,
                  on_retry: Optional[Callable[[int, int], None]] = None, wait_seconds: float = 45.0) -> Iterator[str]:
    """Yield answer text incrementally.

    If the model is busy (a 503), or the connection stalls once, and no text has arrived yet, wait and try again.
    `on_retry(attempt, retries)` is called before each wait so the UI can show a message.
    `wait_seconds` is how long the AI may stay silent before we give up.
    """
    attempt = 0
    while True:
        started = False
        try:
            for text in _chunks_with_deadline(llm, messages, wait_seconds):
                started = True
                yield text
            return
        except Exception as error: 
            retryable = is_busy_error(error) or (isinstance(error, LlmTimeout) and attempt == 0)
            if started or attempt >= retries or not retryable:
                raise
            attempt += 1
            if on_retry:
                on_retry(attempt, retries)
            time.sleep(wait * 2 ** (attempt - 1))