"""Query construction: turn the question and context into LLM messages."""
from __future__ import annotations

import re
from typing import Optional

from .retrieve import Hit

SYSTEM_PROMPT = """You are a senior engineer helping a developer understand a codebase.
You are given numbered context snippets retrieved from the repository (source code, docs, and git commit messages).

Rules:
- Answer ONLY from the provided context. If the context is not enough, say what is missing instead of guessing.
- Explain in clear plain English first, then point to the relevant code.
- Cite every claim with the snippet number in square brackets, e.g. [1] or [2][4]. Cite file paths and line ranges
  in prose when helpful (e.g. `auth/login.py` lines 40-72).
- When you show code, copy it from the context in fenced code blocks with the right language tag. Never invent code,
  file names, or line numbers that are not in the context.
- Snippets marked (surrounding context) are neighbours of a match; use them for understanding, cite them only if needed.
- Be concise. Prefer a short answer with the key code over a long essay."""


def format_context(hits: list[Hit], max_chars: int = 24_000) -> tuple[str, list[Hit]]:
    """Render hits as numbered blocks. Returns (text, hits actually included, in citation order)."""
    blocks, used, total = [], [], 0
    for h in hits:
        c = h.chunk
        label = f"{c.kind} `{c.name}`" if c.name else c.kind
        if c.parent:
            label += f" in class `{c.parent}`"
        tag = " (surrounding context)" if h.expanded else ""
        header = f"[{len(used) + 1}] {c.location} | {c.language} | {label}{tag}"
        body = f"{header}\n```{c.language}\n{c.content}\n```"
        if total + len(body) > max_chars and used:
            break
        blocks.append(body)
        used.append(h)
        total += len(body)
    return "\n\n".join(blocks), used


def build_messages(question: str, hits: list[Hit], history: Optional[list[dict]] = None) -> tuple[list[dict], list[Hit]]:
    """Returns (messages, hits_used). Messages are [{'role','content'}] with a leading system message."""
    context, used = format_context(hits)
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:                  # short rolling chat memory
        msgs.append({"role": m["role"], "content": m["content"]})
    user = (f"Context snippets:\n\n{context}\n\n---\nQuestion: {question}" if used
            else f"No relevant snippets were retrieved.\n\nQuestion: {question}")
    msgs.append({"role": "user", "content": user})
    return msgs, used


def cited_indexes(answer: str, n: int) -> list[int]:
    """1-based snippet numbers cited in the answer, in order of first appearance."""
    seen: list[int] = []
    for m in re.findall(r"\[(\d+)\]", answer):
        i = int(m)
        if 1 <= i <= n and i not in seen:
            seen.append(i)
    return seen
