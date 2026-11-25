""" shows past messages, types out new answers, and runs the question"""
import datetime as dt
import random
import sys
import time
import traceback

import streamlit as st

from rag.llm import friendly_error
from ui.helpers import sync_with_progress
from ui.sources import build_sources, render_sources

# typewriter speed 
TYPE_DELAY = 0.002       
CHARS_PER_TICK = 3       
CATCH_UP_SECONDS = 0.5   


def show_loading(placeholder, text: str = "thinking") -> None:
    """Show a small animated 'thinking...' line. The dots animate in the browser (see chat.css)."""
    placeholder.markdown(
        f'<div class="cc-loading">{text}<span class="cc-dots"><span>.</span><span>.</span><span>.</span></span></div>',
        unsafe_allow_html=True)


def typewriter(pieces, delay: float = TYPE_DELAY):
    """Re-emit streamed text a few characters at a time, with slightly uneven, key-press-like timing.

    `pieces` is whatever the AI has sent so far. Small pieces are typed at the normal speed; a big piece is
    sped up so that it takes at most CATCH_UP_SECONDS.
    """
    for piece in pieces:
        ticks = max(len(piece) / CHARS_PER_TICK, 1)
        per_tick = min(delay * CHARS_PER_TICK, CATCH_UP_SECONDS / ticks)
        for start in range(0, len(piece), CHARS_PER_TICK):
            group = piece[start:start + CHARS_PER_TICK]
            yield group
            pause = per_tick * random.uniform(0.7, 1.3)
            if group.rstrip()[-1:] in (".", "!", "?", ":") or "\n" in group:
                pause += per_tick * 3  
            time.sleep(pause)


def stream_typed(pieces, placeholder) -> str:
    """Type the answer into `placeholder` (replacing the loading line) with a block cursor. Returns the text."""
    text = ""
    for group in typewriter(pieces):
        text += group
        placeholder.markdown(text + "▌")
    placeholder.markdown(text)
    return text


# timing 

def timed(pieces, timings: dict):
    """Pass the AI's text through unchanged, while adding up how long we spent waiting for it (not typing)."""
    iterator = iter(pieces)
    while True:
        started = time.perf_counter()
        try:
            piece = next(iterator)
        except StopIteration:
            return
        timings["waiting"] += time.perf_counter() - started
        timings.setdefault("first_word", time.perf_counter() - timings["start"])
        yield piece


def timing_text(timings: dict) -> str:
    typing = max(timings["total"] - timings["search"] - timings["waiting"], 0.0)
    text = (f"search {timings['search']:.1f}s / waiting for the AI {timings['waiting']:.1f}s "
            f"(first word at {timings.get('first_word', 0.0):.1f}s) / typing {typing:.1f}s / total {timings['total']:.1f}s")
    if timings["retries"]:
        text += f" / {timings['retries']} retries"
    return text


#  messages 

def render_message_text(role: str, text: str) -> None:
    """Show a message. Your own messages get a tiny hidden marker so chat.css can put them on the right."""
    if role == "user":
        st.markdown('<span class="cc-user-msg"></span>', unsafe_allow_html=True)
    st.markdown(text)


def render_history(engine, slug) -> None:
    """Redraw every earlier message (and the sources under each answer)."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_text(message["role"], message["content"])
            if message["role"] == "assistant":
                if message.get("timing"):
                    st.caption(message["timing"])
                render_sources(message.get("sources", []))


def handle_question(engine, cfg, question: str, settings, slug) -> None:
    """Show the question, find relevant code, stream the answer, and save both to the chat history."""
    st.session_state.last_q = time.time()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_message_text("user", question)

    with st.chat_message("assistant"):
        if slug is None or engine.index_for(slug).is_empty():
            answer = ("This demo isn't set up yet." if cfg.demo_mode
                      else "No project is indexed yet. Add a repository from the sidebar first.")
            sources, timing = [], ""
            st.warning(answer)
        else:
            _count_question(engine, cfg)
            if settings.auto_sync and not cfg.demo_mode:
                _auto_update(engine, slug)
            answer, sources, timing = _generate_answer(engine, cfg, question, settings, slug)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "timing": timing})


def _count_question(engine, cfg) -> None:
    """Track usage for the demo limits."""
    st.session_state.q_count += 1
    if cfg.demo_mode:
        engine.usage.bump(dt.date.today().isoformat())


def _auto_update(engine, slug) -> None:
    """If the repo has new commits, re-index only what changed before answering."""
    try:
        if engine.has_new_commits(slug):
            stats = sync_with_progress("New commits detected, updating index...",
                                       lambda progress: engine.auto_sync(slug, progress))
            st.toast("Index updated / " + stats.summary())
    except Exception as error:  
        st.warning(f"Auto-update skipped: {error}")


def _generate_answer(engine, cfg, question: str, settings, slug):
    """Retrieve context, ask the LLM, and type out the reply. Returns (answer_text, sources, timing_text)."""
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    placeholder = st.empty()  
    timings = {"start": time.perf_counter(), "waiting": 0.0, "retries": 0}
    show_loading(placeholder, "searching the codebase")

    def on_retry(attempt, total):
        timings["retries"] = attempt
        show_loading(placeholder, f"model is busy, retrying ({attempt}/{total})")

    try:
        hits = engine.retrieve(question, slug, filters=settings.filters, top_k=settings.top_k, expand=settings.expand)
        timings["search"] = time.perf_counter() - timings["start"]
        if not hits:
            answer = "I couldn't find anything relevant. Try rephrasing, or loosen the filters."
            placeholder.markdown(answer)
            return answer, [], ""

        show_loading(placeholder, "thinking")
        stream, used_hits = engine.answer_stream(
            question, hits, history, on_retry=on_retry,
            on_fallback=lambda model: show_loading(placeholder, f"main model is busy, trying {model}"))
        answer = stream_typed(timed(stream, timings), placeholder)
        timings["total"] = time.perf_counter() - timings["start"]

        timing = "" if cfg.demo_mode else timing_text(timings)   
        if timing:
            st.caption(timing)
        sources = build_sources(engine, slug, used_hits, answer)
        render_sources(sources)
        return answer, sources, timing
    except Exception as error:  
        if friendly_error(error):  
            print(f"[CodebaseChat] {friendly_error(error)}", file=sys.stderr)
        else:  
            print("[CodebaseChat] could not answer the question:", file=sys.stderr)
            traceback.print_exc()
        placeholder.empty()
        answer = _error_message(error, cfg)
        (st.warning if friendly_error(error) else st.error)(answer)
        return answer, [], ""


def _error_message(error: Exception, cfg) -> str:
    """Readable text for known provider errors; details only outside demo mode for unknown ones."""
    friendly = friendly_error(error)
    if friendly:
        return friendly
    if cfg.demo_mode:
        return "Something went wrong while answering. Please try again in a moment."
    return f"Something went wrong: `{error}`"