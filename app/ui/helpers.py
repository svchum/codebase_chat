"""Small shared helpers for the engine, session setup, the theme, and indexing progress."""
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

from rag.config import load_config
from rag.pipeline import Engine

# Engine and session state

def _preload_search_model(engine: Engine) -> None:
    """Load the search model in the background, so the page appears right away and the first question isn't slow.
    If a question arrives before this finishes, it simply waits for it."""
    try:
        engine.embedder.embed_query("warm up")
    except Exception as error: 
        print(f"[CodebaseChat] could not preload the search model: {error}")


@st.cache_resource(show_spinner=False)  
def get_engine() -> Engine:
    """Create the Engine once and reuse it on every rerun."""
    engine = Engine.create(load_config())
    threading.Thread(target=_preload_search_model, args=(engine,), daemon=True).start()
    return engine


def init_session_state() -> None:
    """Give a new visitor their starting values. Existing values are never overwritten."""
    st.session_state.setdefault("messages", [])     
    st.session_state.setdefault("repo_source", "")   
    st.session_state.setdefault("q_count", 0)       
    st.session_state.setdefault("last_q", 0.0)       
    st.session_state.setdefault("show_home", False)  
    st.session_state.setdefault("project", None)     


def current_slug(engine) -> Optional[str]:
    """The project this visitor is chatting with. Falls back to the newest one if the saved choice is gone."""
    slugs = [p.slug for p in engine.projects.list()]
    if st.session_state.project not in slugs:
        st.session_state.project = slugs[0] if slugs else None
    return st.session_state.project


def clear_chat() -> None:
    """Start a fresh conversation (used by the New chat button and when switching projects)."""
    st.session_state.messages = []


def go_home() -> None:
    """Called when the CodebaseChat logo is clicked: show the cover page (the chat is kept)."""
    st.session_state.show_home = True


def set_pending(text: str) -> None:
    """Called when an example button is clicked: remember the question so the main script asks it."""
    st.session_state.pending = text



# Theme: the CSS files in the styles/ folder

UI_DIR = Path(__file__).parent
STYLE_FILES = [  
    "styles/base.css",
    "styles/cover.css",
    "styles/chat.css",
    "styles/widgets.css",
]



def load_css() -> str:
    """Read every CSS file and join them into one string.

    "utf-8-sig" quietly drops the invisible byte-order mark that some Windows editors add to the start of a
    file. Left in, it can make the browser ignore the first rule in the file (in base.css, that is every color and font).
    """
    return "\n".join((UI_DIR / name).read_text(encoding="utf-8-sig") for name in STYLE_FILES)


def apply_theme() -> None:
    """Load the CSS into the page. Call once at the top of every run."""
    css = load_css()
    if "--bg:" not in css or "--font-body:" not in css:
        st.warning("The theme did not load completely: base.css looks empty or incomplete. See app/ui/styles/.")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def fit_chat_to_screen() -> None:
    """Let the chat use the whole width of the page (your questions on the right, answers on the left)."""
    st.markdown(
        "<style>.block-container, [data-testid='stBottomBlockContainer'] { max-width: 100% !important; "
        "margin: 0 !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; }</style>",
        unsafe_allow_html=True)


def center_cover(max_width_px: int = 1000) -> None:
    """Keep the cover page in a centered column so it stays tidy on wide screens."""
    st.markdown(
        "<style>.block-container, [data-testid='stBottomBlockContainer'] "
        f"{{ max-width: {max_width_px}px !important; margin-left: auto !important; margin-right: auto !important; }}</style>",
        unsafe_allow_html=True)



# Indexing progress

def sync_with_progress(label: str, run_sync):
    """Run `run_sync(progress_callback)` inside a status box with a progress bar. Returns its stats."""
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0)
        line = st.empty()

        def progress(stage, done, total):
            line.write(f"{stage}: {done}" + (f" / {total}" if total else ""))
            if total:
                bar.progress(min(done / total, 1.0))

        stats = run_sync(progress)
        status.update(label=f"done / {stats.summary()}", state="complete", expanded=False)
    return stats