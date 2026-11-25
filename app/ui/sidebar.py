"""repository box, project switcher, filters, and retrieval settings."""
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

from rag.index import Filters
from rag.projects import make_slug
from ui.helpers import clear_chat, current_slug, go_home, sync_with_progress


@dataclass
class ChatSettings:
    """What the sidebar tells the chat to do. Defaults apply in demo mode, where most controls are hidden."""
    filters: Filters = field(default_factory=Filters)
    top_k: int = 8
    expand: bool = True
    auto_sync: bool = False


def render_sidebar(engine, cfg) -> ChatSettings:
    """Draw the sidebar and return the visitor's settings."""
    settings = ChatSettings(top_k=cfg.top_k)

    with st.sidebar:
        st.button("CodebaseChat", key="home_btn", on_click=go_home, help="Back to the home page")

        if cfg.demo_mode:
            _demo_info(cfg)
        else:
            settings.auto_sync = _repo_section(engine)

        slug = _project_section(engine)
        _index_stats(engine, cfg, slug)

        if not cfg.demo_mode:
            if slug:
                settings.filters = _filters_section(engine, slug)
            settings.top_k, settings.expand = _retrieval_section(cfg)

        if st.button("New chat", use_container_width=True):
            clear_chat()
            st.rerun()
        if not cfg.demo_mode:
            st.caption(f"llm: {cfg.llm_provider} / embeddings: {cfg.embed_model}")

    return settings


def _demo_info(cfg) -> None:
    st.caption("ask questions about the projects loaded in this demo.")
    left = max(cfg.max_questions_per_session - st.session_state.q_count, 0)
    st.progress(left / max(cfg.max_questions_per_session, 1), text=f"{left} questions left this session")


def _repo_section(engine) -> bool:
    """Repo input + index buttons (personal mode only). Returns whether auto-update is switched on."""
    st.markdown('<div class="cc-label">repository</div>', unsafe_allow_html=True)
    source = st.text_input("GitHub URL or local path", value=st.session_state.repo_source,
                           placeholder="https://github.com/owner/repo", label_visibility="collapsed")
    col_update, col_rebuild = st.columns(2)
    do_update = col_update.button("Index / update", type="primary", use_container_width=True, disabled=not source.strip())
    do_rebuild = col_rebuild.button("Full rebuild", use_container_width=True, disabled=not source.strip())

    if do_update or do_rebuild:
        st.session_state.repo_source = source
        try:
            slug, _ = make_slug(source)
            sync_with_progress("Indexing repository...", lambda progress: engine.index_repo(source, progress, full=do_rebuild))
            st.session_state.project = slug 
            clear_chat()
            st.rerun()
        except Exception as error:  
            st.error(f"Indexing failed: {error}")

    st.caption(f"the last {engine.projects.max_projects} projects indexed are kept and shown in the demo.")
    return st.toggle("Auto-update on new commits", value=True,
                     help="Before each question, check for new commits and re-index only what changed.")


def _project_section(engine) -> Optional[str]:
    """Choose which of the kept projects to chat with. Returns the chosen project's slug."""
    projects = engine.projects.list()
    st.markdown('<div class="cc-label">project</div>', unsafe_allow_html=True)
    if not projects:
        st.caption("no projects indexed yet")
        return None
    names = {p.slug: p.name for p in projects}
    current_slug(engine)  
    st.radio("Project", list(names), key="project", format_func=names.get,
             label_visibility="collapsed", on_change=clear_chat)
    return st.session_state.project


def _index_stats(engine, cfg, slug: Optional[str]) -> None:
    if not slug:
        return
    stats = engine.index_for(slug).stats()
    if not stats["chunks"]:
        return
    st.caption(f"{stats['files']} files / {stats['chunks']} chunks / {stats['commits']} commits")
    if not cfg.demo_mode:
        st.caption(f"indexed {stats['indexed_at']} / commit {(stats['last_commit'] or 'n/a')[:8]}")


def _filters_section(engine, slug: str) -> Filters:
    with st.expander("Filters"):
        st.caption("Narrow which parts of the repo are searched.")
        options = engine.index_for(slug).facets()
        languages = st.multiselect("Language / file type", options["languages"])
        kinds = st.multiselect("Chunk type", options["kinds"],
                               help="function, class, method, module, doc, window, commit")
        name = st.text_input("Function / class name contains")
        path = st.text_input("File path contains", placeholder="auth/  or  .test.")
    return Filters(languages=languages, kinds=kinds, name_contains=name, path_contains=path)


def _retrieval_section(cfg) -> tuple[int, bool]:
    with st.expander("Retrieval"):
        top_k = st.slider("Snippets per answer", 3, 15, cfg.top_k,
                          help="How many of the best-matching chunks are given to the AI as context. "
                               "More = broader answers but slower and noisier.")
        expand = st.toggle("Include surrounding chunks", value=True,
                           help="Also give the AI the chunks next to each top match and the class it belongs to.")
    return top_k, expand