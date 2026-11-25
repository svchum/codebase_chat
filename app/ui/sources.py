"""Sources panel: turns the retrieved chunks into cited snippets with GitHub links and surrounding lines."""
from pathlib import Path
from typing import Optional

import streamlit as st

from rag.prompt import cited_indexes

CONTEXT_LINES = 10  


HIGHLIGHT_LANGS = {"python", "javascript", "typescript", "tsx", "java", "go", "c", "cpp", "csharp", "rust",
                   "ruby", "php", "kotlin", "swift", "scala", "sql", "yaml", "toml", "markdown", "rst",
                   "html", "css", "json"}


def permalink(chunk, stats: dict) -> Optional[str]:
    """GitHub link to the exact lines of a chunk (None if the repo isn't on GitHub)."""
    base = stats.get("github_base")
    if not base:
        return None
    if chunk.kind == "commit":
        return f"{base}/commit/{chunk.commit_hash}"
    sha = chunk.commit_hash or stats.get("last_commit")
    if not sha:
        return None
    anchor = f"#L{chunk.start_line}-L{chunk.end_line}" if chunk.end_line > chunk.start_line else f"#L{chunk.start_line}"
    return f"{base}/blob/{sha}/{chunk.file_path}{anchor}"


def surrounding_lines(chunk, repo_root: str):
    """Read a few lines around the chunk from the real file. Returns (text, first_line, last_line) or None."""
    if chunk.kind == "commit" or chunk.language == "pdf" or not repo_root:
        return None
    try:
        lines = (Path(repo_root) / chunk.file_path).read_text(errors="replace").splitlines()
    except OSError:
        return None 
    first = max(1, chunk.start_line - CONTEXT_LINES)
    last = min(len(lines), chunk.end_line + CONTEXT_LINES)
    return "\n".join(lines[first - 1:last]), first, last


def build_sources(engine, slug, hits, answer: str) -> list[dict]:
    """One dict per snippet that was sent to the AI, marking which ones the answer actually cited."""
    stats = engine.index_for(slug).stats()
    cited = cited_indexes(answer, len(hits))
    sources = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        sources.append({
            "n": number,
            "location": chunk.location,
            "link": permalink(chunk, stats),
            "language": chunk.language,
            "kind": chunk.kind,
            "name": chunk.name,
            "content": chunk.content,
            "surrounding": surrounding_lines(chunk, stats["repo_root"]),
            "expanded": hit.expanded,
            "cited": number in cited,
            "found_by": ", ".join(hit.sources),
        })
    return sources


def render_sources(sources: list[dict]) -> None:
    """Show the cited snippets (or, if none were cited, the retrieved ones) as small dropdowns."""
    if not sources:
        return
    cited = [s for s in sources if s["cited"]]
    shown = cited or [s for s in sources if not s["expanded"]]
    label = f"sources / {'cited' if cited else 'retrieved'} {len(shown)} of {len(sources)} snippets"
    st.markdown(f'<div class="cc-label">{label}</div>', unsafe_allow_html=True)
    for source in shown:
        _render_one_source(source)


def _render_one_source(source: dict) -> None:
    label = f"[{source['n']}] {source['location']}"
    if source["name"]:
        label += f"  /  {source['kind']} {source['name']}"
    language = source["language"] if source["language"] in HIGHLIGHT_LANGS else None

    with st.expander(label):
        if source["link"]:
            st.markdown(f"[open on github]({source['link']})")
        tab_names = ["snippet", "surrounding lines"] if source["surrounding"] else ["snippet"]
        tabs = st.tabs(tab_names)
        with tabs[0]:
            st.code(source["content"], language=language)
            st.caption(f"found by: {source['found_by'] or 'n/a'}")
        if source["surrounding"]:
            text, first, last = source["surrounding"]
            with tabs[1]:
                st.caption(f"{source['location'].split(':')[0]} / lines {first}-{last}")
                st.code(text, language=language)