"""title, description, how-it-works boxes, and a guide dropdown"""
from dataclasses import dataclass
from html import escape

import streamlit as st

from ui.helpers import center_cover, set_pending

INTRO = ("Add a repository to CodebaseChat and ask questions, the way you would ask someone to explain the code to you."
         "It reads the source code, documentation, and commit history, finds the parts "
         "that matter, and answers with highlighted code, file names, line numbers, and links back to GitHub, "
         "so you can always check where an answer came from.")

STEPS = [
    ("1 / Add a repo", "A GitHub URL or local folder, indexed once, then kept up to date."),
    ("2 / Ask anything", "Ask anything about the codebase, and get an answer with links to the source."),
    ("3 / Verify", "Every answer lists its sources, with surrounding lines and links."),
]

TIPS_LEFT = [
    "**Be specific.** Name the function or file you care about.",
]
TIPS_RIGHT = [
    "**Filter** by language, folder, or chunk type in the sidebar.",
    "**Ask follow-ups.** The chat remembers your last few messages.",
]


@dataclass(frozen=True)
class Capability:
    title: str
    description: str
    examples: tuple[str, str]


CAPABILITIES = [
    Capability("Understand a codebase", "A plain-English tour of the architecture and main parts.",
               ("Give me an overview of this project.", "What happens when the app starts?")),
    Capability("Find things fast", "Jump to the file and lines you're looking for.",
               ("Where is user authentication handled?", "Show me the logic that processes API requests.")),
    Capability("Spot discrepancies", "Check that the README and docs match the code.",
               ("Does the README setup match the code?", "Are any documented features missing from the code?")),
    Capability("Trace the history", "Ask what changed, when, and why. Commits are indexed.",
               ("What changed recently, and why?", "Which commits touched the login flow?")),
    Capability("Onboard and prepare", "Learn about a new repo or prep for a review.",
               ("Explain this repo to a new teammate.", "How would I add a new API endpoint?")),
    Capability("Review and sanity-check", "Find loose ends and risky shortcuts.",
               ("Are there any TODOs or risky shortcuts?", "Which code has no obvious tests?")),
]


def render_hero() -> None:
    """The typing title and the tagline under it."""
    st.markdown('<div class="cc-hero"><h1 class="cc-title">CodebaseChat</h1>'
                '<p class="cc-sub">Talk to any GitHub repo.</p></div>', unsafe_allow_html=True)


def render_home(engine, cfg, slug) -> None:
    """Draw the whole cover page."""
    center_cover()
    render_hero()
    st.markdown(f'<div class="cc-intro">{INTRO}</div>', unsafe_allow_html=True)
    _render_status(engine, cfg, slug)
    _render_steps()
    _render_guide()


def _render_status(engine, cfg, slug) -> None:
    project = engine.projects.get(slug) if slug else None
    if project and not engine.index_for(slug).is_empty():
        stats = engine.index_for(slug).stats()
        text = (f"[ready] chatting with <b>{escape(project.name)}</b> / {stats['files']} files / "
                f"{stats['commits']} commits indexed")
        if len(engine.projects.list()) > 1:
            text += " / switch projects in the sidebar"
    elif cfg.demo_mode:
        text = "[not ready] no project index found. Build one with scripts/reindex.py (see README)"
    else:
        text = "[not ready] paste a GitHub URL in the sidebar and click <b>Index / update</b>"
    st.markdown(f'<div class="cc-status">{text}</div>', unsafe_allow_html=True)


def _render_steps() -> None:
    boxes = "".join(f'<div class="cc-step"><b>{title}</b><small>{text}</small></div>' for title, text in STEPS)
    st.markdown(f'<div class="cc-steps">{boxes}</div>', unsafe_allow_html=True)


def _render_guide() -> None:
    """The collapsible guide: example questions you can click, plus a few tips."""
    with st.expander("What can I do here?"):
        _render_capability_cards()
        st.markdown('<div class="cc-label">tips</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        left.markdown("\n".join(f"- {tip}" for tip in TIPS_LEFT))
        right.markdown("\n".join(f"- {tip}" for tip in TIPS_RIGHT))


def _render_capability_cards() -> None:
    """Cards in rows of three; each example button asks that question when clicked."""
    per_row = 3
    for start in range(0, len(CAPABILITIES), per_row):
        columns = st.columns(per_row)
        for column, capability in zip(columns, CAPABILITIES[start:start + per_row]):
            with column, st.container(border=True):
                st.markdown(f'<div class="cc-card-t">{capability.title}</div>'
                            f'<div class="cc-card-d">{capability.description}</div>', unsafe_allow_html=True)
                for i, example in enumerate(capability.examples):
                    st.button(example, key=f"ex_{capability.title}_{i}", on_click=set_pending,
                              args=(example,), use_container_width=True)