"""protect server from heavy use."""
import datetime as dt
import time
from typing import Optional

import streamlit as st


def check_limits(cfg, engine, question: str) -> Optional[str]:
    """Return a friendly message if this question should be refused, or None if it is allowed."""
    if len(question) > cfg.max_question_chars:
        return f"Please keep questions under {cfg.max_question_chars} characters."
    if time.time() - st.session_state.last_q < cfg.cooldown_seconds:
        return "One moment, please wait a couple of seconds between questions."
    if cfg.demo_mode:
        if st.session_state.q_count >= cfg.max_questions_per_session:
            return (f"You've reached the limit of {cfg.max_questions_per_session} questions for this session. "
                    "Refresh the page to start a new session, or run CodebaseChat locally for unlimited use.")
        if engine.usage.count(dt.date.today().isoformat()) >= cfg.daily_question_cap:
            return "The demo has reached its daily question limit. Please come back tomorrow."
    return None