""" Run with:  streamlit run app/streamlit_app.py """
import os
import sys
from pathlib import Path


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  

import streamlit as st  

from ui.limits import check_limits  
from ui.chat import handle_question, render_history  
from ui.home import render_home  
from ui.sidebar import render_sidebar  
from ui.helpers import apply_theme, current_slug, fit_chat_to_screen, get_engine, init_session_state 

st.set_page_config(page_title="CodebaseChat", layout="wide", initial_sidebar_state="expanded")
apply_theme()

engine = get_engine()
cfg = engine.cfg
init_session_state()

settings = render_sidebar(engine, cfg)
slug = current_slug(engine) 

question = st.chat_input("ask anything about the codebase...") or st.session_state.pop("pending", None)


refusal = check_limits(cfg, engine, question) if question else None
if refusal:
    question = None
if question:
    st.session_state.show_home = False 


showing_home = st.session_state.show_home or (not st.session_state.messages and not question)

home_slot = st.empty()  
if showing_home:
    with home_slot.container():
        render_home(engine, cfg, slug)
else:
    home_slot.empty()
    fit_chat_to_screen()
    render_history(engine, slug)

if refusal:
    st.warning(refusal)
if question:
    handle_question(engine, cfg, question, settings, slug)