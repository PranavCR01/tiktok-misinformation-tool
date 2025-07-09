import streamlit as st

def init_state():
    if "files" not in st.session_state:
        st.session_state.files = []
    if "results" not in st.session_state:
        st.session_state.results = None
