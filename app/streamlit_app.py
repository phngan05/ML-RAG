"""Streamlit frontend for the ML Knowledge RAG system.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts the script's folder on sys.path; expose the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend import (  # noqa: E402
    RAGBackend,
    settings_from_env,
)
from src.utils.helpers import get_logger  # noqa: E402

logger = get_logger(__name__)

st.set_page_config(page_title="ML Knowledge RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Connecting to the knowledge base…")
def get_backend() -> RAGBackend:
    """Build the backend once per process and reuse it across reruns."""
    return RAGBackend(settings_from_env())


def render_result_details(result: dict) -> None:
    """Show retrieved sources and the Self-RAG grading trace for an answer."""
    sources = result.get("sources", [])

    with st.expander(f"Sources ({len(sources)})"):
        if not sources:
            st.caption("No documents were retrieved for this question.")
        for index, doc in enumerate(sources, 1):
            metadata = doc.metadata or {}
            st.markdown(
                f"**[{index}] {metadata.get('doc_type', 'unknown')}** · "
                f"`{metadata.get('source', 'unknown')}`"
            )
            st.caption(doc.page_content.strip()[:600] + "…")

    with st.expander("Self-RAG trace"):
        cols = st.columns(3)
        cols[0].metric("Translation", result.get("translation_method", "-"))
        cols[1].metric("Attempts", result.get("attempts", "-"))
        cols[2].metric("Sources", len(sources))
        if result.get("warning"):
            st.warning(result["warning"])
        grading_log = result.get("grading_log", [])
        if grading_log:
            st.dataframe(grading_log, hide_index=True)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message:
            render_result_details(message["result"])


# ---------------------------------------------------------------------- #
# Backend
# ---------------------------------------------------------------------- #

settings = settings_from_env()

st.title("📚 Machine Learning Knowledge RAG")

missing = settings.missing_credentials()
if missing:
    st.info(
        "Set " + " and ".join(f"`{name}`" for name in missing)
        + " in your `.env` file (see `.env.example`) and reload the app."
    )
    st.stop()

try:
    backend = get_backend()
except Exception as exc:
    logger.exception("Backend initialisation failed")
    st.error(f"Could not initialise the RAG backend: {exc}")
    st.stop()

# ---------------------------------------------------------------------- #
# Chat
# ---------------------------------------------------------------------- #

messages = st.session_state.setdefault("messages", [])

_, clear_column = st.columns([5, 1])
with clear_column:
    if st.button("Clear chat", disabled=not messages):
        st.session_state["messages"] = []
        st.rerun()

for message in messages:
    render_message(message)

if not messages:
    st.info(
        "Ask something about Machine Learning.”"
    )

question = st.chat_input("Enter your question…")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and verifying with Self-RAG…"):
                result = backend.ask(question)
        except Exception as exc:
            logger.exception("Query failed")
            st.error(f"Query failed: {exc}")
            messages.append({"role": "assistant", "content": f"⚠️ Query failed: {exc}"})
        else:
            st.markdown(result["answer"])
            render_result_details(result)
            messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "result": result,
                }
            )
