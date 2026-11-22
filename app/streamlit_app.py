"""Streamlit frontend for the ML Knowledge RAG system.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
import time
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

_REFERENCE_STYLE = """
<style>
.rag-ref {
    display: inline-block;
    padding: 0.05rem 0.4rem;
    border-radius: 0.35rem;
    cursor: help;
    transition: background-color 120ms ease, box-shadow 120ms ease;
}
.rag-ref:hover {
    background-color: rgba(255, 193, 7, 0.28);
    box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.55);
}
</style>
"""

st.markdown(_REFERENCE_STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to the knowledge base…")
def get_backend() -> RAGBackend:
    """Build the backend once per process and reuse it across reruns."""
    return RAGBackend(settings_from_env())


def reference_entries(sources: list) -> list[tuple[str, str]]:
    """Return unique (name, excerpt) pairs, keeping the first chunk per source."""
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for doc in sources:
        metadata = doc.metadata or {}
        source = metadata.get("source")
        name = Path(source).stem if source else "unknown"
        if name in seen:
            continue
        seen.add(name)
        excerpt = " ".join((doc.page_content or "").split())
        entries.append((name, excerpt))
    return entries


def render_references(result: dict) -> None:
    """List numbered references; the retrieved excerpt shows on hover."""
    sources = result.get("sources", [])
    if not sources:
        return

    items = []
    for index, (name, excerpt) in enumerate(reference_entries(sources), 1):
        items.append(
            f'<span class="rag-ref" title="{html.escape(excerpt[:800], quote=True)}">'
            f"[{index}] {html.escape(name)}</span>"
        )
    st.markdown("**References**")
    st.markdown("<br>".join(items), unsafe_allow_html=True)


def render_elapsed(elapsed: float | None) -> None:
    """Show response time at the bottom-right of the message."""
    if elapsed is None:
        return
    st.markdown(
        "<div style='text-align:right;font-size:0.75rem;color:#888;'>"
        f"{elapsed:.2f} s</div>",
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message:
            render_references(message["result"])
        render_elapsed(message.get("elapsed"))


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
        "Ask something you want to know about Machine Learning."
    )

question = st.chat_input("Enter your question…")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and verifying with Self-RAG…"):
                started_at = time.perf_counter()
                result = backend.ask(question)
                elapsed = time.perf_counter() - started_at
        except Exception as exc:
            logger.exception("Query failed")
            st.error(f"Query failed: {exc}")
            messages.append({"role": "assistant", "content": f"⚠️ Query failed: {exc}"})
        else:
            st.markdown(result["answer"])
            render_references(result)
            render_elapsed(elapsed)
            messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "result": result,
                    "elapsed": elapsed,
                }
            )
