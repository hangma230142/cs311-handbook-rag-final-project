"""
Giao diện web (Streamlit) cho chatbot CS Handbook (Task 4).
Chạy:  streamlit run app.py
Dùng lại nguyên phần RAG trong run_rag.py (cùng index, cùng prompt, cùng ngưỡng).
"""

import time

import streamlit as st
from langchain_ollama import OllamaEmbeddings

import run_rag as rag

st.set_page_config(page_title="CS Handbook Assistant", layout="centered")

EXAMPLES = [
    "What is CS 203 about?",
    "How many advanced courses must a CS student take?",
    "Which math course is required for the CS major?",
    "What are the learning outcomes of the CS program?",
]
MODELS = list(dict.fromkeys([rag.LLM_MODEL, "qwen2.5:3b", "qwen2.5:1.5b"]))


@st.cache_resource(show_spinner="Loading the handbook index...")
def load_db():
    return rag.get_db(OllamaEmbeddings(model=rag.EMBED_MODEL, base_url=rag.OLLAMA_URL))


@st.cache_resource
def load_llm(model: str):
    return rag.make_llm(model)


def render_assistant(msg: dict):
    st.write(msg["answer"])
    if msg["fallback"]:
        st.caption("This question looks unrelated to the handbook, so the model was not asked.")
    st.caption(f"{msg['seconds']:.1f} s · {msg['model']}")
    pages = ", ".join(sorted({str(s["page"]) for s in msg["sources"]}, key=int))
    with st.expander(f"Sources (pages {pages})"):
        for s in msg["sources"]:
            st.markdown(f"**Page {s['page']}** · relevance {s['score']}")
            st.text(s["content"])


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Answer model", MODELS, help="The model must already be pulled in Ollama.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

# ---------- header ----------
st.title("CS Major Handbook Assistant")
st.caption(
    "Ask about the B.S. Computer Science handbook. Answers come only from the handbook, with the pages used. "
    "Each question is answered on its own: earlier messages are kept as history but are not sent to the model."
)

# ---------- backend ----------
try:
    db = load_db()
except (Exception, SystemExit) as e:
    st.error("Could not load the handbook index. Check that Ollama is running and the PDF is mounted. "
             f"Details: {e}")
    st.stop()

# ---------- history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        else:
            render_assistant(m)

# ---------- new question ----------
prompt = st.chat_input("Ask a question about the CS handbook") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook..."):
            t0 = time.time()
            try:
                r = rag.ask(prompt, db, load_llm(model))
                reply = {"role": "assistant", "answer": r["answer"], "sources": r["sources"],
                         "fallback": r["fallback"], "seconds": time.time() - t0, "model": model}
            except Exception as e:
                reply = {"role": "assistant",
                         "answer": "I could not reach the language model. Check that the Ollama service "
                                   f"is running at {rag.OLLAMA_URL}. ({type(e).__name__})",
                         "sources": [], "fallback": False, "seconds": time.time() - t0, "model": model}
        if reply["sources"]:
            render_assistant(reply)
        else:
            st.error(reply["answer"])
    st.session_state.messages.append(reply)
