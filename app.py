"""
METU Student Assistant — Streamlit Chat UI (Step 5)

Note that this script will run everytime the user makes a change on the webpage such as hitting enter or typing a new letter.
We only cache the MetuAssistant class so that we dont have to reload the vector db and LLM everytime the user makes a change.

"""

from datetime import datetime
import sys
from pathlib import Path

import streamlit as st

# Make sure rag package imports work when Streamlit runs from project root
PROJECT_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path: # sys.path is a built-in list of all directories Python searches when we import a module
    sys.path.insert(0, str(PROJECT_FOLDER))

from rag.engine import MetuAssistant


def messages_to_txt(messages: list[dict]) -> str:
    """Plain-text export of the current chat (including source links)."""
    lines = [
        "METU Student Assistant — chat export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 40,
        "",
    ]
    for message in messages:
        role = (message.get("role") or "unknown").upper()
        lines.append(f"{role}:")
        lines.append((message.get("content") or "").strip())
        sources = message.get("sources") or []
        if role == "ASSISTANT" and sources:
            lines.append("")
            lines.append("Sources:")
            for source in sources:
                title = source.get("title") or "Source"
                url = source.get("url") or ""
                lines.append(f"- {title} — {url}" if url else f"- {title}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="METU Student Assistant",
    page_icon="🧑‍🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        --metu-red: #E31837;
        --metu-gray: #717073;
    }
    /* Hide sidebar entirely */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Top toolbar accent */
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, var(--metu-red) 0%, var(--metu-red) 100%);
    }
    [data-testid="stHeader"] * {
        color: #ffffff !important;
    }
    /* Title emphasis */
    h1 {
        color: var(--metu-red) !important;
        letter-spacing: -0.02em;
    }
    /* Links (Sources, etc.) */
    a, a:visited {
        color: var(--metu-red) !important;
    }
    /* Outer chat-input ring: always METU red (not only on focus) */
    [data-testid="stChatInput"] > div {
        border-color: var(--metu-red) !important;
        box-shadow: 0 0 0 1px var(--metu-red) !important;
    }
    /* Inner textarea: no red ring */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("METU Student Assistant")
st.caption(
    "Welcome! Ask me anything related to METU Ankara.\n"
    "Note: Always confirm the validity of the information you find here on official METU pages."
)


# ---------------------------------------------------------------------------
# Load the RAG engine once (cached so we don't reload the model every message)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading assistant...")
def get_assistant():
    return MetuAssistant()


try:
    assistant = get_assistant()
except Exception as error:
    st.error(str(error))
    st.stop()


# ---------------------------------------------------------------------------
# Chat history (stored in Streamlit session)
# ---------------------------------------------------------------------------

if "messages" not in st.session_state: #checks if a message key already exists in streamlit's built in memory dictionary
    st.session_state.messages = []

# Enable once there is at least one message in the thread
has_messages = len(st.session_state.messages) >= 1

col_new, col_export, _ = st.columns([1, 1, 3])
with col_new:
    if st.button("New chat", use_container_width=True, disabled=not has_messages):
        st.session_state.messages = []
        st.rerun()
with col_export:
    st.download_button(
        label="Export Chat",
        data=messages_to_txt(st.session_state.messages) if has_messages else "",
        file_name=f"metu_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not has_messages,
    )


# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]): #auto etermines the icon to be displayed (user or assistant)
        st.markdown(message["content"]) #renders the actual text of the message as markdown in the chat bubble
        # Show sources under assistant replies
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"): #creates a drop down list for the sources
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})") #interprets the markdown inside and then prints it as a clickable link


# ---------------------------------------------------------------------------
# New user question
# ---------------------------------------------------------------------------

question = st.chat_input("Ask in English or Turkish...")

if question:
    # Prior turns only (current question is not in history yet)
    history = list(st.session_state.messages)

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get RAG answer (short-term memory via history + query rewrite)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = assistant.ask(question, history=history)

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- [{source['title']}]({source['url']})")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
    # Refresh so New chat / Export enable immediately after the first message
    st.rerun()
